"""Unit and process-level tests of the PROPOSED RCA reference kernel (issues #42/#43).

``wavelock.rca_reference`` is a proposed reference mechanism, not preexisting production
behaviour.  These tests pin its fail-closed memory format and its guarded action path; the trace
conformance against the Lean model is ``tests/test_rca_formal_conformance.py``.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from wavelock.rca_reference import (STATUS, GuardedAgent, HmacOperatorVerifier, MemoryCorrupt,
                                    Policy, anchored_deviation, l1, load, operator_credential,
                                    path_drift, predict_breach, step_deviations)
from wavelock.rca_reference import memory as M

ROOT = Path(__file__).resolve().parents[1]
KEY = b"operator-key-for-tests-000000000"
VERIFIER = HmacOperatorVerifier(KEY)


@pytest.fixture
def agent(tmp_path):
    return GuardedAgent.provision(tmp_path, agent="agent-1", baseline=[0, 0],
                                  policy=Policy.make(0, 5, ["rm"]), verifier=VERIFIER)


def lines(path):
    return (Path(path) / "safety-memory.jsonl").read_bytes().split(b"\n")[:-1]


def write_lines(path, ls):
    (Path(path) / "safety-memory.jsonl").write_bytes(b"".join(x + b"\n" for x in ls))


def test_status_is_labelled_proposed():
    assert STATUS == "PROPOSED_REFERENCE"


def test_guarded_execution_and_latch(agent):
    done = []
    assert agent.request("move", done.append) and done == ["move"]
    assert not agent.request("rm", done.append)
    assert agent.observe(1, [2, 2]) and not agent.view.locked
    assert agent.observe(2, [3, 2]) and agent.view.locked
    assert not agent.request("move", done.append) and done == ["move"]


def test_effect_runs_only_after_durable_intent(agent):
    seen = []

    def effect(action):
        seen.append(json.loads(lines(agent.directory)[-1])["kind"])
    agent.request("move", effect)
    assert seen == ["execute"]


def test_failed_intent_write_never_runs_effect(agent, monkeypatch):
    done = []

    def fail(data):
        raise OSError("disk full")
    monkeypatch.setattr(agent, "_write_line", fail)
    with pytest.raises(OSError):
        agent.request("move", done.append)
    assert done == []


def test_restart_recovers_latch(agent, tmp_path):
    agent.observe(1, [9, 0])
    again = GuardedAgent(tmp_path, verifier=VERIFIER)
    assert again.view.locked and not again.request("move", lambda a: None)


def test_reset_requires_bound_credential_and_resolved_deviation(agent):
    agent.observe(1, [9, 0])
    early = operator_credential(KEY, "agent-1", agent.view.length, agent.view.head)
    assert not agent.reset(early)                       # deviation still at the threshold
    agent.observe(2, [1, 0])
    assert not agent.reset(early)                       # bound to an older prefix
    assert not agent.reset(operator_credential(b"x" * 32, "agent-1", agent.view.length,
                                               agent.view.head))   # wrong key
    good = operator_credential(KEY, "agent-1", agent.view.length, agent.view.head)
    assert agent.reset(good) and not agent.view.locked
    agent.observe(3, [9, 0])
    agent.observe(4, [0, 0])
    assert not agent.reset(good)                        # no replay


def test_policy_only_tightens(agent):
    assert not agent.update_policy(Policy.make(1, 9, ["rm"]))   # higher threshold
    assert not agent.update_policy(Policy.make(1, 5, []))       # unprohibits rm
    assert not agent.update_policy(Policy.make(2, 4, ["rm"]))   # skips a version
    assert agent.update_policy(Policy.make(1, 4, ["rm", "sh"]))
    assert not agent.request("sh", lambda a: None)


def test_tightening_below_current_deviation_latches(agent):
    agent.observe(1, [2, 2])
    assert agent.update_policy(Policy.make(1, 4, ["rm"])) and agent.view.locked


@pytest.mark.parametrize("edit", [
    lambda rec: rec.update(i=rec["i"] + 1),
    lambda rec: rec.update(prev="0" * 64),
    lambda rec: rec.update(values=[1.0, 0]),
    lambda rec: rec.update(values=[True, 0]),
    lambda rec: rec.update(seq=0),
    lambda rec: rec.update(extra=1),
    lambda rec: rec.update(kind="unknown"),
])
def test_edited_records_fail_closed(agent, tmp_path, edit):
    agent.observe(1, [1, 1])
    ls = lines(tmp_path)
    rec = json.loads(ls[-1])
    edit(rec)
    write_lines(tmp_path, ls[:-1] + [json.dumps(rec, sort_keys=True, separators=(",", ":")).encode()])
    with pytest.raises(MemoryCorrupt):
        GuardedAgent(tmp_path, verifier=VERIFIER)


def test_non_canonical_and_duplicate_keys_fail_closed(agent, tmp_path):
    agent.observe(1, [1, 1])
    ls = lines(tmp_path)
    write_lines(tmp_path, ls[:-1] + [json.dumps(json.loads(ls[-1])).encode()])   # spaces
    with pytest.raises(MemoryCorrupt):
        GuardedAgent(tmp_path, verifier=VERIFIER)
    write_lines(tmp_path, ls[:-1] + [ls[-1][:-1] + b',"seq":1}'])                # duplicate key
    with pytest.raises(MemoryCorrupt):
        GuardedAgent(tmp_path, verifier=VERIFIER)


def test_inadmissible_record_written_by_another_writer_is_detected(agent, tmp_path):
    """Post-hoc detection, not intervention: an execute record after the latch makes the memory
    unloadable, but its effect (if any) already happened."""
    agent.observe(1, [9, 0])
    view = agent.view
    rec = {"v": 1, "i": view.length, "prev": view.head, "kind": "execute", "action": "move"}
    write_lines(tmp_path, lines(tmp_path) + [M.canonical(rec)])
    with pytest.raises(MemoryCorrupt, match="inadmissible execute"):
        GuardedAgent(tmp_path, verifier=VERIFIER)


def test_missing_torn_and_rolled_back_memory(agent, tmp_path):
    agent.observe(1, [9, 0])
    anchor = (agent.view.length - 1, agent.view.head)
    ls = lines(tmp_path)
    write_lines(tmp_path, ls[:1])
    assert not GuardedAgent(tmp_path, verifier=VERIFIER).view.locked   # rollback loses the latch
    with pytest.raises(MemoryCorrupt, match="rolled back"):
        GuardedAgent(tmp_path, verifier=VERIFIER, anchor=anchor)
    write_lines(tmp_path, ls)
    (tmp_path / "safety-memory.jsonl").write_bytes((tmp_path / "safety-memory.jsonl").read_bytes() + b'{"v"')
    with pytest.raises(MemoryCorrupt, match="torn"):
        GuardedAgent(tmp_path, verifier=VERIFIER)
    os.unlink(tmp_path / "safety-memory.jsonl")
    with pytest.raises(MemoryCorrupt, match="not provisioned"):
        GuardedAgent(tmp_path, verifier=VERIFIER)


def test_provision_never_overwrites(agent, tmp_path):
    with pytest.raises(FileExistsError):
        GuardedAgent.provision(tmp_path, agent="agent-2", baseline=[0], policy=Policy.make(0, 1),
                               verifier=VERIFIER)


def test_load_rejects_bad_genesis():
    rec = M.genesis_record("a", [0], Policy.make(0, 1))
    rec["kind"] = "observe"
    with pytest.raises(MemoryCorrupt):
        load(M.canonical(rec) + b"\n", verifier=VERIFIER)
    with pytest.raises(ValueError):
        M.genesis_record("a", [], Policy.make(0, 1))
    with pytest.raises(ValueError):
        Policy.make(0, 0)


def test_drift_metrics():
    assert l1([1, -2], [0, 0]) == 3 and anchored_deviation([3], [1]) == 2
    trace = [[0], [1], [2], [3]]
    assert step_deviations(trace) == [1, 1, 1] and path_drift(trace) == 3
    assert predict_breach(2, 1, 3, 5) and not predict_breach(2, 1, 2, 5)
    for bad in ([1.5], [True], "12"):
        with pytest.raises(ValueError):
            l1(bad, [0])
    with pytest.raises(ValueError):
        l1([1, 2], [0])


RACER = """
import sys
from wavelock.rca_reference import GuardedAgent, HmacOperatorVerifier
agent = GuardedAgent(sys.argv[1], verifier=HmacOperatorVerifier(b"operator-key-for-tests-000000000"))
role = sys.argv[2]
if role == "sensor":
    for seq in range(1, 40):
        agent.observe(seq, [seq, 0])
else:
    for _ in range(40):
        agent.request("move", lambda a: None)
"""


def test_concurrent_processes_never_execute_after_latch(agent, tmp_path):
    """Real processes: a sensor drifting past the threshold races two actors.  The final memory
    loads, so every durable execute record (written before its effect) was admitted while the
    memory was unlocked; after the first latching observation no execute record follows."""
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    procs = [subprocess.Popen([sys.executable, "-c", RACER, str(tmp_path), role], env=env)
             for role in ("sensor", "actor", "actor")]
    for p in procs:
        assert p.wait(timeout=120) == 0
    final = GuardedAgent(tmp_path, verifier=VERIFIER)
    recs = [json.loads(x) for x in lines(tmp_path)]
    latch = next(i for i, r in enumerate(recs)
                 if r["kind"] == "observe" and l1(r["values"], [0, 0]) >= 5)
    assert final.view.locked
    assert all(r["kind"] != "execute" for r in recs[latch:])
