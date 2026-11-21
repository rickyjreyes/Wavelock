"""Python/Lean conformance for the proposed RCA reference kernel (issues #42 and #43).

``lean/rca_conformance_traces.json`` runs the PROPOSED reference kernel ``wavelock.rca_reference``
with real memory files, several ``GuardedAgent`` instances as processes, and write failures and
crashes injected into ``_write_line`` (and a crash between a durable ``execute`` intent and its
effect).  Traces of a non-reference discipline replace the writer's admissibility check
(``kernel.admissible``) by the ablated guard of ``RCA.guard`` and drive the critical section
without the lock or without the reload, as the discipline says; the loader is unchanged.

``scripts/gen_rca_conformance.py`` turns the same traces into ``RCAConformance.lean``, where each
expected outcome is checked by ``decide`` against ``RCATransitions``.  Agreement on finite traces
is evidence that the model describes the code, not a refinement proof.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import gen_rca_conformance as gen  # noqa: E402

from wavelock.rca_reference import kernel as K  # noqa: E402
from wavelock.rca_reference import memory as M  # noqa: E402

KERNEL = {t["name"]: t for t in json.loads(
    (ROOT / "lean" / "rca_conformance_traces.json").read_text(encoding="utf-8"))["kernel_traces"]}
OPERATOR_KEY = b"rca-conformance-operator-key-0001"


class SimulatedCrash(BaseException):
    """Process death at an injected point (escapes every ``except Exception``)."""


def test_generated_lean_is_current():
    assert (ROOT / "lean" / "WaveLockProofs" / "RCAConformance.lean").read_text(
        encoding="utf-8") == gen.render()


def action(k: int) -> str:
    return f"a{k}"


def to_policy(obj: dict) -> M.Policy:
    return M.Policy.make(obj["version"], obj["threshold"], [action(k) for k in obj["prohibited"]])


def ablated_admissible(disc: dict, issued: set):
    """``RCA.guard D``: the writer's check with the discipline's elements removed."""
    def adm(view, rec, *, verifier):
        kind = rec["kind"]
        if kind == "observe":
            fresh = not disc.get("seqCheck", True) or rec["seq"] > view.last_seq
            return fresh and len(rec["values"]) == len(view.baseline)
        if kind == "policy":
            new = M.Policy.from_json(rec["policy"])
            if disc.get("monotonePolicy", True):
                return M.successor(new, view.policy)
            return new.version == view.policy.version + 1
        if kind == "execute":
            gate = not disc.get("driftGate", True) or not view.locked
            return gate and rec["action"] not in view.policy.prohibited
        if kind == "reset":
            if not (view.locked and view.last_dev < view.policy.threshold):
                return False
            if disc.get("bindCred", True):
                return bool(verifier(view.agent, view.length, view.head, rec["credential"]))
            return rec["credential"] in issued
        return False
    return adm


def complete_lines(data: bytes) -> list:
    parts = data.split(b"\n")
    return parts[:-1]


def symbolic(rec: dict, cred_names: dict) -> list:
    def pol(p):
        return {"version": p["version"], "threshold": p["threshold"],
                "prohibited": [int(a[1:]) for a in p["prohibited"]]}
    kind = rec["kind"]
    if kind == "genesis":
        return ["genesis", int(rec["agent"].split("-")[1]), rec["baseline"], pol(rec["policy"])]
    if kind == "observe":
        return ["observe", rec["seq"], rec["values"]]
    if kind == "policy":
        return ["policy", pol(rec["policy"])]
    if kind == "execute":
        return ["execute", int(rec["action"][1:])]
    return ["reset", cred_names[rec["credential"]]]


class Proc:
    def __init__(self):
        self.agent = None
        self.alive = True
        self.view = None
        self.lock = None


def run_kernel(trace: dict, base: Path, monkeypatch, discipline=None) -> dict:
    disc = trace["discipline"] if discipline is None else discipline
    disc = {} if disc == "reference" else disc
    lock, reload, persist = (disc.get(k, True) for k in ("lock", "reload", "persist"))
    verifier = K.HmacOperatorVerifier(OPERATOR_KEY)
    log_path = base / K.LOG_NAME
    procs: dict = {}
    creds: dict = {}
    issued: set = set()
    state = {"anchor": None, "agent": None}
    effects, acked, results = [], [], []
    if disc:
        monkeypatch.setattr(K, "admissible", ablated_admissible(disc, issued))

    def data() -> bytes:
        return log_path.read_bytes() if log_path.exists() else b""

    def head():
        lines = complete_lines(data())
        return len(lines), (M.digest(lines[-1]) if lines else M.ZERO)

    def proc(p):
        return procs.setdefault(p, Proc())

    def ensure(pr):
        if pr.agent is None:
            pr.agent = K.GuardedAgent(base, verifier=verifier, anchor=state["anchor"])
            if not persist:
                pr.agent._write_line = lambda data: None
        return pr.agent

    def effect(name):
        effects.append(int(name[1:]))

    def fields_of(req):
        kind = req[0]
        if kind == "observe":
            return {"kind": "observe", "seq": req[1], "values": req[2]}
        if kind == "policy":
            return {"kind": "policy", "policy": to_policy(req[1]).to_json()}
        if kind == "execute":
            return {"kind": "execute", "action": action(req[1])}
        return {"kind": "reset", "credential": creds[req[1]]}

    def begin(p):
        pr = proc(p)
        if not pr.alive or pr.view is not None:
            return
        if lock and state.get("holder") is not None:
            return  # the lock is busy: a blocked acquisition changes nothing (RCA.step)
        try:
            agent = ensure(pr)
            if lock:
                pr.lock = agent._lock()
                pr.lock.__enter__()
                state["holder"] = p
            pr.view = agent._load() if reload else agent.view
        except M.MemoryCorrupt:
            release(pr)

    def release(pr):
        if pr.lock is not None:
            pr.lock.__exit__(None, None, None)
            pr.lock = None
            state["holder"] = None
        pr.view = None

    def submit(p, req, outcome):
        pr = proc(p)
        if not pr.alive or pr.view is None:
            results.append(None)
            return
        agent, fields = pr.agent, fields_of(req)
        real = agent._write_line
        run_effect = effect
        if outcome == "ok_crash_before_effect":
            def run_effect(name):  # the process dies after the intent's fsync, before the effect
                raise SimulatedCrash()
        elif outcome != "ok":
            what, effect_kind = outcome.split("_")

            def faulty(data, _real=real, _what=what, _effect=effect_kind):
                if persist and _effect == "line":
                    _real(data)
                elif persist and _effect == "torn":
                    _real(data[: len(data) // 2])
                raise OSError("injected write failure") if _what == "fail" else SimulatedCrash()
            agent._write_line = faulty
        try:
            ok = agent._submit(pr.view, fields,
                               run_effect if fields["kind"] == "execute" else None)
        except OSError:
            results.append(True)
        except SimulatedCrash:
            results.append(True)
            release(pr)
            pr.alive, pr.agent = False, None
            return
        else:
            results.append(ok)
            if ok:
                acked.append(fields)
        finally:
            if pr.agent is not None:
                agent._write_line = real
        release(pr)

    for ev in trace["events"]:
        kind = ev[0]
        if kind == "provision":
            state["agent"] = f"agent-{ev[1]}"
            try:
                K.GuardedAgent.provision(base, agent=state["agent"], baseline=ev[2],
                                         policy=to_policy(ev[3]), verifier=verifier)
            except FileExistsError:
                pass
        elif kind in ("observe", "policy", "execute", "reset"):
            begin(ev[1])
            submit(ev[1], [kind, *ev[2:-1]], ev[-1])
        elif kind == "begin":
            begin(ev[1])
        elif kind == "submit":
            submit(ev[1], ev[2], ev[3])
        elif kind == "issue":
            if ev[1] not in creds:
                length, h = head()
                creds[ev[1]] = K.operator_credential(OPERATOR_KEY, state["agent"], length, h)
                issued.add(creds[ev[1]])
        elif kind == "anchor":
            length, h = head()
            state["anchor"] = (length - 1, h)
            for pr in procs.values():
                if pr.agent is not None:
                    pr.agent._anchor = state["anchor"]
        elif kind == "crash":
            pr = proc(ev[1])
            release(pr)
            pr.alive, pr.agent = False, None
        elif kind == "restart":
            pr = proc(ev[1])
            if not pr.alive:
                pr.alive = True
                try:
                    ensure(pr)
                except M.MemoryCorrupt:
                    pr.agent = None
        elif kind == "rollback":
            lines = complete_lines(data())[: ev[1]]
            if lines:
                log_path.write_bytes(b"".join(line + b"\n" for line in lines))
            elif log_path.exists():
                log_path.unlink()
        elif kind == "direct":
            effect(action(ev[1]))
        else:
            raise ValueError(ev)

    raw = data()
    names = {v: k for k, v in creds.items()}
    try:
        M.load(raw, verifier=verifier, anchor=state["anchor"])
        loads = True
    except M.MemoryCorrupt:
        loads = False
    log = [symbolic(json.loads(line), names) for line in complete_lines(raw)]
    ack = [symbolic({**f, "kind": f["kind"]}, names) for f in acked]
    return {"log": log, "torn": bool(raw) and not raw.endswith(b"\n"), "loads": loads,
            "effects": effects, "acked": ack, "results": results}


def expected_kernel(trace: dict) -> dict:
    exp = trace["expected"]
    return {"log": exp["log"], "torn": exp["torn"], "loads": exp["loads"],
            "effects": [k for k, _ in exp["effects"]], "acked": exp["acked"],
            "results": exp["results"]}


@pytest.mark.parametrize("name", sorted(KERNEL))
def test_kernel_trace_matches_lean(name, tmp_path, monkeypatch):
    trace = KERNEL[name]
    assert run_kernel(trace, tmp_path, monkeypatch) == expected_kernel(trace)


ABLATIONS = sorted(n for n, t in KERNEL.items() if t["discipline"] != "reference")


@pytest.mark.parametrize("name", ABLATIONS)
def test_reference_kernel_rejects_each_ablation_attack(name, tmp_path, monkeypatch):
    """Negative control: the events of every ablation trace, run against the unmodified
    reference kernel, give a different outcome, and every effect that still happens was admitted
    on a memory that permitted it: the final memory loads (every record, in particular every
    write-ahead ``execute`` record, was admissible at its position; ``load_refines_spec``) and
    each effect has its ``execute`` record."""
    trace = KERNEL[name]
    got = run_kernel(trace, tmp_path, monkeypatch, discipline="reference")
    assert got != expected_kernel(trace)
    assert got["loads"] and not got["torn"]
    assert len(got["effects"]) == sum(1 for r in got["log"] if r[0] == "execute")


def test_kernel_mutated_expectation_is_detected(tmp_path, monkeypatch):
    trace = copy.deepcopy(KERNEL["kernel_drift_latch"])
    trace["expected"]["results"][-1] = True
    assert run_kernel(trace, tmp_path, monkeypatch) != expected_kernel(trace)


GUARD_SENSITIVE = ("kernel_init_and_guard", "kernel_drift_latch", "kernel_policy_updates",
                   "kernel_duplicate_and_stale_observation", "kernel_operator_reset")


@pytest.mark.parametrize("name", GUARD_SENSITIVE)
def test_removing_the_guard_fails_reference_traces(name, tmp_path, monkeypatch):
    """Negative control: with the writer's admissibility check replaced by "admit everything",
    each guard-sensitive reference trace no longer matches its Lean outcome."""
    monkeypatch.setattr(K, "admissible", lambda view, rec, *, verifier: True)
    trace = KERNEL[name]
    got = run_kernel(trace, tmp_path, monkeypatch, discipline="reference")
    assert got != expected_kernel(trace)


def test_removing_persistence_and_reload_fails_restart_trace(tmp_path, monkeypatch):
    """Negative control: without durable writes (and with the in-memory view reused), the latch of
    ``kernel_restart_keeps_latch`` does not survive the restart and the actions execute."""
    trace = KERNEL["kernel_restart_keeps_latch"]
    got = run_kernel(trace, tmp_path, monkeypatch, discipline={"persist": False, "reload": False})
    assert got != expected_kernel(trace) and got["effects"] == [3, 3]
