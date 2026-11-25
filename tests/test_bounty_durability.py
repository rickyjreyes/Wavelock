"""Real process races and interrupted-write tests for local acceptance state."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from wavelock.chain.ots_blocks import build_signed_ots_block
from wavelock.crypto.wavelock_ots import generate_ots_keypair, sign_ots
from wavelock.storage import durability

ROOT = Path(__file__).resolve().parents[1]


def environment(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(ROOT), "WAVELOCK_DATA_DIR": str(tmp_path / "node"),
           "WAVELOCK_OTS_STATE_DIR": str(tmp_path / "signer-state"),
           "WAVELOCK_REQUIRE_OTS": "1", "POW_TARGET": "f"*64,
           "PYTHONIOENCODING": "utf-8"}
    env.pop("WAVELOCK_CONFIG", None)
    env.pop("WAVELOCK_OTS_LEDGER", None)
    return env


def run(code, tmp_path, *args):
    return subprocess.run([sys.executable, "-c", code, *map(str, args)],
                          env=environment(tmp_path), cwd=tmp_path,
                          capture_output=True, text=True, timeout=30)


def test_atomic_file_fsync_failure_never_publishes_partial_artifact(tmp_path, monkeypatch):
    target = tmp_path / "artifact.json"
    target.write_bytes(b"previous-complete-file")
    def fail(fd):
        raise OSError("injected file fsync failure")
    monkeypatch.setattr(durability.os, "fsync", fail)
    with pytest.raises(OSError):
        durability.atomic_write(target, b"replacement")
    assert target.read_bytes() == b"previous-complete-file"
    missing = tmp_path / "missing.json"
    with pytest.raises(OSError):
        durability.atomic_write(missing, b"new", exclusive=True)
    assert not missing.exists()


def test_directory_fsync_failure_does_not_report_success(tmp_path, monkeypatch):
    def fail(path):
        raise OSError("injected directory fsync failure")
    monkeypatch.setattr(durability, "fsync_directory", fail)
    target = tmp_path / "used"
    with pytest.raises(OSError):
        durability.atomic_write(target, b"used", exclusive=True)
    # Publication may have happened, but only complete bytes can be visible.
    assert not target.exists() or target.read_bytes() == b"used"


def test_process_exit_after_signer_claim_burns_key(tmp_path):
    kp = generate_ots_keypair()
    secret = tmp_path / "secret.json"
    secret.write_text(json.dumps(kp["secret_key"]))
    crash = run("""
import json,os,sys
from wavelock.crypto import wavelock_ots as ots
ots._secret_slice = lambda *a, **k: os._exit(73)
ots.sign_ots(json.load(open(sys.argv[1])), 'first')
""", tmp_path, secret)
    assert crash.returncode == 73
    retry = run("""
import json,sys
from wavelock.crypto.wavelock_ots import sign_ots,OTSKeyReuseError
try:
    sign_ots(json.load(open(sys.argv[1])), 'second')
except OTSKeyReuseError:
    sys.exit(0)
sys.exit(1)
""", tmp_path, secret)
    assert retry.returncode == 0, retry.stderr


def test_process_exit_between_replay_and_append_stays_consumed(tmp_path):
    kp = generate_ots_keypair()
    block = build_signed_ots_block(kp["secret_key"], kp["public_key"], ["one"], difficulty=0)
    path = tmp_path / "block.json"
    path.write_text(json.dumps(block.to_dict()))
    crash = run("""
import json,os,sys
from wavelock.network import server
from wavelock.chain.config import Config
server.save_block_to_disk = lambda b: os._exit(74)
server.try_accept_block_dict(json.load(open(sys.argv[1])), Config())
""", tmp_path, path)
    assert crash.returncode == 74, crash.stderr
    retry = run("""
import json,sys
from wavelock.network import server
from wavelock.chain.config import Config
assert not server.try_accept_block_dict(json.load(open(sys.argv[1])), Config())
assert server.CHAIN.blocks == []
""", tmp_path, path)
    assert retry.returncode == 0, retry.stdout + retry.stderr


@pytest.mark.parametrize("kind", ["chain", "replay", "replay-sqlite", "signer"])
def test_same_host_multiprocess_exclusion(tmp_path, kind):
    paths = []
    first = generate_ots_keypair()
    for index in range(2):
        if kind == "chain":
            kp = first if index == 0 else generate_ots_keypair()
            value = build_signed_ots_block(kp["secret_key"], kp["public_key"], [str(index)], difficulty=0).to_dict()
        elif kind == "signer":
            value = first["secret_key"]
        else:
            if index == 0:
                signature = sign_ots(first["secret_key"], "once")
            value = {"public_key": first["public_key"], "signature": signature}
        path = tmp_path / f"input-{index}.json"
        path.write_text(json.dumps(value))
        paths.append(path)
    worker = """
import json,sys
from wavelock.network import server
from wavelock.chain.config import Config
from wavelock.crypto import ots_ledger
from wavelock.crypto.wavelock_ots import sign_ots,OTSKeyReuseError
data=json.load(open(sys.argv[1])); kind=sys.argv[2]
if kind == 'replay-sqlite': ots_ledger._fcntl=None
if kind.startswith('replay'): ledger=ots_ledger.PersistentOTSReplayLedger('shared.jsonl')
if kind == 'chain': server.CHAIN.load_from_disk(require_ots=True)
print('READY',flush=True); sys.stdin.readline()
if kind == 'chain':
    result=server.try_accept_block_dict(data,Config())
elif kind == 'signer':
    try: sign_ots(data,'one'); result=True
    except OTSKeyReuseError: result=False
else:
    result=ledger.accept(data['public_key'],'once',data['signature'])
print('RESULT='+str(int(result)),flush=True)
"""
    processes = [subprocess.Popen([sys.executable, "-c", worker, str(p), kind],
                 cwd=tmp_path, env=environment(tmp_path), stdin=subprocess.PIPE,
                 stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for p in paths]
    try:
        for process in processes:
            assert process.stdout.readline().strip() == "READY"
        for process in processes:
            process.stdin.write("GO\n")
            process.stdin.flush()
        results = []
        for process in processes:
            output, error = process.communicate(timeout=30)
            assert process.returncode == 0, output + error
            results.append(int(output.strip().split("RESULT=")[-1]))
        assert sorted(results) == [0, 1]
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    if kind == "chain":
        records = (tmp_path / "node/ledger/blk00000.jsonl").read_text().splitlines()
        assert len(records) == 1


@pytest.mark.parametrize("record", [{}, [], {"v": 2}, {"v": 1, "one_time_key_id": "x",
    "leaf_id": "f"*64, "transcript": "bad"}])
def test_malformed_replay_records_fail_closed(tmp_path, record):
    from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger, OTSLedgerError
    path = tmp_path / "malformed.jsonl"
    path.write_text(json.dumps(record) + "\n")
    with pytest.raises(OTSLedgerError):
        PersistentOTSReplayLedger(str(path))
