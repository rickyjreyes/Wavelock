"""Exercise the supported workflow through fresh CLI processes and real files."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import pytest

from wavelock.chain.config import Config, load_config
from wavelock.chain.ots_blocks import (
    build_signed_ots_block, verify_ots_block, verify_ots_chain,
)
from wavelock.crypto.wavelock_ots import generate_ots_keypair


ROOT = Path(__file__).resolve().parents[1]


def test_real_config_loader_enforces_ots(tmp_path, monkeypatch):
    monkeypatch.delenv("WAVELOCK_REQUIRE_OTS", raising=False)
    assert load_config().require_ots is True
    monkeypatch.setenv("WAVELOCK_REQUIRE_OTS", "0")
    assert load_config().require_ots is False
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"require_ots": True, "port": 9010}))
    assert load_config(path).require_ots is True
    assert load_config(path).port == 9010
    path.write_text(json.dumps({"require_ost": True}))
    with pytest.raises(ValueError, match="unknown configuration"):
        load_config(path)
    path.write_text(json.dumps({"require_ots": "maybe"}))
    with pytest.raises(ValueError, match="expected true/false"):
        load_config(path)


def test_cli_sign_mine_restart_and_public_only_verify(tmp_path):
    data_dir = tmp_path / "node"
    env = {**os.environ, "WAVELOCK_DATA_DIR": str(data_dir),
           "WAVELOCK_OTS_STATE_DIR": str(tmp_path / "signer-state"),
           "WAVELOCK_REQUIRE_OTS": "1", "POW_TARGET": "0" + "f" * 63,
           "PYTHONPATH": str(ROOT)}
    env.pop("WAVELOCK_CONFIG", None)
    env.pop("WAVELOCK_OTS_LEDGER", None)

    def cli(*arguments, ok=True):
        result = subprocess.run(
            [sys.executable, "-m", "wavelock.chain.cli", *arguments],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
        )
        assert (result.returncode == 0) is ok, result.stdout + result.stderr
        if ok:
            assert "DEPRECATED" not in result.stdout + result.stderr
        return result

    cli("add", "ricky", "--encrypt", "--passphrase", "test-only-passphrase")
    secret_path = tmp_path / "keys/ricky/wl_ots_secret.json"
    original_secret = secret_path.read_bytes()
    cli("add", "ricky", ok=False)
    assert secret_path.read_bytes() == original_secret
    cli("sign", "ricky", "--message", "research artifact abc", "--difficulty", "1",
        "--passphrase", "test-only-passphrase")
    used_secret = json.loads(secret_path.read_text())
    assert used_secret["used"] is True
    assert "seed_hex" not in used_secret
    assert used_secret["seed_enc"] == json.loads(original_secret)["seed_enc"]
    payload_path = tmp_path / "signed_message.json"
    payload = json.loads(payload_path.read_text())
    assert payload["format"] == "WaveLock-SignedBlock-v1"
    public = payload["block"]["meta"]["ots_auth"]["public_key"]
    assert not {"seed_hex", "psi_0", "psi_star"}.intersection(public)
    cli("sign", "ricky", "--message", "second", "--output", "second.json",
        "--passphrase", "test-only-passphrase", ok=False)
    cli("verify", "--signed-path", "signed_message.json")

    tampered = copy.deepcopy(payload)
    tampered["block"]["messages"] = ["changed body"]
    (tmp_path / "tampered.json").write_text(json.dumps(tampered))
    cli("verify", "--signed-path", "tampered.json", ok=False)
    cli("mine", "--signed-path", "tampered.json", ok=False)
    cli("mine")
    ledger_path = data_dir / "ledger/blk00000.jsonl"
    assert len(ledger_path.read_text().splitlines()) == 1
    shutil.rmtree(tmp_path / "keys")  # verification must not load any secret.
    cli("verify")
    cli("audit")
    cli("mine", ok=False)
    # A second independent signer extends the chain without the first secret.
    cli("keygen", "--out", "keys/second")
    cli("sign", "--secret", "keys/second/wl_ots_secret.json", "--message", "next",
        "--difficulty", "1", "--output", "next.json")
    cli("mine", "--signed_path", "next.json")
    shutil.rmtree(tmp_path / "keys")
    cli("verify")
    assert len(ledger_path.read_text().splitlines()) == 2

    # Rebuild the actual replay cache from the actual accepted chain on restart.
    (data_dir / "ledger/ots_replay.jsonl").unlink()
    check = subprocess.run([
        sys.executable, "-c",
        "from wavelock.network import server; "
        "server.CHAIN.load_from_disk(require_ots=True); "
        "b=server.CHAIN.blocks[0]; "
        "assert server.CONSENSUS_OTS_LEDGER.is_consumed(b.meta['ots_auth']['signature']); "
        "assert not server._verify_ots_block(b)",
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert check.returncode == 0, check.stdout + check.stderr

    # Global flags also reach delegated commands before runtime paths load.
    alternate = tmp_path / "historical"
    legacy = subprocess.run([
        sys.executable, "-m", "wavelock.chain.cli", "--data-dir", str(alternate),
        "legacy", "view",
    ], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert legacy.returncode == 0, legacy.stdout + legacy.stderr
    assert "Ledger is empty" in legacy.stdout
    assert "DEPRECATED" in legacy.stdout + legacy.stderr
    assert alternate.joinpath("ledger").is_dir()


@pytest.mark.parametrize("field,value", [("nonce", None), ("hash", None), ("difficulty", -1)])
def test_malformed_block_deserialization_never_mines(monkeypatch, field, value):
    from wavelock.chain.Block import Block
    block = Block(1, ["message"], "0" * 64, difficulty=0).to_dict()
    block[field] = value
    def unexpected_mining(self):
        pytest.fail("deserialization attempted to mine untrusted input")
    monkeypatch.setattr(Block, "mine_block", unexpected_mining)
    with pytest.raises(ValueError):
        Block.from_dict(block)


def test_signing_fails_closed_when_state_cannot_be_written(tmp_path, monkeypatch):
    from wavelock.crypto.wavelock_ots import WaveLockOTSError, sign_ots
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("cannot store signer state here")
    monkeypatch.setenv("WAVELOCK_OTS_STATE_DIR", str(blocked / "state"))
    kp = generate_ots_keypair()
    with pytest.raises(WaveLockOTSError, match="cannot durably claim"):
        sign_ots(kp["secret_key"], "never publish this signature")
    assert not kp["secret_key"].get("used")


def test_public_chain_rejects_reuse_and_header_tampering():
    kp = generate_ots_keypair()
    first = build_signed_ots_block(kp["secret_key"], kp["public_key"], ["first"], difficulty=0)
    second = build_signed_ots_block(kp["secret_key"], kp["public_key"], ["second"],
                                    index=2, previous_hash=first.hash, difficulty=0,
                                    allow_reuse=True)
    assert verify_ots_block(first) and verify_ots_block(second)
    assert not verify_ots_chain([first, second])
    assert verify_ots_chain([first])
    first.hash = "0" * 64
    assert not verify_ots_chain([first])


def test_default_node_rejects_legacy_authentication(monkeypatch):
    from wavelock.chain.Block import Block
    from wavelock.network import server
    monkeypatch.setattr(server, "CHAIN", server.ChainState())
    monkeypatch.setattr(server, "_verify_curvature", lambda *args: True)
    block = Block(1, ["legacy"], "0" * 64, difficulty=0)
    assert not server.try_accept_block(block, Config(pow_target="f" * 64))


def test_concurrent_siblings_cannot_both_append(monkeypatch):
    from wavelock.network import server
    from wavelock.crypto.wavelock_ots import OTSReplayLedger
    monkeypatch.setattr(server, "CHAIN", server.ChainState())
    monkeypatch.setattr(server, "save_block_to_disk", lambda b: None)
    monkeypatch.setattr(server, "broadcast_inv", lambda h: None)
    monkeypatch.setattr(server, "CONSENSUS_OTS_LEDGER", OTSReplayLedger())
    blocks = []
    for message in ("one", "two"):
        kp = generate_ots_keypair()
        blocks.append(build_signed_ots_block(kp["secret_key"], kp["public_key"],
                                             [message], difficulty=0))
    barrier = threading.Barrier(2)
    results = []
    def accept(block):
        barrier.wait()
        results.append(server.try_accept_block(block, Config(pow_target="f" * 64)))
    threads = [threading.Thread(target=accept, args=(b,)) for b in blocks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert sorted(results) == [False, True]
    assert len(server.CHAIN.blocks) == 1


def test_sqlite_fallback_lock_rejects_concurrent_replay(tmp_path, monkeypatch):
    from wavelock.crypto import ots_ledger
    from wavelock.crypto.wavelock_ots import sign_ots
    monkeypatch.setattr(ots_ledger, "_fcntl", None)
    kp = generate_ots_keypair()
    sig = sign_ots(kp["secret_key"], "once")
    ledgers = [ots_ledger.PersistentOTSReplayLedger(str(tmp_path / "replay.jsonl"))
               for _ in range(4)]
    barrier = threading.Barrier(4)
    results = []
    def accept(ledger):
        barrier.wait()
        results.append(ledger.accept(kp["public_key"], "once", sig))
    threads = [threading.Thread(target=accept, args=(ledger,)) for ledger in ledgers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert sum(results) == 1
