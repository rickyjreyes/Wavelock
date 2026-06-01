"""
Mythos integration-layer break pins (M1 / M2 / M3) — now HARD-PASSING.

Claude Mythos found three integration-layer bounty blockers after the A/B/D
remediation. The pure cryptographic core (``verify_ots``) held, but the block
acceptance / replay layer had cheap outs:

  * M1 — the OTS block signature did not bind the block body.
  * M2 — durable replay control failed open after deleting the side ledger and
    restarting (reconstruction depended on the current config / side cache).
  * M3 — no inter-process lock on ``PersistentOTSReplayLedger``; two instances on
    the same file could both accept the same identity.

These pins were authored xfail (secure behavior not yet implemented). After the
fix they are NORMAL passing tests asserting the secure behavior, plus they drive
the matching attack PoCs in ``attacks/`` and assert each break is now BLOCKED.
"""

import copy
import threading
import types

import pytest

from wavelock.chain.Block import Block
from wavelock.crypto.ots_ledger import (
    PersistentOTSReplayLedger,
    OTSLedgerError,
    INTERPROCESS_LOCKING,
)
from wavelock.crypto.wavelock_ots import (
    generate_ots_keypair,
    sign_ots,
    verify_ots,
)
from wavelock.network import server

from attacks import (
    ots_block_body_unbound,
    ots_ledger_reconstruction_failopen,
    ots_ledger_concurrent_double_accept,
)


def _fresh_ledger(tmp_path):
    return PersistentOTSReplayLedger(path=str(tmp_path / "ots_replay.jsonl"))


# ===========================================================================
# M1 — OTS signature binds the canonical block body
# ===========================================================================

def test_m1_benign_signature_cannot_authorize_malicious_body(tmp_path):
    """A valid signature over "hello world" must not authorize a different body."""
    led = _fresh_ledger(tmp_path)
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]

    benign = "hello world"
    sig = sign_ots(sec, benign)
    assert verify_ots(pub, benign, sig) is True  # crypto-valid over benign text

    meta = server.build_ots_block_meta(pub, benign, sig)
    b = Block(index=1, messages=["transfer 1000000 to attacker"],
              previous_hash="0" * 64, difficulty=1, block_type="OTS", meta=meta)
    assert server._verify_ots_block(b, None, ledger=led) is False


def test_m1_body_tampering_rejected(tmp_path):
    """Mutating Block.messages after signing must reject."""
    led = _fresh_ledger(tmp_path)
    kp = generate_ots_keypair(entropy_bits=256)
    good = server.build_signed_ots_block(kp["secret_key"], kp["public_key"],
                                         ["pay alice 5"])
    tampered = Block(index=good.index, messages=["pay attacker 5000000"],
                     previous_hash=good.previous_hash, difficulty=1,
                     block_type="OTS", meta=copy.deepcopy(good.meta))
    assert server._verify_ots_block(tampered, None, ledger=led) is False
    # The untouched, body-bound block still verifies.
    assert server._verify_ots_block(good, None, ledger=led) is True


def test_m1_transcript_changes_when_messages_change():
    """The canonical transcript/digest must change if Block.messages changes."""
    kp = generate_ots_keypair(entropy_bits=256)
    b1 = server.build_signed_ots_block(kp["secret_key"], kp["public_key"],
                                       ["pay alice 5"])
    # Same auth meta, different body.
    b2 = Block(index=b1.index, messages=["pay alice 6"],
               previous_hash=b1.previous_hash, difficulty=1,
               block_type="OTS", meta=copy.deepcopy(b1.meta))
    assert server.canonical_ots_block_message(b1) != server.canonical_ots_block_message(b2)
    assert server.canonical_ots_block_digest(b1) != server.canonical_ots_block_digest(b2)


def test_m1_transcript_changes_with_block_type_and_prev():
    kp = generate_ots_keypair(entropy_bits=256)
    b = server.build_signed_ots_block(kp["secret_key"], kp["public_key"], ["x"])
    retyped = Block(index=b.index, messages=list(b.messages),
                    previous_hash=b.previous_hash, difficulty=1,
                    block_type="GENERIC", meta=copy.deepcopy(b.meta))
    reparented = Block(index=b.index, messages=list(b.messages),
                       previous_hash="1" * 64, difficulty=1,
                       block_type="OTS", meta=copy.deepcopy(b.meta))
    assert server.canonical_ots_block_digest(b) != server.canonical_ots_block_digest(retyped)
    assert server.canonical_ots_block_digest(b) != server.canonical_ots_block_digest(reparented)


def test_m1_attack_poc_now_blocked():
    assert ots_block_body_unbound.benign_signature_authorizes_malicious_body() is False
    assert ots_block_body_unbound.body_tampering_after_signing() is False


# ===========================================================================
# M2 — replay reconstruction is a function of accepted chain state, not cfg
# ===========================================================================

def test_m2_replay_rejected_after_side_ledger_delete_and_restart(tmp_path, monkeypatch):
    """Deleting ots_replay.jsonl + restarting must NOT reopen replay if the
    accepted chain still contains the prior OTS block (via load_from_disk)."""
    path = str(tmp_path / "ots_replay.jsonl")
    led1 = PersistentOTSReplayLedger(path=path)
    kp = generate_ots_keypair(entropy_bits=256)
    b1 = server.build_signed_ots_block(kp["secret_key"], kp["public_key"],
                                       ["pay alice 5"])
    assert server._verify_ots_block(b1, None, ledger=led1) is True

    # Attacker deletes the side replay cache.
    import os
    os.remove(path)

    # Restart node: fresh empty ledger object + load accepted chain from disk.
    led2 = PersistentOTSReplayLedger(path=path)
    monkeypatch.setattr(server, "CONSENSUS_OTS_LEDGER", led2)
    monkeypatch.setattr(server, "load_all_blocks", lambda: [b1])
    server.CHAIN.load_from_disk()

    # The consumed identity is reconstructed from chain state.
    assert led2.is_consumed(b1.meta["ots_auth"]["signature"]) is True
    # A replay of the same OTS identity is rejected.
    b2 = Block(index=2, messages=list(b1.messages), previous_hash=b1.previous_hash,
               difficulty=1, block_type="OTS", meta=copy.deepcopy(b1.meta))
    assert server._verify_ots_block(b2, None, ledger=led2) is False


def test_m2_reconstruction_indexes_block_by_structure_independent_of_cfg(tmp_path):
    """An accepted block carrying OTS auth but no block_type/auth_scheme markers
    (as if accepted only because cfg.require_ots was set) must still be indexed."""
    path = str(tmp_path / "ots_replay.jsonl")
    kp = generate_ots_keypair(entropy_bits=256)
    signed = server.build_signed_ots_block(kp["secret_key"], kp["public_key"],
                                           ["pay alice 5"])
    auth = signed.meta["ots_auth"]
    stripped = Block(index=1, messages=list(signed.messages),
                     previous_hash=signed.previous_hash, difficulty=1,
                     block_type="GENERIC", meta={"ots_auth": copy.deepcopy(auth)})
    led = PersistentOTSReplayLedger(path=path)
    server._reconstruct_consumed_ots([stripped], led)
    assert led.is_consumed(auth["signature"]) is True


def test_m2_malformed_ots_auth_in_accepted_chain_fails_closed(tmp_path):
    """A block that claims OTS but has malformed auth must fail closed."""
    path = str(tmp_path / "ots_replay.jsonl")
    led = PersistentOTSReplayLedger(path=path)
    bad = Block(index=1, messages=["x"], previous_hash="0" * 64, difficulty=1,
                block_type="OTS",
                meta={"auth_scheme": server.OTS_SCHEME,
                      "ots_auth": {"public_key": {}, "message": "m",
                                   "signature": "not-a-dict"}})
    with pytest.raises(OTSLedgerError):
        server._reconstruct_consumed_ots([bad], led)


def test_m2_load_from_disk_fails_closed_on_malformed_ots_block(tmp_path, monkeypatch):
    bad = Block(index=1, messages=["x"], previous_hash="0" * 64, difficulty=1,
                block_type="OTS", meta={"auth_scheme": server.OTS_SCHEME})
    monkeypatch.setattr(server, "load_all_blocks", lambda: [bad])
    monkeypatch.setattr(server, "CONSENSUS_OTS_LEDGER",
                        PersistentOTSReplayLedger(path=str(tmp_path / "l.jsonl")))
    with pytest.raises(OTSLedgerError):
        server.CHAIN.load_from_disk()


def test_m2_attack_poc_now_blocked():
    assert ots_ledger_reconstruction_failopen.replay_after_side_ledger_delete() is False
    assert ots_ledger_reconstruction_failopen.reconstruction_indexes_cfg_only_block() is True


# ===========================================================================
# M3 — inter-process lock prevents double accept across instances
# ===========================================================================

@pytest.mark.skipif(not INTERPROCESS_LOCKING,
                    reason="inter-process flock is POSIX-only")
def test_m3_two_instances_cannot_both_accept(tmp_path):
    path = str(tmp_path / "ots_replay.jsonl")
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "one-time")

    led1 = PersistentOTSReplayLedger(path=path)
    led2 = PersistentOTSReplayLedger(path=path)
    r1 = led1.accept(pub, "one-time", copy.deepcopy(sig))
    r2 = led2.accept(pub, "one-time", copy.deepcopy(sig))
    assert (r1, r2) == (True, False)


@pytest.mark.skipif(not INTERPROCESS_LOCKING,
                    reason="inter-process flock is POSIX-only")
def test_m3_concurrent_duplicate_acceptance_accepts_at_most_one(tmp_path):
    path = str(tmp_path / "ots_replay.jsonl")
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "race")

    n = 8
    ledgers = [PersistentOTSReplayLedger(path=path) for _ in range(n)]
    results = [False] * n
    barrier = threading.Barrier(n)

    def worker(i):
        barrier.wait()
        results[i] = ledgers[i].accept(pub, "race", copy.deepcopy(sig))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(1 for r in results if r) == 1


@pytest.mark.skipif(not INTERPROCESS_LOCKING,
                    reason="inter-process flock is POSIX-only")
def test_m3_attack_poc_now_blocked():
    assert ots_ledger_concurrent_double_accept.concurrent_accept_count() == 1
    assert ots_ledger_concurrent_double_accept.two_instances_double_accept() is False


# ===========================================================================
# Authoritative single ledger (M3 unification): no independent bypass ledger
# ===========================================================================

def test_single_authoritative_ledger_no_bypass():
    """OTS_LEDGER must be the same object as the authoritative consensus ledger,
    so a signature consumed via verify_ots_payload is also consumed for consensus."""
    assert server.OTS_LEDGER is server.CONSENSUS_OTS_LEDGER
