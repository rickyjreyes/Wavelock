"""
Consensus / block-acceptance integration tests for WaveLock-OTS (Finding D).

These pin the *stateful* half of the boundary: OTS verification is now wired
into the real block-acceptance path (``server.try_accept_block`` →
``server._verify_ots_block``) backed by the durable
:class:`PersistentOTSReplayLedger`. The pure cryptographic check stays in
``verify_ots`` (covered by test_ots_security / test_ots_redteam); here we prove
the acceptance layer rejects replays, copied-key second-signs, cold-copy
duplicates, malformed auth, and legacy SIGv2 on an OTS-required block.

WaveLock-OTS is experimental and NOT production-ready; Finding D is only fully
closed when every accepting node runs this rejection against a ledger derived
from agreed chain state. The host-local signing registry is defense-in-depth.
"""

import copy
import types

import pytest

from wavelock.chain.Block import Block
from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger, OTSLedgerError
from wavelock.crypto.wavelock_ots import (
    generate_ots_keypair,
    sign_ots,
    verify_ots,
)
from wavelock.network import server


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _fresh_ledger(tmp_path):
    return PersistentOTSReplayLedger(path=str(tmp_path / "ots_replay.jsonl"))


def _ots_block(pub, msg, sig, index=1, prev="0" * 64, difficulty=1):
    """Build a mined, OTS-authenticated block (difficulty kept low for speed)."""
    meta = server.build_ots_block_meta(pub, msg, sig)
    return Block(
        index=index,
        messages=[f"ots: {msg}"],
        previous_hash=prev,
        difficulty=difficulty,
        block_type="OTS",
        meta=meta,
    )


# ---------------------------------------------------------------------------
# 1. valid first OTS block accepted
# ---------------------------------------------------------------------------

def test_valid_first_ots_block_accepted(tmp_path):
    led = _fresh_ledger(tmp_path)
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "pay alice 5")
    b = _ots_block(pub, "pay alice 5", sig)
    assert server._verify_ots_block(b, None, ledger=led) is True


def test_valid_first_ots_block_accepted_end_to_end(tmp_path, monkeypatch):
    """Full try_accept_block path: OTS block is routed to OTS verify+ledger."""
    led = _fresh_ledger(tmp_path)
    monkeypatch.setattr(server, "CONSENSUS_OTS_LEDGER", led)
    monkeypatch.setattr(server, "_verify_pow_and_linkage", lambda b, cfg: True)
    appended = []
    monkeypatch.setattr(server.CHAIN, "append", lambda b: appended.append(b))
    monkeypatch.setattr(server, "broadcast_inv", lambda h: None)

    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "genesis-ots")
    b = _ots_block(pub, "genesis-ots", sig)
    assert server.try_accept_block(b, types.SimpleNamespace()) is True
    assert len(appended) == 1


# ---------------------------------------------------------------------------
# 2. replay of the exact same OTS signature rejected
# ---------------------------------------------------------------------------

def test_replay_same_ots_signature_rejected(tmp_path):
    led = _fresh_ledger(tmp_path)
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "pay alice 5")
    b1 = _ots_block(pub, "pay alice 5", sig)
    b2 = _ots_block(pub, "pay alice 5", copy.deepcopy(sig), index=2)

    assert server._verify_ots_block(b1, None, ledger=led) is True
    # Exact same signature again => same one_time_key_id/leaf => rejected.
    assert server._verify_ots_block(b2, None, ledger=led) is False


# ---------------------------------------------------------------------------
# 3. different message signed with the SAME copied secret key rejected
#    after the first acceptance
# ---------------------------------------------------------------------------

def test_different_message_same_copied_key_rejected_after_first(tmp_path):
    led = _fresh_ledger(tmp_path)
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]

    sig1 = sign_ots(sec, "pay alice 5")
    b1 = _ots_block(pub, "pay alice 5", sig1)
    assert server._verify_ots_block(b1, None, ledger=led) is True

    # A copied secret key (taken before use) signs a DIFFERENT message.
    # allow_reuse simulates the copy producing a second, individually-valid sig.
    sig2 = sign_ots(sec, "pay mallory 1000000", allow_reuse=True)
    assert verify_ots(pub, "pay mallory 1000000", sig2) is True  # crypto-valid
    b2 = _ots_block(pub, "pay mallory 1000000", sig2, index=2)
    # ...but the ledger rejects it: same one_time_key_id already consumed.
    assert server._verify_ots_block(b2, None, ledger=led) is False


# ---------------------------------------------------------------------------
# 4. second-host / cold-copy simulation rejected by the (durable) ledger
# ---------------------------------------------------------------------------

def test_second_host_cold_copy_rejected_by_durable_ledger(tmp_path):
    """A copy moved to another host bypasses the host-local registry but the
    durable consensus ledger still rejects the duplicate key id.

    Simulated by reopening a fresh ledger object on the SAME file (as a second
    node reading shared/persisted chain-derived state would) and replaying.
    """
    path = str(tmp_path / "ots_replay.jsonl")
    led_host1 = PersistentOTSReplayLedger(path=path)
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]

    sig1 = sign_ots(sec, "pay alice 5")
    b1 = _ots_block(pub, "pay alice 5", sig1)
    assert server._verify_ots_block(b1, None, ledger=led_host1) is True

    # "Cold copy" of the secret key signs again on another host (allow_reuse),
    # and a second node loads the persisted ledger and sees the consumption.
    led_host2 = PersistentOTSReplayLedger(path=path)
    sig2 = sign_ots(sec, "pay alice 5 again", allow_reuse=True)
    b2 = _ots_block(pub, "pay alice 5 again", sig2, index=2)
    assert server._verify_ots_block(b2, None, ledger=led_host2) is False


def test_durable_ledger_survives_reopen(tmp_path):
    """Consumed ids persist across ledger object lifetimes (durability)."""
    path = str(tmp_path / "ots_replay.jsonl")
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "durable")

    led = PersistentOTSReplayLedger(path=path)
    assert led.accept(pub, "durable", sig) is True

    reopened = PersistentOTSReplayLedger(path=path)
    assert reopened.is_consumed(sig) is True
    assert reopened.accept(pub, "durable", sig) is False


# ---------------------------------------------------------------------------
# 5. malformed or missing auth / replay fields rejected
# ---------------------------------------------------------------------------

def test_missing_ots_auth_rejected(tmp_path):
    led = _fresh_ledger(tmp_path)
    b = Block(index=1, messages=["ots: x"], previous_hash="0" * 64,
              difficulty=1, block_type="OTS", meta={"auth_scheme": server.OTS_SCHEME})
    assert server._verify_ots_block(b, None, ledger=led) is False


def test_malformed_auth_fields_rejected(tmp_path):
    led = _fresh_ledger(tmp_path)
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "m")

    # signature not a dict
    bad_meta = {"auth_scheme": server.OTS_SCHEME,
                "ots_auth": {"public_key": pub, "message": "m", "signature": "nope"}}
    b = Block(index=1, messages=["ots: m"], previous_hash="0" * 64,
              difficulty=1, block_type="OTS", meta=bad_meta)
    assert server._verify_ots_block(b, None, ledger=led) is False

    # message missing
    bad_meta2 = {"auth_scheme": server.OTS_SCHEME,
                 "ots_auth": {"public_key": pub, "signature": sig}}
    b2 = Block(index=1, messages=["ots: m"], previous_hash="0" * 64,
               difficulty=1, block_type="OTS", meta=bad_meta2)
    assert server._verify_ots_block(b2, None, ledger=led) is False


def test_tampered_revealed_slice_rejected_without_consuming(tmp_path):
    led = _fresh_ledger(tmp_path)
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "ok")
    tampered = copy.deepcopy(sig)
    tampered["revealed_slices"][0] = "00" * 32
    b_bad = _ots_block(pub, "ok", tampered)
    assert server._verify_ots_block(b_bad, None, ledger=led) is False
    # Not consumed => the genuine signature still goes through.
    b_good = _ots_block(pub, "ok", sig, index=2)
    assert server._verify_ots_block(b_good, None, ledger=led) is True


# ---------------------------------------------------------------------------
# 6. legacy SIGv2 rejected when OTS is required
# ---------------------------------------------------------------------------

def test_legacy_sigv2_rejected_on_ots_required_block(tmp_path):
    led = _fresh_ledger(tmp_path)
    kp = generate_ots_keypair(entropy_bits=256)
    pub = kp["public_key"]
    legacy_sig = {"scheme": "WLv2", "signature": "deadbeef" * 8}
    meta = {"auth_scheme": server.OTS_SCHEME,
            "ots_auth": {"public_key": pub, "message": "m", "signature": legacy_sig}}
    b = Block(index=1, messages=["ots: m"], previous_hash="0" * 64,
              difficulty=1, block_type="OTS", meta=meta)
    assert server._verify_ots_block(b, None, ledger=led) is False


def test_block_requires_ots_detection():
    # by block_type
    b1 = Block(index=1, messages=["x"], previous_hash="0" * 64, difficulty=1,
               block_type="OTS", meta={})
    assert server.block_requires_ots(b1) is True
    # by meta.auth_scheme
    b2 = Block(index=1, messages=["x"], previous_hash="0" * 64, difficulty=1,
               block_type="GENERIC", meta={"auth_scheme": server.OTS_SCHEME})
    assert server.block_requires_ots(b2) is True
    # by cfg.require_ots
    b3 = Block(index=1, messages=["x"], previous_hash="0" * 64, difficulty=1,
               block_type="GENERIC", meta={})
    assert server.block_requires_ots(b3, types.SimpleNamespace(require_ots=True)) is True
    # otherwise not
    assert server.block_requires_ots(b3, types.SimpleNamespace(require_ots=False)) is False


def test_ots_required_block_never_falls_back_to_curvature(tmp_path, monkeypatch):
    """An OTS-required block must NOT be accepted via the legacy curvature path."""
    led = _fresh_ledger(tmp_path)
    monkeypatch.setattr(server, "CONSENSUS_OTS_LEDGER", led)
    monkeypatch.setattr(server, "_verify_pow_and_linkage", lambda b, cfg: True)
    # If the legacy path were ever consulted for an OTS block, force it to pass;
    # acceptance must STILL fail because the OTS auth is bogus.
    monkeypatch.setattr(server, "_verify_curvature", lambda b, cfg: True)
    monkeypatch.setattr(server.CHAIN, "append", lambda b: None)
    monkeypatch.setattr(server, "broadcast_inv", lambda h: None)

    kp = generate_ots_keypair(entropy_bits=256)
    pub = kp["public_key"]
    legacy_sig = {"scheme": "WLv2", "signature": "deadbeef"}
    meta = {"auth_scheme": server.OTS_SCHEME,
            "ots_auth": {"public_key": pub, "message": "m", "signature": legacy_sig}}
    b = Block(index=1, messages=["ots: m"], previous_hash="0" * 64,
              difficulty=1, block_type="OTS", meta=meta)
    assert server.try_accept_block(b, types.SimpleNamespace()) is False


# ---------------------------------------------------------------------------
# ledger corruption fails closed
# ---------------------------------------------------------------------------

def test_corrupt_ledger_file_fails_closed(tmp_path):
    path = tmp_path / "ots_replay.jsonl"
    path.write_text("{ this is not valid json\n")
    with pytest.raises(OTSLedgerError):
        PersistentOTSReplayLedger(path=str(path))
