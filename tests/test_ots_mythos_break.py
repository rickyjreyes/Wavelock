"""
Claude-Mythos adversarial red-team regression tests for WaveLock-OTS.

Full write-up: ``attacks/WAVELOCK_MYTHOS_BREAK_REPORT.md``.

These pin three breaks that survive the A/B/C/D remediation. For each finding we
provide TWO tests, matching the established pattern in ``test_ots_redteam.py``:

* a ``test_break_*`` that asserts the exploit SUCCEEDS today (it passes now and
  documents the live break — regression evidence);
* a ``test_secure_*`` marked ``xfail(strict=True)`` that asserts the DESIRED
  secure behavior. It is an expected-failure until the bug is fixed; once fixed
  it XPASSes and the strict marker flips it to a hard failure, forcing the marker
  (and this comment) to be removed.

Findings:
  M1 (HIGH)   — OTS block signature does not bind ``Block.messages``; a block's
                transaction payload is unauthenticated relative to the signed
                message. Any published, unspent signature mints a first-accept
                OTS block with an arbitrary body.
  M2 (HIGH)   — replay-ledger reconstruction uses ``block_requires_ots(b)``
                without cfg, narrower than the acceptance predicate; deleting the
                JSONL and restarting re-opens replay for require_ots-classified
                blocks (fail-OPEN).
  M3 (MEDIUM) — ``PersistentOTSReplayLedger`` has no inter-instance/file lock;
                two instances on one file both accept the same signature.
"""

import os
import tempfile
import types

import pytest

from wavelock.chain.Block import Block
from wavelock.crypto.ots_ledger import PersistentOTSReplayLedger
from wavelock.crypto.wavelock_ots import generate_ots_keypair, sign_ots, verify_ots
from wavelock.network import server

from attacks.ots_block_body_unbound import block_body_is_unbound
from attacks.ots_ledger_reconstruction_failopen import (
    replay_after_delete_and_restart,
)
from attacks.ots_ledger_concurrent_double_accept import two_instances_double_accept


def _fresh_ledger(tmp_path):
    return PersistentOTSReplayLedger(path=str(tmp_path / "ots_replay.jsonl"))


# ---------------------------------------------------------------------------
# M1 — OTS block signature does not bind the block body (Block.messages)
# ---------------------------------------------------------------------------

def test_break_ots_block_body_is_unbound():
    """LIVE BREAK: a block whose body != signed message is accepted."""
    assert block_body_is_unbound() is True


def test_break_published_signature_authors_arbitrary_block(tmp_path):
    """A victim's benign, unspent signature backs an attacker-chosen payload."""
    led = _fresh_ledger(tmp_path)
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    # Victim publishes a signature over a benign message; never broadcasts a block.
    victim_sig = sign_ots(sec, "I agree to terms v1")
    assert verify_ots(pub, "I agree to terms v1", victim_sig) is True

    # Attacker reuses it to author a block carrying a malicious transaction list.
    meta = server.build_ots_block_meta(pub, "I agree to terms v1", victim_sig)
    evil = Block(index=1, messages=["pay mallory 1000000"], previous_hash="0" * 64,
                 difficulty=1, block_type="OTS", meta=meta)
    assert server._verify_ots_block(evil, None, ledger=led) is True


@pytest.mark.xfail(strict=True, reason="M1: OTS sig does not bind Block.messages")
def test_secure_ots_block_must_bind_body(tmp_path):
    """DESIRED: an OTS block whose body is not what was signed must be rejected."""
    led = _fresh_ledger(tmp_path)
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "hello world")
    meta = server.build_ots_block_meta(pub, "hello world", sig)
    evil = Block(index=1, messages=["TRANSFER 1000000 TO ATTACKER"],
                 previous_hash="0" * 64, difficulty=1, block_type="OTS", meta=meta)
    # The body ("TRANSFER ...") was never authorised by the signature ("hello world").
    assert server._verify_ots_block(evil, None, ledger=led) is False


# ---------------------------------------------------------------------------
# M2 — replay-ledger reconstruction is narrower than acceptance (fail-open)
# ---------------------------------------------------------------------------

def test_break_replay_after_ledger_delete_and_restart():
    """LIVE BREAK: a consumed sig replays after JSONL delete + restart."""
    assert replay_after_delete_and_restart() is True


def test_break_reconstruction_predicate_is_narrower_than_acceptance():
    """The two predicates disagree for a require_ots-only OTS block."""
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "m")
    meta = {"ots_auth": {"public_key": pub, "message": "m", "signature": sig}}
    b = Block(index=1, messages=["ots"], previous_hash="0" * 64, difficulty=1,
              block_type="GENERIC", meta=meta)
    cfg = types.SimpleNamespace(require_ots=True)
    # Acceptance treats it as OTS; reconstruction (no cfg) does not.
    assert server.block_requires_ots(b, cfg) is True
    assert server.block_requires_ots(b) is False


@pytest.mark.xfail(strict=True,
                   reason="M2: reconstruction drops require_ots-classified blocks")
def test_secure_replay_blocked_after_ledger_delete_and_restart():
    """DESIRED: deleting the JSONL must not re-open replay if chain state remains."""
    assert replay_after_delete_and_restart() is False


# ---------------------------------------------------------------------------
# M3 — no inter-instance/file lock on the durable ledger (double-accept)
# ---------------------------------------------------------------------------

def test_break_two_ledger_instances_double_accept():
    """LIVE BREAK: two instances on one file both accept the same signature."""
    assert two_instances_double_accept() is True


@pytest.mark.xfail(strict=True,
                   reason="M3: no flock/O_EXCL; instances do not serialise")
def test_secure_two_ledger_instances_single_accept():
    """DESIRED: a file-level claim must let at most one instance accept."""
    assert two_instances_double_accept() is False
