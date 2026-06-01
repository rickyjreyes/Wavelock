"""
WaveLock-OTS red-team regression tests.

Full write-up: ``attacks/WAVELOCK_OTS_REDTEAM.md``.

This file has two kinds of tests:

1. ``test_break_*`` — DEMONSTRATIONS that the attack currently SUCCEEDS. They
   pass today and pin the vulnerability so it cannot regress silently. When a
   defect is fixed, the corresponding demonstration must be updated (the attack
   should then fail), which is the intended signal.

2. ``test_secure_*`` — assertions of the DESIRED secure behavior. They are
   marked ``xfail(strict=True)`` because the current code does NOT yet satisfy
   them: running pytest reports them as ``xfailed`` (= failing as expected).
   Once the matching fix lands, each flips to ``XPASS`` and — because
   ``strict=True`` — the suite turns red, forcing the marker to be removed and
   the property to become a hard guarantee.

These are intentionally added as failing/expected-failing tests per the audit
brief. Do not delete them to make CI green; fix the code instead.
"""

import copy

import pytest

from wavelock.crypto.wavelock_ots import (
    generate_ots_keypair,
    sign_ots,
    verify_ots,
)
from attacks.ots_key_substitution import (
    merkle_root_is_ignored,
    key_substitution_forgery,
)
from attacks.ots_signature_malleability import malleate
from attacks.ots_reuse_to_total_forgery import (
    collect_both_halves,
    forge_arbitrary,
)


# ---------------------------------------------------------------------------
# Finding A — merkle_root / pk_commitments are not bound in verify_ots
# ---------------------------------------------------------------------------


def test_break_merkle_root_is_ignored():
    """DEMO (passes now): a garbage merkle_root does not affect verification."""
    assert merkle_root_is_ignored() is True


def test_break_key_substitution_forgery():
    """DEMO (passes now): victim-fingerprint + attacker-commitments verifies."""
    assert key_substitution_forgery() is True


@pytest.mark.xfail(strict=True,
                   reason="Finding A: verify_ots never checks merkle_root "
                          "against pk_commitments.")
def test_secure_verify_rejects_inconsistent_merkle_root():
    kp = generate_ots_keypair(seed=bytes(range(32)))
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "legit message")
    mangled = copy.deepcopy(pub)
    mangled["merkle_root"] = "de" * 32
    # A correct verifier must recompute the Merkle root over pk_commitments and
    # reject when the published root does not match.
    assert verify_ots(mangled, "legit message", sig) is False


# ---------------------------------------------------------------------------
# Finding B — signature malleability (strong unforgeability broken)
# ---------------------------------------------------------------------------


def test_break_signature_is_malleable():
    """DEMO (passes now): several mutated signatures all still verify."""
    results = malleate()
    assert all(r["verifies"] for r in results)
    assert len({r["bytes"] for r in results}) >= 3


@pytest.mark.xfail(strict=True,
                   reason="Finding B: one_time_key_id is carried in the "
                          "signature but never bound at verify time.")
def test_secure_verify_rejects_mutated_key_id():
    kp = generate_ots_keypair(seed=bytes(range(32)))
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "pay alice 5")
    mutated = copy.deepcopy(sig)
    mutated["one_time_key_id"] = "ATTACKER-CHOSEN-ID"
    assert verify_ots(pub, "pay alice 5", mutated) is False


@pytest.mark.xfail(strict=True,
                   reason="Finding B: message_digest may be dropped; verify "
                          "accepts a signature with the field removed.")
def test_secure_verify_requires_message_digest_field():
    kp = generate_ots_keypair(seed=bytes(range(32)))
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "m")
    no_digest = copy.deepcopy(sig)
    no_digest.pop("message_digest", None)
    # A strict verifier should not silently accept a signature missing a
    # field it claims to bind.
    assert verify_ots(pub, "m", no_digest) is False


# ---------------------------------------------------------------------------
# Finding C — one-time reuse is catastrophic (canonical Lamport break)
# ---------------------------------------------------------------------------


def test_break_reuse_enables_total_forgery():
    """DEMO (passes now): ~48 reuses harvest both halves → forge anything."""
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    known = collect_both_halves(sec, pub, n_signatures=48)
    forged = forge_arbitrary(pub, known, "TRANSFER 1000000 TO ATTACKER")
    assert forged is not None
    assert verify_ots(pub, "TRANSFER 1000000 TO ATTACKER", forged) is True


# ---------------------------------------------------------------------------
# Finding D — one-time enforcement is advisory file state (copy/crash/race)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(strict=True,
                   reason="Finding D: one-time-ness is only a mutable `used` "
                          "bool in the secret dict; a copy made before signing "
                          "signs again. There is no tamper-resistant guard.")
def test_secure_copied_key_cannot_sign_twice():
    kp = generate_ots_keypair(entropy_bits=256)
    pub = kp["public_key"]
    original = kp["secret_key"]
    # Simulate a file copy taken BEFORE the first signature.
    copy_before_use = copy.deepcopy(original)

    sig1 = sign_ots(original, "msg one")
    assert verify_ots(pub, "msg one", sig1)

    # The copy still has used=False, so it signs a *second* message under the
    # same one-time key. A robust one-time scheme must make this impossible.
    sig2 = sign_ots(copy_before_use, "msg two")
    assert verify_ots(pub, "msg two", sig2) is False
