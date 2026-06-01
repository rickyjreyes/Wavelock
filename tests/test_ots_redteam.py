"""
WaveLock-OTS red-team regression tests.

Full write-up: ``attacks/WAVELOCK_OTS_REDTEAM.md`` (see the STATUS section at the
top for what is fixed vs. inherent).

History
-------
This file originally pinned four findings (A–D) with a mix of "the attack
succeeds today" demonstrations and ``xfail(strict=True)`` assertions of the
*desired* secure behavior. Findings A and B are now FIXED in
``wavelock/crypto/wavelock_ots.py``:

* **A** — ``verify_ots``/``load_public_key`` recompute the Merkle root from
  ``pk_commitments`` and recompute the public-key fingerprint over the canonical
  public key, rejecting garbage roots and fingerprint/key-substitution.
* **B** — ``verify_ots`` enforces the exact canonical signature field set
  (no missing, no extra), the scheme/version/hash_alg constants, a present and
  correct ``message_digest``, and binds ``one_time_key_id`` /
  ``public_key_fingerprint`` / ``params_hash`` / ``psi_commitment`` to the key.

So the former ``xfail`` secure-behavior tests now pass normally (the markers are
removed), and the former "attack succeeds" demonstrations now assert the attack
is DEFENDED. The runnable PoC scripts in ``attacks/`` are preserved as
regression evidence.

Findings C and D are NOT cryptographically fixed by A/B:

* **C** — OTS reuse → total forgery is *inherent* to Lamport-style OTS. The
  reuse PoC is preserved and still produces a verifying forgery once the key is
  reused; the only defense is to never reuse the key (and to reject duplicate
  ``one_time_key_id`` at the ledger).
* **D** — file ``used=true`` is advisory. A host-local atomic key-state registry
  now stops a copied secret key from signing twice *on the same host*; a copy
  moved to another host still bypasses it, so a server/ledger duplicate-key
  check (modeled by ``OTSReplayLedger``) is required in production.
"""

import copy

import pytest

from wavelock.crypto.wavelock_ots import (
    VERSION,
    OTSKeyReuseError,
    OTSReplayLedger,
    generate_ots_keypair,
    sign_ots,
    verify_ots,
    public_key_fingerprint,
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


def _kp_and_sig(message="legit message"):
    kp = generate_ots_keypair(seed=bytes(range(32)))
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, message)
    return pub, sec, sig


# ---------------------------------------------------------------------------
# Finding A — merkle_root / pk_commitments are now bound in verify/load
# ---------------------------------------------------------------------------


def test_attack_merkle_root_substitution_is_defended():
    """The Finding-A PoC must now FAIL (garbage merkle_root is rejected)."""
    assert merkle_root_is_ignored() is False


def test_attack_key_substitution_is_defended():
    """The Finding-A PoC must now FAIL (victim fingerprint + attacker keys)."""
    assert key_substitution_forgery() is False


def test_secure_verify_rejects_inconsistent_merkle_root():
    """garbage merkle_root fails: recomputed root != stored root → reject."""
    pub, _, sig = _kp_and_sig()
    mangled = copy.deepcopy(pub)
    mangled["merkle_root"] = "de" * 32
    assert verify_ots(mangled, "legit message", sig) is False


def test_secure_substituted_commitments_under_victim_fingerprint_fail():
    """substituted pk_commitments under victim fingerprint fails."""
    victim = generate_ots_keypair(seed=bytes(range(32)))
    attacker = generate_ots_keypair(seed=bytes(range(100, 132)))
    vpub = victim["public_key"]
    apub, asec = attacker["public_key"], attacker["secret_key"]
    asig = sign_ots(asec, "TRANSFER 1000000 TO ATTACKER")

    forged = copy.deepcopy(apub)
    # Keep the victim's identity fields; keep attacker commitments/params.
    forged["one_time_key_id"] = vpub["one_time_key_id"]
    forged["merkle_root"] = vpub["merkle_root"]
    forged["public_key_fingerprint"] = vpub["public_key_fingerprint"]
    assert verify_ots(forged, "TRANSFER 1000000 TO ATTACKER", asig) is False


def test_secure_reordered_commitments_fail_deterministically():
    """reordered commitments fail deterministically.

    Design choice: order is significant (the Merkle root and fingerprint are
    computed over pk_commitments in order). Swapping two rows changes the
    recomputed root/fingerprint, so the stored values no longer match → reject.
    """
    pub, _, sig = _kp_and_sig()
    reordered = copy.deepcopy(pub)
    reordered["pk_commitments"][0], reordered["pk_commitments"][1] = (
        reordered["pk_commitments"][1],
        reordered["pk_commitments"][0],
    )
    assert verify_ots(reordered, "legit message", sig) is False


# ---------------------------------------------------------------------------
# Finding A — public-key canonical shape
# ---------------------------------------------------------------------------


def test_secure_public_key_extra_field_fails():
    """public key extra field fails (non-canonical shape rejected)."""
    pub, _, sig = _kp_and_sig()
    bad = copy.deepcopy(pub)
    bad["surprise"] = "extra"
    assert verify_ots(bad, "legit message", sig) is False


def test_secure_public_key_missing_field_fails():
    """public key missing field fails (non-canonical shape rejected)."""
    pub, _, sig = _kp_and_sig()
    bad = copy.deepcopy(pub)
    bad.pop("psi_commitment")
    assert verify_ots(bad, "legit message", sig) is False


# ---------------------------------------------------------------------------
# Finding B — signature malleability is now closed (strong unforgeability)
# ---------------------------------------------------------------------------


def test_attack_signature_malleability_is_defended():
    """Every Finding-B mutation must now FAIL to verify."""
    results = malleate()
    assert all(r["verifies"] is False for r in results)


def test_secure_verify_rejects_mutated_key_id():
    """changed one_time_key_id fails."""
    pub, _, sig = _kp_and_sig("pay alice 5")
    mutated = copy.deepcopy(sig)
    mutated["one_time_key_id"] = "ATTACKER-CHOSEN-ID"
    assert verify_ots(pub, "pay alice 5", mutated) is False


def test_secure_verify_rejects_changed_version():
    """changed version fails."""
    pub, _, sig = _kp_and_sig()
    mutated = copy.deepcopy(sig)
    mutated["version"] = VERSION + 1
    assert verify_ots(pub, "legit message", mutated) is False


def test_secure_verify_rejects_changed_hash_alg():
    """changed hash_alg fails."""
    pub, _, sig = _kp_and_sig()
    mutated = copy.deepcopy(sig)
    mutated["hash_alg"] = "MD5"
    assert verify_ots(pub, "legit message", mutated) is False


def test_secure_verify_requires_message_digest_field():
    """missing message_digest fails (no missing-field bypass)."""
    pub, _, sig = _kp_and_sig("m")
    no_digest = copy.deepcopy(sig)
    no_digest.pop("message_digest", None)
    assert verify_ots(pub, "m", no_digest) is False


def test_secure_verify_rejects_wrong_message_digest():
    """wrong message_digest fails (recomputed digest must match)."""
    pub, _, sig = _kp_and_sig("m")
    bad = copy.deepcopy(sig)
    bad["message_digest"] = "00" * 32
    assert verify_ots(pub, "m", bad) is False


def test_secure_verify_rejects_extra_signature_field():
    """extra signature field fails (exact field set enforced)."""
    pub, _, sig = _kp_and_sig()
    bad = copy.deepcopy(sig)
    bad["injected"] = {"arbitrary": "payload"}
    assert verify_ots(pub, "legit message", bad) is False


def test_secure_verify_rejects_removed_signature_field():
    """removed signature field fails (exact field set enforced)."""
    pub, _, sig = _kp_and_sig()
    bad = copy.deepcopy(sig)
    bad.pop("psi_commitment")
    assert verify_ots(pub, "legit message", bad) is False


def test_secure_verify_rejects_modified_params_hash():
    """modified params_hash fails (sig params_hash must match the key)."""
    pub, _, sig = _kp_and_sig()
    bad = copy.deepcopy(sig)
    bad["params_hash"] = "00" * 32
    assert verify_ots(pub, "legit message", bad) is False


def test_secure_verify_rejects_modified_psi_commitment():
    """modified psi_commitment fails (sig psi_commitment must match the key)."""
    pub, _, sig = _kp_and_sig()
    bad = copy.deepcopy(sig)
    bad["psi_commitment"] = "00" * 32
    assert verify_ots(pub, "legit message", bad) is False


def test_secure_verify_rejects_mutated_fingerprint():
    """A signature carrying the wrong public_key_fingerprint is rejected."""
    pub, _, sig = _kp_and_sig()
    bad = copy.deepcopy(sig)
    bad["public_key_fingerprint"] = "00" * 32
    assert verify_ots(pub, "legit message", bad) is False


# ---------------------------------------------------------------------------
# Finding C — one-time reuse is catastrophic (INHERENT to Lamport-style OTS)
# ---------------------------------------------------------------------------


def test_break_reuse_enables_total_forgery():
    """DEMO (still succeeds): reuse harvests both halves → forge anything.

    This is intentionally preserved. The A/B hardening does NOT fix C — strict
    field/fingerprint checks cannot stop an attacker who has observed both
    Lamport halves and assembles a fully canonical signature. The only defense
    is to never reuse the key (Finding D mitigation) and to reject duplicate
    one_time_key_id at the ledger (``OTSReplayLedger``).
    """
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    known = collect_both_halves(sec, pub, n_signatures=48)
    forged = forge_arbitrary(pub, known, "TRANSFER 1000000 TO ATTACKER")
    assert forged is not None
    assert verify_ots(pub, "TRANSFER 1000000 TO ATTACKER", forged) is True


# ---------------------------------------------------------------------------
# Finding D — one-time enforcement: host-local registry + ledger model
# ---------------------------------------------------------------------------


def test_signing_twice_with_same_local_key_is_rejected_by_default():
    """Default: a second signature with the same in-memory key is refused."""
    kp = generate_ots_keypair(entropy_bits=256)
    sec = kp["secret_key"]
    sign_ots(sec, "first")
    with pytest.raises(OTSKeyReuseError):
        sign_ots(sec, "second")


def test_secure_copied_key_cannot_sign_twice():
    """A copy taken BEFORE first use cannot sign again ON THE SAME HOST.

    The host-local atomic key-state registry claims the one_time_key_id on the
    first signature; the copy (still used=False) is then refused. NOTE: a copy
    moved to a DIFFERENT host with no shared registry still bypasses this — see
    docs/WAVELOCK_MERKLE_ROADMAP.md; production MUST reject duplicate
    one_time_key_id at the verifier/ledger layer.
    """
    kp = generate_ots_keypair(entropy_bits=256)
    pub = kp["public_key"]
    original = kp["secret_key"]
    copy_before_use = copy.deepcopy(original)

    sig1 = sign_ots(original, "msg one")
    assert verify_ots(pub, "msg one", sig1)

    with pytest.raises(OTSKeyReuseError):
        sign_ots(copy_before_use, "msg two")


def test_ledger_rejects_duplicate_one_time_key_id():
    """A server/ledger MUST reject a second accepted sig under one key id.

    Models the production duplicate-key rejection that defends against Finding C
    even if a copied key signs twice on another host (``allow_reuse`` here
    simulates that bypass producing a second, individually-valid signature).
    """
    ledger = OTSReplayLedger()
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]

    sig1 = sign_ots(sec, "pay alice 5")
    assert ledger.accept(pub, "pay alice 5", sig1) is True

    # A second, individually-valid signature under the SAME one_time_key_id
    # (as a copied key on another host could produce) is rejected by the ledger.
    sig2 = sign_ots(sec, "pay bob 9", allow_reuse=True)
    assert verify_ots(pub, "pay bob 9", sig2) is True  # cryptographically valid
    assert ledger.accept(pub, "pay bob 9", sig2) is False  # but replay-rejected


def test_ledger_rejects_invalid_signature_without_consuming():
    """Fail-closed: an invalid signature is never accepted or recorded."""
    ledger = OTSReplayLedger()
    kp = generate_ots_keypair(entropy_bits=256)
    pub, sec = kp["public_key"], kp["secret_key"]
    sig = sign_ots(sec, "ok")
    tampered = copy.deepcopy(sig)
    tampered["revealed_slices"][0] = "00" * 32
    assert ledger.accept(pub, "ok", tampered) is False
    # The key id was not consumed, so the genuine signature still goes through.
    assert ledger.accept(pub, "ok", sig) is True
