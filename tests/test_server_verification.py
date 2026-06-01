"""
Server verification must FAIL CLOSED.

These tests pin the security-relevant behavior of the P2P server's curvature
verification after removing the non-strict bypasses:

  * Trust-list membership alone NEVER accepts a block.
  * Missing message/signature is rejected.
  * Unpublished proof material is rejected (no "allow due to policy").
  * Untrusted commitment is rejected.
  * A genuinely valid (legacy) signature with published proof is accepted.

The legacy SIGv2 mechanism is still what the server verifies, but it now fails
closed; new deployments should move to WaveLock-OTS.
"""

import types

import numpy as np
import pytest

from wavelock.network import server
from wavelock.chain.WaveLock import CurvatureKeyPair


def _block(messages):
    return types.SimpleNamespace(messages=messages, index=1)


def _fields(msg, sig, com):
    return [f"message: {msg}", f"signature: {sig}", f"commitment: {com}"]


@pytest.fixture
def legacy_signed():
    kp = CurvatureKeyPair(n=4, seed=5, test_mode=True)
    com = kp.commitment
    msg = "hello"
    sig = kp.sign(msg)
    psi = np.asarray(kp.psi_star, dtype=np.float64).copy()
    return {"kp": kp, "com": com, "msg": msg, "sig": sig, "psi": psi}


def test_trust_only_does_not_pass(monkeypatch, legacy_signed):
    """Commitment trusted but no published proof => REJECT (not accept)."""
    com = legacy_signed["com"]
    monkeypatch.setattr(server, "_load_trusted_commitments", lambda *a, **k: [com])
    monkeypatch.setattr(server, "_load_published_psi", lambda c: None)
    b = _block(_fields(legacy_signed["msg"], legacy_signed["sig"], com))
    assert server._verify_curvature(b, None) is False


def test_untrusted_commitment_rejected(monkeypatch, legacy_signed):
    monkeypatch.setattr(server, "_load_trusted_commitments", lambda *a, **k: [])
    monkeypatch.setattr(server, "_load_published_psi",
                        lambda c: legacy_signed["psi"])
    b = _block(_fields(legacy_signed["msg"], legacy_signed["sig"],
                       legacy_signed["com"]))
    assert server._verify_curvature(b, None) is False


def test_missing_signature_rejected(monkeypatch, legacy_signed):
    com = legacy_signed["com"]
    monkeypatch.setattr(server, "_load_trusted_commitments", lambda *a, **k: [com])
    monkeypatch.setattr(server, "_load_published_psi",
                        lambda c: legacy_signed["psi"])
    b = _block([f"message: hello", f"commitment: {com}"])  # no signature
    assert server._verify_curvature(b, None) is False


def test_unpublished_proof_rejected(monkeypatch, legacy_signed):
    com = legacy_signed["com"]
    monkeypatch.setattr(server, "_load_trusted_commitments", lambda *a, **k: [com])
    monkeypatch.setattr(server, "_load_published_psi", lambda c: None)
    b = _block(_fields(legacy_signed["msg"], legacy_signed["sig"], com))
    assert server._verify_curvature(b, None) is False


def test_invalid_signature_rejected(monkeypatch, legacy_signed):
    com = legacy_signed["com"]
    monkeypatch.setattr(server, "_load_trusted_commitments", lambda *a, **k: [com])
    monkeypatch.setattr(server, "_load_published_psi",
                        lambda c: legacy_signed["psi"])
    bad_sig = "deadbeef" * 8
    b = _block(_fields(legacy_signed["msg"], bad_sig, com))
    assert server._verify_curvature(b, None) is False


def test_valid_signature_accepted(monkeypatch, legacy_signed):
    com = legacy_signed["com"]
    monkeypatch.setattr(server, "_load_trusted_commitments", lambda *a, **k: [com])
    monkeypatch.setattr(server, "_load_published_psi",
                        lambda c: legacy_signed["psi"])
    b = _block(_fields(legacy_signed["msg"], legacy_signed["sig"], com))
    assert server._verify_curvature(b, None) is True


def test_non_strict_bypass_removed_or_disabled(monkeypatch, legacy_signed):
    """The old `if not cfg.require_full_verify: return True` bypass is gone.

    Even with a cfg that sets require_full_verify=False, a block with a trusted
    commitment but no valid signature/proof must NOT be accepted.
    """
    com = legacy_signed["com"]
    monkeypatch.setattr(server, "_load_trusted_commitments", lambda *a, **k: [com])
    monkeypatch.setattr(server, "_load_published_psi", lambda c: None)
    cfg = types.SimpleNamespace(require_full_verify=False)
    b = _block(_fields(legacy_signed["msg"], legacy_signed["sig"], com))
    assert server._verify_curvature(b, cfg) is False
