# tests/test_signature_commitment_v2.py
import os, copy, hashlib, json
import numpy as np
import pytest

try:
    import cupy as cp
except Exception:
    pytest.skip("CuPy not available for this environment", allow_module_level=True)

import wavelock.chain.WaveLock as WL


def make_kp(n=4, seed=123):
    # Explicit test-mode unlock
    kp = WL.CurvatureKeyPair(n=n, seed=seed, test_mode=True)
    # sanity: WLv2 by default
    assert str(kp.commitment).startswith(WL.SCHEMA_V2 + ":"), "Expected WLv2 commitment"
    return kp


def test_control_passes():
    kp = make_kp()
    msg = "hello wave-lock"
    sig = kp.sign(msg)
    assert kp.verify(msg, sig), "Control path should verify"


def test_bitflip_in_psi_breaks_both_commitment_and_signature():
    kp = make_kp()
    msg = "demo"
    sig = kp.sign(msg)
    stored_commit = kp.commitment

    # clone and flip 1 bit in ψ★ (test-only allowed)
    psi_bytes = cp.asnumpy(kp.psi_star).view(np.uint8)
    tampered = psi_bytes.copy()
    tampered[0] ^= 0x01  # flip 1 bit
    psi_tampered = cp.asarray(
        tampered.view(np.float64).reshape(kp.psi_star.shape)
    )

    kp2 = WL.CurvatureKeyPair(n=4, test_mode=True)
    kp2.psi_star = psi_tampered
    kp2.psi_0 = cp.zeros_like(kp2.psi_star)
    kp2.commitment = stored_commit  # pretend same commitment

    assert not kp2.verify(msg, sig), "1-bit ψ★ change must invalidate verification"


def test_header_param_change_breaks_commitment_and_signature(monkeypatch):
    """
    Simulate a 'header tamper' by changing a physics constant on the verifier.
    Since SIGv2 and WLv2 bind header→(alpha,beta,theta,epsilon,delta),
    verification must fail when the verifier uses different constants.
    """
    kp = make_kp()
    msg = "header-bound"
    sig = kp.sign(msg)
    stored_commit = kp.commitment

    kp2 = WL.CurvatureKeyPair(n=4, test_mode=True)
    kp2.psi_star = kp.psi_star.copy()
    kp2.psi_0 = cp.zeros_like(kp2.psi_star)
    kp2.commitment = stored_commit

    old_alpha = WL.alpha
    monkeypatch.setattr(WL, "alpha", old_alpha * 1.0001)

    try:
        assert not kp2.verify(msg, sig), "Header (param) change must invalidate verification"
    finally:
        monkeypatch.setattr(WL, "alpha", old_alpha)


def test_message_change_breaks_signature():
    kp = make_kp()
    msg = "original-message"
    sig = kp.sign(msg)

    assert kp.verify(msg, sig), "control sign/verify should pass"
    assert not kp.verify(msg + "!", sig), "message 1-char change must fail"
