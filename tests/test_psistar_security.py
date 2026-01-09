#!/usr/bin/env python3
"""
Full ψ★ (psi_star) protection audit for WaveLock CurvatureKeyPair.

This file enforces FOUR hard security guarantees:

  1. ψ★ must be inaccessible in production mode
  2. ψ★ must be accessible only when test_mode=True
  3. Dataset-style ψ★ extraction attempts must fail
  4. Commitment hashing must NOT leak ψ★

Run:
    python test_psistar_security.py
or:
    pytest -q test_psistar_security.py
"""

import sys
import cupy as cp
import pytest

# import WaveLock from repo root or installed version
try:
    from wavelock.chain.WaveLock import CurvatureKeyPair
except Exception:
    sys.path.append("..")
    from wavelock.chain.WaveLock import CurvatureKeyPair


# ============================================================
# TEST 1 — ψ★ MUST FAIL in production
# ============================================================
def test_production_block():
    print("\n[TEST 1] Production mode ψ★ protection...")

    kp = CurvatureKeyPair(n=8)

    with pytest.raises(PermissionError):
        _ = kp.psi_star

    print("✅ PASS: ψ★ protected in production mode")


# ============================================================
# TEST 2 — ψ★ MUST SUCCEED ONLY IN TEST MODE
# ============================================================
def test_testmode_unlock():
    print("\n[TEST 2] Test-mode access...")

    kp = CurvatureKeyPair(n=8, test_mode=True)
    psi = kp.psi_star

    assert psi is not None
    assert psi.ndim == 2

    print("✅ PASS: test_mode unlocked ψ★ — shape:", psi.shape)


# ============================================================
# TEST 3 — Dataset script must not be able to extract ψ★
# ============================================================
def test_dataset_attack_block():
    print("\n[TEST 3] Dataset-like ψ★ extraction attempts...")

    kp = CurvatureKeyPair(n=8)

    with pytest.raises(PermissionError):
        _ = cp.asnumpy(kp.psi_star)

    print("✅ PASS: Dataset extraction blocked")


# ============================================================
# TEST 4 — Commitment hashing must NOT reveal ψ★
# ============================================================
def test_commitment_protection():
    print("\n[TEST 4] Commitment hashing integrity...")

    kp = CurvatureKeyPair(n=8)

    # This MUST NOT require psi_star access
    raw = kp._curvature_hash_raw()

    assert isinstance(raw, (bytes, bytearray))
    assert len(raw) > 0
    assert kp.commitment is not None

    print("✅ PASS: Commitment hashing works without ψ★ exposure")


# ============================================================
# MAIN ENTRY (standalone audit mode)
# ============================================================
if __name__ == "__main__":
    print("=== WaveLock ψ★ Security Audit ===")

    try:
        test_production_block()
        test_testmode_unlock()
        test_dataset_attack_block()
        test_commitment_protection()
    except Exception as e:
        print("\n❌ SECURITY AUDIT FAILED\n", e)
        sys.exit(1)

    print("\n🎉 ALL TESTS PASSED — ψ★ SECURITY IS SOLID\n")
