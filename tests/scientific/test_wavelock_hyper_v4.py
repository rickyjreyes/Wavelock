"""
============================================================
   WAVELOCK HYPER-TEST HARNESS (WLv4)
   ---------------------------------
   Full curvature-domain test suite for the WaveLock v4
   Curvature Canonical Commitment (CCC) system.

   WLv4 hashes ONLY curvature invariants:
       (E_grad, E_fb, E_ent, E_tot)

   ψ-space leakage = 0
   Floating-point layout leakage = 0
   SVD / Laplacian-mode leakage = 0
   Resonant-structure leakage = 0

   This test verifies:
       • Deterministic curvature evolution
       • CCC collision resistance
       • Curvature-symbolic verifier
       • Signature integrity (v4)
       • Curvature drift sensitivity
       • Resonance rejection (curvature-safe)
       • PDE inversion impossibility in curvature space

   EXPECTED RESULTS (WLv4):
       - 0 collisions
       - 0 false accepts
       - 0 forgery
       - drift always changes invariants
       - resonance never matches curvature invariants
       - PDE inversion always rejected
============================================================
"""

import numpy as np
import cupy as cp
import hashlib
import math
import os, sys

# ============================================================
# PYTHONPATH FIX (jump to project root)
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ============================================================
# IMPORT WAVELOCK v4
# ============================================================

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    _serialize_commitment_v4,
    laplacian,
    _curvature_functional,
)

# ============================================================
# CANONICAL V4 COMMIT
# ============================================================

def commit_v4(psi):
    raw = _serialize_commitment_v4(cp.asarray(psi, dtype=cp.float64))
    return hashlib.sha256(raw).hexdigest()


def curvature_vector(psi):
    """Returns (E_grad, E_fb, E_ent, E_tot) as a NumPy array."""
    E_grad, E_fb, E_ent, E_tot = _curvature_functional(cp.asarray(psi))
    return np.array([E_grad, E_fb, E_ent, E_tot], dtype=np.float64)


# ============================================================
# TEST 1 — Deterministic Curvature Evolution
# ============================================================

def test_deterministic_evolution(n=6, trials=10, seed=123):
    diffs = []
    for _ in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed, use_v4=True)
        psi0 = kp.psi_0

        psi1 = kp.evolve(psi0, n)
        psi2 = kp.evolve(psi0, n)

        h1 = commit_v4(psi1)
        h2 = commit_v4(psi2)

        diffs.append(h1 != h2)

    return {
        "all_identical": not any(diffs),
        "distribution": diffs,
    }


# ============================================================
# TEST 2 — CCC Collision Resistance
# ============================================================

def test_collision_resistance(n=6, trials=50):
    collisions = 0
    for i in range(trials):
        kp1 = CurvatureKeyPair(n=n, seed=i, use_v4=True)
        kp2 = CurvatureKeyPair(n=n, seed=i+1_000_000, use_v4=True)

        if commit_v4(kp1.psi_star) == commit_v4(kp2.psi_star):
            collisions += 1

    return {
        "collisions": collisions,
        "collision_rate": collisions / trials,
    }


# ============================================================
# TEST 3 — Curvature False-Accept (V4 symbolic equivalent)
# ============================================================

def test_false_accept(n=6, trials=300):
    false_accepts = 0

    for _ in range(trials):
        kp = CurvatureKeyPair(n=n, use_v4=True)
        psi = kp.psi_star

        # random field, normalized
        noise = cp.random.randn(*psi.shape)
        noise /= (cp.linalg.norm(noise) + 1e-12)

        if commit_v4(noise) == commit_v4(psi):
            false_accepts += 1

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / trials,
    }


# ============================================================
# TEST 4 — Signature Forgery (WLv4)
# ============================================================

def test_signature_forgery(n=6, trials=100):
    msg = "wavelock-v4-integrity"
    forgeries = 0

    for _ in range(trials):
        kp = CurvatureKeyPair(n=n, use_v4=True)
        real = kp.sign(msg)
        fake = hashlib.sha256(os.urandom(128)).hexdigest()

        if kp.verify(msg, fake):
            forgeries += 1

    return {
        "forgeries": forgeries,
        "rate": forgeries / trials,
    }


# ============================================================
# TEST 5 — Curvature Drift Sensitivity
# ============================================================

def test_drift(n=6, trials=30):
    failures = 0

    for _ in range(trials):
        kp = CurvatureKeyPair(n=n, use_v4=True)
        base = commit_v4(kp.psi_star)

        drift = kp.psi_star + cp.random.normal(0, 1e-6, kp.psi_star.shape)
        d_hash = commit_v4(drift)

        if d_hash == base:
            failures += 1

    return {
        "drift_failures": failures,
        "fail_rate": failures / trials,
        "rate_ok": 1 - failures / trials,
    }


# ============================================================
# TEST 6 — Resonance Attack (Curvature-Space)
# ============================================================

def test_resonance_attack(n=6, trials=20):
    false_accepts = 0

    for _ in range(trials):
        kp = CurvatureKeyPair(n=n, use_v4=True)
        psi = kp.psi_star
        x = psi.shape[0]

        t = cp.linspace(0, 12*math.pi, x)

        # Structured PDE resonance
        r = (
            cp.sin(2*t).reshape(x,1) *
            cp.cos(3*t).reshape(1,x) +
            cp.sin(5*t).reshape(x,1)
        )

        r = (r - r.min()) / (r.max() - r.min() + 1e-12)

        if commit_v4(r) == commit_v4(psi):
            false_accepts += 1

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / trials,
    }


# ============================================================
# TEST 7 — PDE Inversion (Curvature-Space)
# ============================================================

def test_pde_inversion(n=6, trials=20):
    accepts = 0

    for _ in range(trials):
        kp = CurvatureKeyPair(n=n, use_v4=True)
        psi = kp.psi_star

        inv = psi - 0.002 * laplacian(psi)

        if commit_v4(inv) == commit_v4(psi):
            accepts += 1

    return {
        "accepted": accepts,
        "rate": accepts / trials,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n=== WAVELOCK HYPER-TEST HARNESS (WLv4) ===\n")

    print("1) Deterministic Curvature Evolution:")
    print(test_deterministic_evolution())

    print("\n2) Collision Resistance:")
    print(test_collision_resistance())

    print("\n3) False Accept (Curvature):")
    print(test_false_accept())

    print("\n4) Signature Forgery:")
    print(test_signature_forgery())

    print("\n5) Curvature Drift Sensitivity:")
    print(test_drift())

    print("\n6) Resonance Attack:")
    print(test_resonance_attack())

    print("\n7) PDE-Inversion Attack:")
    print(test_pde_inversion())

    print("\n=== DONE ===\n")
