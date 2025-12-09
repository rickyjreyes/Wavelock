# wavelock_hypertest.py (WLv3)
# Fully corrected, hardened, and reinforced Hyper-Test Harness for WaveLock

import numpy as np
import cupy as cp
import hashlib
import time
import math
import os, sys

# ============================================================
# PYTHON PATH FIX
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# ============================================================
# IMPORT WAVELOCK CORE
# ============================================================

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    symbolic_verifier,
    laplacian,
    _serialize_commitment_v2,
)

# ============================================================
# UTILITIES
# ============================================================

def sha256_bits(arr):
    """Hash CuPy array -> 256-bit numpy vector (bit array)."""
    raw = cp.asnumpy(arr).tobytes()
    h = hashlib.sha256(raw).digest()
    return np.unpackbits(np.frombuffer(h, dtype=np.uint8))

def sha256_hex(arr):
    """Hash CuPy array -> hex digest."""
    raw = cp.asnumpy(arr).tobytes()
    return hashlib.sha256(raw).hexdigest()

def bit_diff(a, b):
    return int(np.sum(a != b))

# ============================================================
# TEST 1: DETERMINISTIC ψ-EVOLUTION
# ============================================================

def test_deterministic_evolution(n=6, trials=20, seed=1234):
    diffs = []
    for _ in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed)

        psi0 = kp.psi_0.astype(cp.float64)
        psi1 = kp.evolve(psi0, n)
        psi2 = kp.evolve(psi0, n)

        if psi1.shape != psi2.shape:
            return {"error": "Shape mismatch"}

        h1 = sha256_bits(psi1)
        h2 = sha256_bits(psi2)

        diffs.append(bit_diff(h1, h2))

    return {
        "max_diff": max(diffs),
        "min_diff": min(diffs),
        "all_zero": all(d == 0 for d in diffs),
        "distribution": diffs,
    }

# ============================================================
# TEST 2: ψ*-COLLISION RESISTANCE
# ============================================================

def test_collision_resistance(n=6, trials=50):
    collisions = 0
    for i in range(trials):
        kp1 = CurvatureKeyPair(n=n, seed=i)
        kp2 = CurvatureKeyPair(n=n, seed=i + 777777)

        c1 = sha256_hex(kp1.psi_star)
        c2 = sha256_hex(kp2.psi_star)

        if c1 == c2:
            collisions += 1

    return {
        "collisions": collisions,
        "collision_rate": collisions / trials,
    }

# ============================================================
# TEST 3: SYMBOLIC VERIFIER FALSE-ACCEPTANCE
# ============================================================

def test_symbolic_verifier_false_accept(n=6, trials=200):
    false_accepts = 0
    for _ in range(trials):
        kp = CurvatureKeyPair(n=n)
        psi_star = kp.psi_star.astype(cp.float32)

        # adversarial random normalized noise field
        noise = cp.random.randn(*psi_star.shape).astype(cp.float32)
        noise = noise / (cp.linalg.norm(noise) + 1e-12)

        if symbolic_verifier(noise, psi_star):
            false_accepts += 1

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / trials,
    }

# ============================================================
# TEST 4: SIGNATURE FORGERY
# ============================================================

def test_signature_forgery(n=6, trials=100):
    forgeries = 0
    msg = "hyper-test-msg"

    for _ in range(trials):
        kp = CurvatureKeyPair(n=n)

        real_sig = kp.sign(msg)
        # random hex string of incorrect form
        fake_sig = hashlib.sha256(os.urandom(64)).hexdigest()

        if kp.verify(msg, fake_sig):
            forgeries += 1

    return {
        "forgeries": forgeries,
        "rate": forgeries / trials,
    }

# ============================================================
# TEST 5: DRIFT SENSITIVITY
# ============================================================

def test_drift(n=6, trials=30):
    failures = 0
    for _ in range(trials):
        kp = CurvatureKeyPair(n=n)
        base_hex = sha256_hex(kp.psi_star)

        drift = (kp.psi_star +
                 cp.random.normal(0, 1e-6, kp.psi_star.shape).astype(cp.float64))

        drift_hex = sha256_hex(drift)

        if base_hex == drift_hex:
            failures += 1

    return {
        "sensitivity_failures": failures,
        "fail_rate": failures / trials,
        "rate_ok": 1 - failures / trials,
    }

# ============================================================
# TEST 6: RESONANCE ATTACK
# ============================================================

def test_resonance_attack(n=6, trials=20):
    false_accepts = 0

    for _ in range(trials):
        kp = CurvatureKeyPair(n=n)
        psi_star = kp.psi_star

        x = psi_star.shape[0]
        t = cp.linspace(0, 20 * math.pi, x)

        # stronger, structured 2D resonance field
        r = (cp.sin(3 * t).reshape(x,1) *
             cp.cos(5 * t).reshape(1,x) +
             cp.sin(7 * t).reshape(x,1))

        # normalize
        r = (r - r.min()) / (r.max() - r.min() + 1e-12)

        if symbolic_verifier(r.astype(cp.float32), psi_star):
            false_accepts += 1

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / trials,
    }

# ============================================================
# TEST 7: PDE INVERSION ATTEMPT
# ============================================================

def test_pde_inversion(n=6, trials=20):
    accepts = 0

    for _ in range(trials):
        kp = CurvatureKeyPair(n=n)
        psi_star = kp.psi_star

        # naïve inverse-like perturbation
        inv = psi_star - 0.001 * laplacian(psi_star)

        if symbolic_verifier(inv.astype(cp.float32), psi_star):
            accepts += 1

    return {
        "accepted": accepts,
        "rate": accepts / trials,
    }





# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n=== WAVELOCK HYPER-TEST HARNESS (WLv3) ===\n")

    print("1) Deterministic Evolution:")
    print(test_deterministic_evolution())

    print("\n2) Collision Resistance:")
    print(test_collision_resistance())

    print("\n3) Symbolic Verifier False Acceptance:")
    print(test_symbolic_verifier_false_accept())

    print("\n4) Signature Forgery:")
    print(test_signature_forgery())

    print("\n5) Drift Sensitivity:")
    print(test_drift())

    print("\n6) Resonance Attack:")
    print(test_resonance_attack())

    print("\n7) PDE Inversion Attack:")
    print(test_pde_inversion())

    print("\n=== DONE ===\n")
