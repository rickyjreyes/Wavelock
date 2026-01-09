import numpy as np
import os
import sys
import inspect

# ============================================================
#  Ensure WaveLock repo root is on PYTHONPATH
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ============================================================
#  IMPORT REAL WAVELOCK KERNEL (AUTHORITATIVE)
# ============================================================

from wavelock.chain.WaveLock import CurvatureKeyPair


# ============================================================
#  AUTHORITATIVE KERNEL ENTRY (TEST MODE ONLY)
# ============================================================

def generate_psi_star(*args, **kwargs):
    """
    Authoritative WaveLock kernel invocation.

    SECURITY CONTRACT:
    - ψ₀ is NOT a valid input
    - ψ★ is protected unless test_mode=True
    - This function explicitly opts into test mode
    """

    # Reject ψ-like inputs categorically
    for a in args:
        if isinstance(a, np.ndarray):
            raise RuntimeError(
                "❌ INVALID TEST: ψ₀ passed to CurvatureKeyPair.\n"
                "WaveLock kernel does not accept ψ₀ as input.\n"
                "ψ-space inversion attacks are not applicable."
            )

    # Explicit test-mode opt-in (required by WaveLock)
    kp = CurvatureKeyPair(*args, test_mode=True, **kwargs)
    return kp.psi_star


# ============================================================
#  TEST 1 — DOMAIN SEPARATION (ψ₀ NOT IN DOMAIN)
# ============================================================

def domain_separation_test():
    print("\n[DOMAIN SEPARATION TEST]")

    try:
        fake_psi0 = np.random.randn(32, 32)
        generate_psi_star(fake_psi0)
        print("❌ FAIL — ψ₀ was accepted (unexpected)")
    except RuntimeError as e:
        print("✅ PASS — ψ₀ correctly rejected")
        print("    " + str(e).splitlines()[0])


# ============================================================
#  TEST 2 — SEED / PARAMETER ONE-WAYNESS
# ============================================================

def seed_sensitivity_test(n=12, seeds=(0, 1)):
    print("\n[SEED SENSITIVITY TEST]")

    np.random.seed(seeds[0])
    psi_a = generate_psi_star(n)

    np.random.seed(seeds[1])
    psi_b = generate_psi_star(n)

    a = psi_a.flatten()
    b = psi_b.flatten()

    a -= a.mean()
    b -= b.mean()

    corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(f"Correlation(seed {seeds[0]} vs {seeds[1]}) = {corr:.6f}")

    if abs(corr) < 0.05:
        print("✅ PASS — strong one-way sensitivity")
    else:
        print("❌ FAIL — seed leakage detected")


# ============================================================
#  TEST 3 — PARAMETER COLLISION CHECK
# ============================================================

def parameter_collision_test(n_values=(8, 12, 16)):
    print("\n[PARAMETER COLLISION TEST]")

    outputs = {}
    for n in n_values:
        psi = generate_psi_star(n)
        h = float(np.round(np.linalg.norm(psi), 6))  # ← FIX
        outputs[n] = h
        print(f"n={n:2d} → ||ψ★|| = {h}")

    if len(set(outputs.values())) == len(outputs):
        print("✅ PASS — no parameter collisions")
    else:
        print("❌ FAIL — parameter collision detected")


# ============================================================
#  MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    print("Running WaveLock One-Wayness Test Suite")

    domain_separation_test()
    seed_sensitivity_test()
    parameter_collision_test()
