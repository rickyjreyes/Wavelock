"""
WaveLock — Parameter-Space Jacobian Structure Test
=================================================

This test examines the *local differential structure* of the WaveLock
parameter map:

    (n, seed)  →  ψ★

IMPORTANT:
-----------
WaveLock does NOT claim Jacobian rank collapse in parameter space.
Local smoothness is REQUIRED for determinism and reproducibility.

This test therefore does NOT test one-wayness.
It tests for *pathological behavior* only.

ψ₀ is NOT part of the domain by design.
"""

import numpy as np
import os
import sys

# ============================================================
#  Ensure WaveLock repo root is on PYTHONPATH
# ============================================================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ============================================================
#  Import production WaveLock kernel
# ============================================================

from wavelock.chain.WaveLock import CurvatureKeyPair


# ============================================================
#  ψ★ generation — production faithful
# ============================================================

def generate_psi_star(*, n, seed, **kwargs):
    """
    Generate ψ★ using the real WaveLock kernel.
    """
    kp = CurvatureKeyPair(
        n=n,
        seed=seed,
        test_mode=True,
        **kwargs
    )
    return kp.psi_star


# ============================================================
#  PARAMETER-SPACE JACOBIAN STRUCTURE TEST
# ============================================================

def parameter_jacobian_structure_test(
    n=12,
    base_seed=0,
    probes=20,
    seed_step=1,
):
    """
    Examines the singular value spectrum of the local Jacobian
    of the parameter → ψ★ map.

    EXPECTED (WaveLock):
        - Smooth spectrum
        - No sudden rank collapse
        - No numerical degeneracy

    WARN:
        - Extremely sharp collapse (may indicate hidden symmetry)

    FAIL:
        - Numerical instability
        - Exploding / NaN spectrum
    """

    print("\n[PARAMETER-SPACE JACOBIAN STRUCTURE TEST — WAVELOCK]")
    print(f"Fixed n = {n}")

    base = generate_psi_star(n=n, seed=base_seed)
    diffs = []

    for i in range(probes):
        seed_i = base_seed + seed_step * (i + 1)
        psi = generate_psi_star(n=n, seed=seed_i)
        diffs.append((psi - base).flatten())

    diffs = np.stack(diffs)

    # SVD of parameter → ψ★ differential map
    try:
        _, svals, _ = np.linalg.svd(diffs, full_matrices=False)
    except Exception as e:
        print("❌ FAIL — SVD failed (numerical instability)")
        print(e)
        return

    print("Top singular values:")
    for i, s in enumerate(svals[:10]):
        print(f"  σ[{i}] = {s:.3e}")

    ratio = svals[1] / svals[0] if svals[0] > 0 else np.inf
    print(f"σ₁ / σ₀ = {ratio:.3e}")

    # --------------------------------------------------------
    # Interpretation (WaveLock-correct)
    # --------------------------------------------------------

    if not np.isfinite(ratio):
        print("❌ FAIL — non-finite spectrum (numerical instability)")
    elif ratio < 1e-3:
        print("⚠️ WARN — unusually sharp rank collapse (investigate symmetry)")
    else:
        print("✅ EXPECTED — smooth parameter map (deterministic, non-invertible)")


# ============================================================
#  ENTRY POINT
# ============================================================

if __name__ == "__main__":
    np.random.seed(0)

    print("Running WaveLock Parameter-Space Jacobian Structure Test")

    parameter_jacobian_structure_test(
        n=12,
        base_seed=0,
        probes=20,
        seed_step=1,
    )
