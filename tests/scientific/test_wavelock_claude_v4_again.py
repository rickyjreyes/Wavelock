import numpy as np
import cupy as cp
import hashlib
import sys
import os
from scipy.linalg import svd

# =====================================================================
# IMPORTS — REAL WAVELOCK
# =====================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    _serialize_commitment_v4,   # <-- NEW
)

# =====================================================================
# COMMITMENT HELPER — v4 curvature-canonical commitment
# =====================================================================

def commit(psi):
    """Return WaveLock v4 curvature-canonical commitment."""
    psi_cp = cp.asarray(psi, dtype=cp.float64)
    raw = _serialize_commitment_v4(psi_cp)
    return hashlib.sha256(raw).hexdigest()

# =====================================================================
# PDE WRAPPER — USE REAL WAVELOCK EVOLVE
# =====================================================================

def evolve_real(kp, psi0, T=20):
    """Run real WaveLock evolution (v3/v4 operator)."""
    psi0_cp = cp.asarray(psi0, dtype=cp.float64)
    out = kp.evolve(psi0_cp, T)
    return cp.asnumpy(out)

# =====================================================================
# 1. ROTATIONAL DIFFERENTIAL ATTACK (v4 safe)
# =====================================================================

def rotational_attack(n=6):
    kp = CurvatureKeyPair(n=n)
    psi = cp.asnumpy(kp.psi_star)
    h0 = commit(psi)

    results = []
    for angle in range(1, 4):  # 90,180,270
        rotated = np.rot90(psi, k=angle)
        h_rot = commit(rotated)
        matched = (h_rot == h0)
        results.append({"angle_90k": angle, "matched": matched, "hash": h_rot[:16]})
        if matched:
            return {"broken": True, "results": results}

    return {"broken": False, "results": results}

# =====================================================================
# 2. SYMMETRY ATTACK (v4 safe)
# =====================================================================

def symmetry_attack(n=6):
    kp = CurvatureKeyPair(n=n)
    psi = cp.asnumpy(kp.psi_star)
    h0 = commit(psi)

    results = []

    # rotations
    for k in range(1, 4):
        h = commit(np.rot90(psi, k=k))
        match = (h == h0)
        results.append({"symmetry": f"rot{k}", "matched": match})
        if match:
            return {"broken": True, "results": results}

    # flips
    for name, op in [
        ("fliplr", np.fliplr),
        ("flipud", np.flipud),
        ("transpose", lambda x: x.T),
    ]:
        h = commit(op(psi))
        match = (h == h0)
        results.append({"symmetry": name, "matched": match})
        if match:
            return {"broken": True, "results": results}

    return {"broken": False, "results": results}

# =====================================================================
# 3. LOW-RANK SVD ATTACK (v4 safe)
# =====================================================================

def lowrank_attack(n=6, max_rank=10):
    kp = CurvatureKeyPair(n=n)
    psi = cp.asnumpy(kp.psi_star)
    h0 = commit(psi)

    U, S, Vt = svd(psi, full_matrices=False)

    results = []
    for r in range(1, min(max_rank, len(S)) + 1):
        recon = U[:, :r] @ np.diag(S[:r]) @ Vt[:r, :]
        h = commit(recon)
        match = (h == h0)
        results.append({"rank": r, "matched": match, "hash": h[:16]})
        if match:
            return {"broken": True, "results": results}

    return {"broken": False, "results": results}


# =====================================================================
# 4. DIFFERENTIAL CRYPTANALYSIS (v4 fixed)
# =====================================================================

def differential_attack(n=6, trials=500, delta=1e-6, T=20):
    """
    Tests avalanche behavior under real WaveLock PDE.
    v4 commitment is highly sensitive — collisions should be zero.
    """
    kp = CurvatureKeyPair(n=n)
    collisions = 0

    for trial in range(trials):
        # random ψ0
        psi0 = np.random.randn(n, n)

        # small perturbation
        psi1 = psi0.copy()
        i, j = np.random.randint(0, n, 2)
        psi1[i, j] += delta

        # real PDE evolution
        out0 = evolve_real(kp, psi0, T)
        out1 = evolve_real(kp, psi1, T)

        h0 = commit(out0)
        h1 = commit(out1)

        if h0 == h1:
            collisions += 1

        if trial % 100 == 0:
            print(f"  [Differential] trial {trial}/{trials}")

    return {
        "collisions": collisions,
        "rate": collisions / trials,
        "matched": collisions > 0
    }






# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("\n=== WAVELOCK NASTIEST ATTACK SUITE (v4-upgraded) ===\n")

    print("1) Rotational Differential Attack:")
    print(rotational_attack())

    print("\n2) Symmetry Attack:")
    print(symmetry_attack())

    print("\n3) Low-Rank SVD Attack:")
    print(lowrank_attack())

    print("\n4) Differential Cryptanalysis Attack:")
    print(differential_attack())

    print("\n=== DONE ===\n")
