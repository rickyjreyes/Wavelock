import numpy as np
import cupy as cp
import hashlib
import sys
import os
from scipy.linalg import svd, eigh

# ================================================================
# IMPORTS
# ================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    laplacian,
    _serialize_commitment_v3,
)

# ================================================================
# CANONICAL V3 COMMITMENT
# ================================================================

def commit_v3(psi):
    """
    V3 high-precision commitment:
    - float64
    - big-endian
    - canonical serialization
    """
    psi = cp.asarray(psi, dtype=cp.float64)
    raw = _serialize_commitment_v3(psi)
    return hashlib.sha256(raw).hexdigest()


# ================================================================
# 1) LOW-RANK APPROXIMATION ATTACK (SVD)
# ================================================================

def svd_lowrank_attack_v3(n=6, max_rank=20):
    """
    Test whether rank-k SVD approximations collide under v3 commit.
    EXPECTED FOR V3: No collision until full rank.
    """
    kp = CurvatureKeyPair(n=n, use_v3=True)
    psi = cp.asnumpy(kp.psi_star)
    h0 = commit_v3(psi)

    U, S, Vt = svd(psi, full_matrices=False)

    results = []

    for k in range(1, min(max_rank, len(S))):
        recon = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
        h_k = commit_v3(recon)
        match = (h_k == h0)

        results.append({
            "rank": k,
            "matched": match,
            "hash": h_k[:16],
        })

        if match:
            return {"broken": True, "results": results}

    return {"broken": False, "results": results}


# ================================================================
# 2) LAPLACIAN EIGENDECOMP ATTACK
# ================================================================

def laplacian_eig_attack_v3(n=6, max_eigs=20):
    """
    Test whether ψ lies in low-dimensional Δ-eigenspace under v3.
    EXPECTED FOR V3: No collisions except full eigenbasis.
    """
    kp = CurvatureKeyPair(n=n, use_v3=True)
    psi = cp.asnumpy(kp.psi_star)
    h0 = commit_v3(psi)

    side = psi.shape[0]
    size = side * side

    # Build 2D Laplacian matrix
    L = np.zeros((size, size))
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            L[idx, idx] = -4
            L[idx, ((i+1) % side)*side + j] = 1
            L[idx, ((i-1) % side)*side + j] = 1
            L[idx, i*side + ((j+1) % side)] = 1
            L[idx, i*side + ((j-1) % side)] = 1

    # Compute eigenvectors
    vals, vecs = eigh(L)
    idx = np.argsort(np.abs(vals))[::-1]
    vecs = vecs[:, idx]

    results = []

    flat = psi.flatten()

    for k in range(1, min(max_eigs, size)):
        V = vecs[:, :k]
        coeffs = V.T @ flat
        recon_flat = V @ coeffs
        recon = recon_flat.reshape((side, side))

        h_k = commit_v3(recon)
        match = (h_k == h0)

        results.append({
            "n_eigs": k,
            "matched": match,
            "hash": h_k[:16]
        })

        if match:
            return {"broken": True, "results": results}

    return {"broken": False, "results": results}


# ================================================================
# 3) PRECISION CASCADE ATTACK
# ================================================================

def precision_attack_v3(n=6):
    """
    Tests float64→float32→float16 collisions under v3.
    EXPECTED FOR V3: No match except float64.
    """
    kp = CurvatureKeyPair(n=n, use_v3=True)
    psi = cp.asnumpy(kp.psi_star)

    h64 = commit_v3(psi)

    results = []

    # Float64
    results.append({
        "precision": "float64",
        "matched": True,
        "hash": h64[:16],
    })

    # Float32
    psi32 = psi.astype(np.float32).astype(np.float64)
    h32 = commit_v3(psi32)
    results.append({
        "precision": "float32",
        "matched": h32 == h64,
        "hash": h32[:16],
    })

    # Float16
    psi16 = psi.astype(np.float16).astype(np.float64)
    h16 = commit_v3(psi16)
    results.append({
        "precision": "float16",
        "matched": h16 == h64,
        "hash": h16[:16],
    })

    # Quantization bins
    for bits in [8, 16, 24, 32]:
        minv, maxv = psi.min(), psi.max()
        q = np.round((psi - minv) / (maxv - minv) * (2**bits - 1))
        dq = (q / (2**bits - 1)) * (maxv - minv) + minv
        hq = commit_v3(dq)

        results.append({
            "quant_bits": bits,
            "matched": hq == h64,
            "hash": hq[:16],
        })

    return {"results": results}


# ================================================================
# 4) SYMMETRY ATTACK
# ================================================================

def symmetry_attack_v3(n=6):
    """
    Test invariance under rotations & flips.
    EXPECTED FOR V3: No symmetries preserved.
    """
    kp = CurvatureKeyPair(n=n, use_v3=True)
    psi = cp.asnumpy(kp.psi_star)
    h0 = commit_v3(psi)

    results = []

    for k in range(1, 4):  # 90, 180, 270 rotations
        r = np.rot90(psi, k=k)
        h = commit_v3(r)
        match = (h == h0)
        results.append({"symmetry": f"rot90_{k}", "matched": match})
        if match:
            return {"broken": True, "results": results}

    for sym, op in [
        ("fliplr", np.fliplr),
        ("flipud", np.flipud),
        ("transpose", lambda x: x.T),
    ]:
        r = op(psi)
        h = commit_v3(r)
        match = (h == h0)
        results.append({"symmetry": sym, "matched": match})
        if match:
            return {"broken": True, "results": results}

    return {"broken": False, "results": results}






# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    print("\n=== WAVELOCK V3 ATTACK SUITE ===\n")

    print("1) SVD Low-Rank Attack (v3):")
    print(svd_lowrank_attack_v3())

    print("\n2) Laplacian Eigen Attack (v3):")
    print(laplacian_eig_attack_v3())

    print("\n3) Precision Cascade Attack (v3):")
    print(precision_attack_v3())

    print("\n4) Symmetry Attack (v3):")
    print(symmetry_attack_v3())

    print("\n=== DONE ===\n")
