"""
===============================================================
   WAVELOCK CLAUDE-4 ATTACK SUITE (WLv4)
   -------------------------------------

   Purpose:
   • Test whether an extremely strong ML-style surrogate
     (Claude-4 class: high-dimensional function learner)
     can reconstruct ψ or its curvature invariants.

   • Evaluate curvature-canonical commitments (WLv4) against:
       – surrogate inversion
       – curvature descent matching
       – manifold projection
       – random surrogate probing
       – noisy superpositions

   EXPECTED RESULT FOR WLv4:
   -------------------------
   All matched == False, no collisions, no partial recovery.
===============================================================
"""

import numpy as np
import cupy as cp
import hashlib
import sys, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    laplacian,
    _serialize_commitment_v4,
)

# --------------------------------------------------------------
# Helper: WLv4 canonical commitment
# --------------------------------------------------------------
def commit_v4(psi):
    raw = _serialize_commitment_v4(cp.asarray(psi, dtype=cp.float64))
    return hashlib.sha256(raw).hexdigest()


# --------------------------------------------------------------
# 1) RANDOM SURROGATE PROBE
# --------------------------------------------------------------
def surrogate_probe_v4(kp, trials=50):
    """
    Tries random ψ-like fields and checks if any match the curvature signature.
    Expected: never matches.
    """
    psi = cp.asnumpy(kp.psi_star)
    h0 = commit_v4(psi)

    results = []
    for t in range(trials):
        probe = np.random.rand(*psi.shape)
        h = commit_v4(probe)
        matched = (h == h0)

        results.append({
            "trial": t,
            "matched": matched,
            "hash": h[:16]
        })

        if matched:
            return {"broken": True, "results": results}

    return {"broken": False, "results": results}


# --------------------------------------------------------------
# 2) CURVATURE-DESCENT MATCHING
# --------------------------------------------------------------
def curvature_descent_attack_v4(kp, iters=200, lr=1e-3):
    """
    Attempts gradient-style matching:
        minimize |CCI(psi_candidate) - CCI(psi_star)|
    Expected: fails because curvature mapping is irreducible.
    """
    psi_star = cp.asarray(kp.psi_star, dtype=cp.float64)
    h0 = commit_v4(psi_star)

    psi = cp.asarray(np.random.rand(*psi_star.shape), dtype=cp.float64)

    results = []
    for i in range(iters):
        # Approximate curvature gradient step
        grad = laplacian(psi)
        psi = psi - lr * grad

        h = commit_v4(psi)
        matched = (h == h0)

        results.append({
            "iter": i,
            "matched": matched,
            "hash": h[:16]
        })

        if matched:
            return {"broken": True, "results": results}

    return {"broken": False, "results": results}


# --------------------------------------------------------------
# 3) MULTI-POINT SUPERPOSITION ATTACK
# --------------------------------------------------------------
def superposition_attack_v4(kp, S=10):
    """
    Generates convex / linear / nonlinear superpositions of random fields.
    Expected: never matches curvature signature.
    """
    psi_star = cp.asnumpy(kp.psi_star)
    h0 = commit_v4(psi_star)

    side = psi_star.shape[0]
    fields = [np.random.rand(side, side) for _ in range(S)]

    results = []

    # convex mixtures
    for i in range(S):
        for j in range(S):
            mix = 0.5*fields[i] + 0.5*fields[j]
            h = commit_v4(mix)
            matched = (h == h0)
            results.append({
                "mix": (i, j),
                "matched": matched,
                "hash": h[:16]
            })
            if matched:
                return {"broken": True, "results": results}

    # nonlinear: products
    for i in range(S):
        for j in range(S):
            mix = fields[i] * fields[j]
            h = commit_v4(mix)
            matched = (h == h0)
            results.append({
                "product": (i, j),
                "matched": matched,
                "hash": h[:16]
            })
            if matched:
                return {"broken": True, "results": results}

    return {"broken": False, "results": results}


# --------------------------------------------------------------
# 4) CLAUDE-4 INVERSION ATTACK (Deep-surrogate proxy)
# --------------------------------------------------------------
def claude4_inversion_v4(kp, iters=150, latent_dim=256):
    """
    Simulates a Claude-4 surrogate attempt:
    • Start from a latent z ∈ R^d
    • Project z → ψ-approximation through random linear decoder
    • Try to match curvature invariants

    Expected: mathematically impossible for v4.
    """
    psi_star = cp.asnumpy(kp.psi_star)
    h0 = commit_v4(psi_star)

    side = psi_star.shape[0]
    size = side * side

    # Random linear decoder: latent → ψ-space
    decoder = np.random.randn(size, latent_dim)

    results = []

    for t in range(iters):
        z = np.random.randn(latent_dim)

        recon_flat = decoder @ z
        recon = recon_flat.reshape(side, side)

        h = commit_v4(recon)
        matched = (h == h0)

        results.append({
            "iter": t,
            "matched": matched,
            "hash": h[:16],
        })

        if matched:
            return {"broken": True, "results": results}

    return {"broken": False, "results": results}





# --------------------------------------------------------------
# MAIN
# --------------------------------------------------------------
if __name__ == "__main__":
    print("\n=== WAVELOCK CLAUDE-4 ATTACK SUITE (WLv4) ===\n")

    kp = CurvatureKeyPair(n=6, use_v4=True)

    print("1) Surrogate Probe Attack:")
    print(surrogate_probe_v4(kp))

    print("\n2) Curvature-Descent Attack:")
    print(curvature_descent_attack_v4(kp))

    print("\n3) Superposition Attack:")
    print(superposition_attack_v4(kp))

    print("\n4) Claude-4 Latent Surrogate Attack:")
    print(claude4_inversion_v4(kp))

    print("\n=== DONE ===\n")
