import numpy as np
import cupy as cp
import hashlib
import sys, os, time

# ---------------------------------------------------------------
# PATH FIX
# ---------------------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Load WaveLock v4 core
from wavelock.chain.WaveLock import (
    _serialize_commitment_v4,
    _curvature_functional,
    laplacian
)

# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def commit_v4(psi):
    raw = _serialize_commitment_v4(cp.asarray(psi, dtype=cp.float64))
    return hashlib.sha256(raw).hexdigest()

def curvature_vec(psi):
    E_grad, E_fb, E_ent, E_tot = _curvature_functional(cp.asarray(psi, dtype=cp.float64))
    return np.array([E_grad, E_fb, E_ent, E_tot], dtype=float)

def loss(a, b):
    return np.linalg.norm(a - b)

# ---------------------------------------------------------------
# 1) MULTISCALE RANDOM + GRAD DESCENT
# ---------------------------------------------------------------

def heavy_attack(psi_target, rounds=50, iters=1500, lr=1e-4, noise_scale=1.0):
    base_vec = curvature_vec(psi_target)
    base_hash = commit_v4(psi_target)

    side = psi_target.shape[0]
    shape = psi_target.shape

    best_dist = float("inf")
    best_match = None

    attempts = []

    for R in range(rounds):
        print(f"\n=== ROUND {R+1}/{rounds} ===")

        # Start with random ψ
        psi = cp.random.randn(*shape) * noise_scale

        for i in range(iters):
            # curvature & hash
            vec = curvature_vec(psi)
            h = commit_v4(psi)

            d = loss(vec, base_vec)
            if d < best_dist:
                best_dist = d

            # Check hash collision
            if h == base_hash:
                print(f"\n>>> COLLISION FOUND at round {R}, iter {i}!")
                return {
                    "collision": True,
                    "round": R,
                    "iter": i,
                    "vector": vec,
                    "hash": h[:16],
                }

            # Jacobian-free gradient step using Laplacian-like descent
            # This is an adversarial attempt to reshape geometry
            psi = psi - lr * laplacian(psi)

            # periodic "kick" to explore manifold
            if i % 200 == 0 and i > 0:
                psi += cp.random.randn(*shape) * (noise_scale / (1 + i/500))

        attempts.append({"round": R, "best_dist": best_dist})

    return {
        "collision": False,
        "best_dist": best_dist,
        "attempts": attempts,
        "target_hash": base_hash[:16],
        "target_vector": base_vec.tolist()
    }

# ---------------------------------------------------------------
# 2) PDE-STRUCTURED FIELD SEARCH
# ---------------------------------------------------------------

def pde_structured_search(psi_target, modes=20):
    base_vec = curvature_vec(psi_target)
    base_hash = commit_v4(psi_target)

    side = psi_target.shape[0]
    x = cp.linspace(0, 2*np.pi, side)

    best_match = float("inf")

    for a in range(1, modes):
        for b in range(1, modes):
            for c in range(1, 5):
                psi2 = (cp.sin(a*x).reshape(side,1) +
                        cp.cos(b*x).reshape(1,side) +
                        cp.sin(c*(x+x.reshape(-1,1))))

                vec2 = curvature_vec(psi2)
                h2 = commit_v4(psi2)

                d = loss(vec2, base_vec)
                if d < best_match:
                    best_match = d

                if h2 == base_hash:
                    return {
                        "collision": True,
                        "mode": (a,b,c),
                        "hash": h2[:16],
                        "vector": vec2.tolist()
                    }

    return {
        "collision": False,
        "best_dist": best_match,
        "target_hash": base_hash[:16]
    }


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

if __name__ == "__main__":
    print("\n=== WaveLock v4 Heavy Infinite-Collapse Search ===\n")

    side = 64
    psi = cp.random.rand(side, side)

    print("Running MULTISCALE ADVERSARIAL SEARCH...")
    res1 = heavy_attack(psi, rounds=20, iters=1000, lr=5e-5)
    print("\nResult:", res1)

    print("\nRunning PDE-STRUCTURED SEARCH...")
    res2 = pde_structured_search(psi, modes=15)
    print("\nResult:", res2)

    print("\n=== DONE ===\n")
