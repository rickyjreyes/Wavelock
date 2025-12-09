import numpy as np
import cupy as cp
import hashlib
import sys, os, time

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    laplacian,
    symbolic_verifier,
    _serialize_commitment_v2,
)

# =====================================================
# HELPERS
# =====================================================

def commit(psi):
    """Compute WLv3 SHA256 commitment."""
    return hashlib.sha256(_serialize_commitment_v2(cp.asarray(psi))).hexdigest()

# =====================================================
# 1. QUANTUM PHASE ESTIMATION (QPE-SIM)
# =====================================================
# Quantum-inspired spectral-phase estimator.
# No curvature functional used.

def qpe_sim_attack(n=6, steps=50):
    kp = CurvatureKeyPair(n=n)
    psi_star = kp.psi_star
    h0 = commit(psi_star)

    for k in range(1, steps + 1):
        # Extract phase of Fourier transform under pseudo-eigen kick
        phase = cp.angle(cp.fft.fft2(psi_star * cp.exp(1j * k)))
        recon = cp.real(cp.fft.ifft2(cp.exp(1j * phase)))

        if commit(recon) == h0:
            return {"matched": True, "step": k}

    return {"matched": False}

# =====================================================
# 2. HHL-LIKE LINEAR SYSTEM ATTACK
# =====================================================
# Approximates ψ* by solving A ψ = b for random A.
# No curvature functional used.

def hhl_attack(n=6):
    kp = CurvatureKeyPair(n=n)
    psi_star = kp.psi_star
    h0 = commit(psi_star)

    side = psi_star.shape[0]

    # Attacker incorrectly assumes a linear PDE
    A = cp.eye(side) + 0.01 * cp.random.rand(side, side)
    b = cp.mean(psi_star, axis=0)

    try:
        x = cp.linalg.solve(A, b)
        approx = cp.outer(x, x)
    except Exception:
        return {"matched": False}

    return {"matched": commit(approx) == h0}

# =====================================================
# 3. MULTISCALE WAVELET ATTACK (valid version)
# =====================================================
# Haar-like upsampling using kron.
# No curvature functional used.

def wavelet_attack(n=6):
    kp = CurvatureKeyPair(n=n)
    psi_star = kp.psi_star
    h0 = commit(psi_star)

    side = psi_star.shape[0]
    psi = cp.zeros_like(psi_star)

    # Valid scales only (must divide side)
    scales = [1, 2, 4, 8, 16, 32]
    scales = [s for s in scales if (side // s) > 0]

    for scale in scales:
        h = side // scale
        w = side // scale

        patch = cp.kron(cp.random.rand(h, w), cp.ones((scale, scale)))
        psi += 0.01 * patch

        if commit(psi) == h0:
            return {"matched": True, "scale": scale}

    return {"matched": False}

# =====================================================
# 4. MANIFOLD LEARNING ATTACK
# =====================================================
# Attacker tries low-rank PCA reconstruction.
# No curvature functional used.

def manifold_attack(n=6, dim=4):
    kp = CurvatureKeyPair(n=n)
    psi_star = kp.psi_star
    h0 = commit(psi_star)

    vec = psi_star.flatten()

    # Rank-1 SVD decomposition
    U, S, Vt = cp.linalg.svd(vec.reshape(1, -1), full_matrices=False)

    # attacker only keeps top dim components
    recon = (U[:, :dim] @ Vt[:dim, :]).reshape(psi_star.shape)

    return {"matched": commit(recon) == h0}

# =====================================================
# 5. ADJOINT PDE (reverse-time Laplacian) ATTACK
# =====================================================
# Simple surrogate for reverse-evolution.
# No curvature functional used.

def adjoint_attack(n=6, steps=200, lr=1e-4):
    kp = CurvatureKeyPair(n=n)
    psi_star = kp.psi_star
    h0 = commit(psi_star)

    psi = cp.random.normal(0, 1, psi_star.shape)

    for i in range(steps):
        adj = -(laplacian(psi) - laplacian(psi_star))
        psi = psi + lr * adj

        if commit(psi) == h0:
            return {"matched": True, "iteration": i}

    return {"matched": False}

# =====================================================
# 6. DUAL-SPACE ψ + ∇ψ CYCLIC ATTACK
# =====================================================
# Attacker jointly evolves ψ and its Laplacian.
# No curvature functional used.

def dual_space_attack(n=6, steps=800):
    kp = CurvatureKeyPair(n=n)
    psi_star = kp.psi_star
    h0 = commit(psi_star)

    psi = cp.random.rand(*psi_star.shape)
    grad = laplacian(psi)

    for i in range(steps):
        psi = psi - 0.001 * grad
        grad = laplacian(psi)

        mixed = 0.5 * psi + 0.5 * grad

        if commit(mixed) == h0:
            return {"matched": True, "iteration": i}

    return {"matched": False}









# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    print("\n=== WAVELOCK PHASE 4 — MAXIMUM ADVERSARIAL SUITE (WLv3) ===\n")

    print("1) QPE Simulation Attack:")
    print(qpe_sim_attack())

    print("\n2) HHL Linear-System Attack:")
    print(hhl_attack())

    print("\n3) Multiscale Wavelet Attack:")
    print(wavelet_attack())

    print("\n4) Manifold Learning Attack:")
    print(manifold_attack())

    print("\n5) Adjoint PDE Inversion:")
    print(adjoint_attack())

    print("\n6) Dual-Space ψ/∇ψ Attack:")
    print(dual_space_attack())

    print("\n=== DONE ===")
