import numpy as np
import cupy as cp
import hashlib, math, sys, os, time

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    laplacian,
    symbolic_verifier,
    _serialize_commitment_v2,
)

# ============================================================
# Helpers
# ============================================================

def commit(psi):
    """Compute SHA256 commitment of ψ-field."""
    return hashlib.sha256(_serialize_commitment_v2(cp.asarray(psi))).hexdigest()

# ============================================================
# 1. GROVER-SIM ATTACK  (Quantum amplitude amplification)
# ============================================================

def grover_sim_attack(n=6, iterations=2000):
    kp = CurvatureKeyPair(n=n)
    target = kp.psi_star
    h0 = commit(target)

    psi = cp.random.normal(0, 1, target.shape)

    for t in range(iterations):
        oracle = -cp.abs(laplacian(psi))

        psi = psi + 0.001 * oracle
        psi = psi / (cp.linalg.norm(psi) + 1e-12)

        if commit(psi) == h0:
            return {"matched": True, "iteration": t}

    return {"matched": False}

# ============================================================
# 2. QAOA-SIM ATTACK (Quantum alternating optimizer)
# ============================================================

def qaoa_sim_attack(n=6, layers=2000, step=1e-4):
    kp = CurvatureKeyPair(n=n)
    target = kp.psi_star
    h0 = commit(target)

    psi = cp.random.rand(*target.shape)

    for i in range(layers):
        psi = psi - step * laplacian(psi)
        psi = cp.sin(cp.pi * psi)

        if commit(psi) == h0:
            return {"matched": True, "layer": i}

    return {"matched": False}

# ============================================================
# 3. QFT-RECONSTRUCTION ATTACK (partial Fourier invert)
# ============================================================

def qft_reconstruction_attack(n=6):
    kp = CurvatureKeyPair(n=n)
    target = kp.psi_star
    h0 = commit(target)
    side = target.shape[0]

    F = cp.fft.fft2(target)

    ranks = [1, 2, 4, 8, 16, 32]

    for r in ranks:
        mask = cp.zeros_like(F)
        mask[:r, :r] = F[:r, :r]

        guess = cp.real(cp.fft.ifft2(mask))

        if commit(guess) == h0:
            return {"matched": True, "rank": r}

    return {"matched": False}

# ============================================================
# 4. QUANTUM GIBBS SAMPLING ATTACK
# ============================================================

def gibbs_qsim_attack(n=6, samples=5000, T=0.01):
    kp = CurvatureKeyPair(n=n)
    target = kp.psi_star
    h0 = commit(target)

    for _ in range(samples):
        psi = cp.random.normal(0, T, target.shape)

        weight = cp.exp(-cp.abs(laplacian(psi)) / (T + 1e-12))
        psi = psi * weight

        if commit(psi) == h0:
            return {"matched": True}

    return {"matched": False}

# ============================================================
# 5. QUANTUM RANDOM WALK COLLISION SEARCH
# ============================================================

def qrw_collision_attack(n=6, steps=10000):
    kp = CurvatureKeyPair(n=n)
    target = kp.psi_star
    h0 = commit(target)

    psi = cp.random.normal(0, 1, target.shape)

    for i in range(steps):
        psi = psi + cp.random.normal(0, 0.002, psi.shape)
        psi = cp.tanh(psi)

        if commit(psi) == h0:
            return {"matched": True, "step": i}

    return {"matched": False}












# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n=== WAVELOCK POST-QUANTUM ATTACK SUITE (WLv3) ===\n")

    print("1) Grover-Sim Attack:")
    print(grover_sim_attack())

    print("\n2) QAOA-Sim Attack:")
    print(qaoa_sim_attack())

    print("\n3) QFT-Based Reconstruction:")
    print(qft_reconstruction_attack())

    print("\n4) Quantum Gibbs Sampling Attack:")
    print(gibbs_qsim_attack())

    print("\n5) Quantum Random Walk Collision Search:")
    print(qrw_collision_attack())

    print("\n=== DONE ===")
