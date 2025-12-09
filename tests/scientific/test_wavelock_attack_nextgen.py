# wavelock_attack_nextgen.py
# WaveLock Phase-2 Adversarial Attack Suite (WLv2-final, curvature_functional-free)

import numpy as np
import cupy as cp
import hashlib, math, sys, os

# =====================================================================
# PATH FIX
# =====================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# =====================================================================
# IMPORT WAVELOCK CORE
# =====================================================================

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    symbolic_verifier,
    laplacian,
    _serialize_commitment_v2,
)

# =====================================================================
# HELPERS
# =====================================================================

def commit(psi):
    """Return WaveLock commitment for any ψ-field."""
    psi = cp.asarray(psi, dtype=cp.float32)
    raw = _serialize_commitment_v2(psi)
    return hashlib.sha256(raw).hexdigest()

def same(a, b):
    return commit(a) == commit(b)

# =====================================================================
# 1. GRADIENT SURROGATE ATTACK v2 (curvature-free)
# =====================================================================

def gradient_attack_v2(n=6, iterations=3000, lr=4e-5):
    """
    White-box gradient surrogate attack.
    Uses Laplacian (curvature proxy) and bi-Laplacian as surrogate gradient.
    """
    kp = CurvatureKeyPair(n=n)
    target = kp.psi_star
    target_hash = commit(target)

    psi = cp.random.normal(0, 1, target.shape).astype(cp.float32)

    for i in range(iterations):
        # Gradient surrogate:
        grad = laplacian(psi) + 0.001 * laplacian(laplacian(psi))
        psi -= lr * grad

        if commit(psi) == target_hash:
            return {"matched": True, "iteration": i}

    return {"matched": False}

# =====================================================================
# 2. MONTE CARLO ANNEALING ATTACK (energy proxy)
# =====================================================================

def energy_proxy(psi):
    """
    Simple surrogate energy:
    ||Δψ||^2 + ||ψ||^2
    (No curvature_functional required)
    """
    L = laplacian(psi)
    return float(cp.sum(L*L) + cp.sum(psi*psi))

def monte_carlo_attack(n=6, steps=5000, T_start=1.0, T_end=0.01):
    kp = CurvatureKeyPair(n=n)
    target = kp.psi_star
    target_hash = commit(target)

    psi = cp.random.rand(*target.shape).astype(cp.float32)

    for t in range(steps):
        T = T_start * (1 - t / steps) + T_end * (t / steps)

        proposal = psi + cp.random.normal(0, T, psi.shape).astype(cp.float32)

        E_old = energy_proxy(psi)
        E_new = energy_proxy(proposal)

        if cp.random.rand() < cp.exp(-(E_new - E_old) / (T + 1e-9)):
            psi = proposal

        if commit(psi) == target_hash:
            return {"matched": True, "step": t}

    return {"matched": False}

# =====================================================================
# 3. FOURIER-ψ ADVERSARIAL SHELL ATTACK (fixed shell logic)
# =====================================================================

def fourier_shell_attack(n=6, depth=40):
    kp = CurvatureKeyPair(n=n)
    target = kp.psi_star
    target_hash = commit(target)

    side = target.shape[0]
    fx = cp.fft.fftfreq(side)
    fy = cp.fft.fftfreq(side)
    X, Y = cp.meshgrid(fx, fy)

    for d in range(1, depth+1):
        radius = 0.02 * d
        shell = cp.exp(-((X**2 + Y**2 - radius)**2) * 5000)

        wave = cp.real(cp.fft.ifft2(shell))
        wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-12)

        if commit(wave) == target_hash:
            return {"matched": True, "depth": d}

    return {"matched": False}

# =====================================================================
# 4. ZETA-PHASE LAYERED ATTACK (corrected)
# =====================================================================

def zeta_phase_attack(n=6, layers=50):
    kp = CurvatureKeyPair(n=n)
    target = kp.psi_star
    target_hash = commit(target)

    side = target.shape[0]
    psi = cp.zeros_like(target)

    x = cp.linspace(0, 1, side)

    for k in range(1, layers+1):
        freq = k * 11
        band = cp.sin(2 * cp.pi * freq * x)
        ring = cp.outer(band, band)

        psi += ring
        psi_norm = (psi - psi.min()) / (psi.max() - psi.min() + 1e-12)

        if commit(psi_norm) == target_hash:
            return {"matched": True, "layer": k}

    return {"matched": False}

# =====================================================================
# 5. CURVEHASH v3 MULTI-ROUND ATTACK
# =====================================================================

def curvehash_v3_attack(n=6, rounds=12):
    kp = CurvatureKeyPair(n=n)
    target = kp.psi_star
    target_hash = commit(target)

    psi = cp.random.rand(*target.shape).astype(cp.float32)

    for r in range(rounds):
        salt = cp.float32(0.03 * (r + 1))

        psi = psi + salt * laplacian(psi)
        psi = cp.tanh(psi)

        if commit(psi) == target_hash:
            return {"matched": True, "round": r}

    return {"matched": False}




# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("\n=== WAVELOCK PHASE 2 ADVERSARIAL ATTACK SUITE (v2-final) ===\n")

    print("1) Gradient Surrogate v2:")
    print(gradient_attack_v2())

    print("\n2) Monte Carlo Annealing Attack:")
    print(monte_carlo_attack())

    print("\n3) Fourier Shell Attack:")
    print(fourier_shell_attack())

    print("\n4) Zeta-Phase Layered Attack:")
    print(zeta_phase_attack())

    print("\n5) Curve-Hash v3 Multi-Round Attack:")
    print(curvehash_v3_attack())

    print("\n=== DONE ===\n")
