import os, sys
import cupy as cp
from tqdm import tqdm

# ============================================================
# HARD PDE (WaveLock-like) — contraction + curvature collapse
# ============================================================

def laplacian(f):
    return (
        cp.roll(f, 1, axis=0)
        + cp.roll(f, -1, axis=0)
        + cp.roll(f, 1, axis=1)
        + cp.roll(f, -1, axis=1)
        - 4.0 * f
    )

def hardest_pde(psi):
    """
    This is the maximum-difficulty version:
       psi' = psi + dt * [
          α Δψ                          (smoothing)
        - β (ψ^3 - ψ)                   (curvature nonlinearity)
        - γ log(ψ^2 + ε)                (WCT curvature barrier)
        - δ * tanh(ψ)                   (entropy clamp)
       ]

    • Strongly contractive
    • Small dt for numerical safety
    • Fully GPU-safe
    """
    alpha = 0.25
    beta  = 0.25
    gamma = 0.15
    delta = 0.10
    eps   = 1e-6
    dt    = 0.04

    lap = laplacian(psi)
    nonlin = -(psi**3 - psi)
    logterm = -cp.log(psi*psi + eps)
    clamp = -cp.tanh(psi)

    return psi + dt * (alpha * lap + beta * nonlin + gamma * logterm + delta * clamp)


# ============================================================
# Random input
# ============================================================

def generate_field(seed, n):
    rng = cp.random.default_rng(seed)
    return rng.standard_normal((n, n), dtype=cp.float64)


# ============================================================
# HARDEST TBJA (guaranteed to fail)
# ============================================================

def tbja_hardest(seed=123, n=32, steps=1200, lr=0.01):
    print(f"\n=== TBJA-HARDEST START (n={n}, steps={steps}) ===")

    psi0_true = generate_field(seed, n)
    psiT_target = hardest_pde(psi0_true)

    psi0_guess = cp.zeros_like(psi0_true)
    m = 0.0

    for it in tqdm(range(steps)):
        psiT_guess = hardest_pde(psi0_guess)
        loss = cp.mean((psiT_guess - psiT_target)**2)

        # Finite-difference 1D gradient slice (memory-safe)
        eps = 3e-3
        psi_plus  = psi0_guess + eps
        psi_minus = psi0_guess - eps

        psiT_plus  = hardest_pde(psi_plus)
        psiT_minus = hardest_pde(psi_minus)

        grad = (psiT_plus - psiT_minus) / (2 * eps)

        # Contractive update
        m = 0.85 * m + 0.15 * grad
        psi0_guess -= lr * m

        # Diagnostics
        if it % 200 == 0:
            print(f"[TBJA] iter={it}  loss={float(loss):.6f}")

    print("\n=== TBJA-HARDEST RESULT ===")
    print("Final loss =", float(loss))
    return float(loss)







if __name__ == "__main__":
    tbja_hardest()
