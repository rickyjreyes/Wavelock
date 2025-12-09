# ============================================================
# WaveLock True-Backprop Attack (TBJA)
# ============================================================
# This is the strongest possible inversion attempt:
# backpropagate through *all PDE steps* using autograd + FP64,
# estimate ∇ψ0 ψ*, and run Adam to drive ψ0 → ψ*.
# ============================================================

import torch
import torch.nn.functional as F
import numpy as np
import hashlib
import cupy as cp

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    _serialize_commitment_v2
)

# ============================================================
# Utility: SHA-256 commitment
# ============================================================

def commit(psi_np):
    psi_cp = cp.asarray(psi_np)
    raw = _serialize_commitment_v2(psi_cp)
    return hashlib.sha256(raw).hexdigest()

# ============================================================
# PyTorch implementation of WaveLock PDE evolution
# (identical operator, but differentiable)
# ============================================================

def laplacian_torch(x):
    return (
        -4*x
        + torch.roll(x, 1, dims=0)
        + torch.roll(x, -1, dims=0)
        + torch.roll(x, 1, dims=1)
        + torch.roll(x, -1, dims=1)
    )

def evolve_torch(psi, T=50, dt=0.01, alpha=1.0, beta=1.0, gamma=0.3, eps=1e-12):
    """
    Differentiable surrogate of the real WaveLock PDE:
    ψ_{t+1} = ψ_t + dt * (αΔψ − β(ψ^3 − ψ) − γ log(ψ^2 + ε))
    """
    for _ in range(T):
        L = laplacian_torch(psi)
        F = alpha*L - beta*(psi**3 - psi) - gamma*torch.log(psi*psi + eps)
        psi = psi + dt * F
    return psi

# ============================================================
# TRUE BACKPROP ATTACK
# ============================================================

def true_backprop_attack(n=6, steps=3000, lr=5e-3):
    kp = CurvatureKeyPair(n=n)
    target = cp.asnumpy(kp.psi_star)
    target_hash = commit(target)

    side = target.shape[0]

    # Target field as torch tensor
    psi_target = torch.tensor(target, dtype=torch.float64)

    # Attacker guess for ψ0
    psi0 = torch.randn((side, side), dtype=torch.float64, requires_grad=True)

    opt = torch.optim.Adam([psi0], lr=lr)

    for step in range(steps):
        opt.zero_grad()

        psi_star_hat = evolve_torch(psi0)

        # L2 reconstruction loss
        loss = F.mse_loss(psi_star_hat, psi_target)

        loss.backward()
        opt.step()

        # Check exact hash match (extremely unlikely)
        if step % 50 == 0:
            cur = psi_star_hat.detach().numpy()
            if commit(cur) == target_hash:
                return {
                    "matched": True,
                    "iteration": step,
                    "loss": float(loss)
                }

    return {
        "matched": False,
        "final_loss": float(loss.detach().numpy())
    }






# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    print("\n=== TRUE BACKPROP JACOBIAN ATTACK (TBJA) ===\n")
    print(true_backprop_attack())
    print("\n=== DONE ===\n")
