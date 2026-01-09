#!/usr/bin/env python3
import os
import sys
import cupy as cp
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from wavelock.chain.WaveLock import CurvatureKeyPair


# ============================================================
# Professional-Grade PDE Forward Factory
# ============================================================

def pde_forward_factory(kp):
    def forward(psi):
        return kp.evolve(cp.asarray(psi), kp.n)
    return forward


# ============================================================
# Multi-Norm, Adaptive TBJA Optimizer (Professional)
# ============================================================

def tbja_pro_attempt(kp, pde_forward,
                     steps=15000,
                     restarts=8,
                     base_lr=0.01,
                     momentum=0.9,
                     anneal=0.999,
                     noise_scale=0.005):

    target = kp.psi_star
    best_loss = float("inf")

    print(f"[INFO] TBJA-Pro attack: restarts={restarts}, steps={steps}")

    for r in range(restarts):

        print(f"\n=== Restart {r+1}/{restarts} ===")

        # Random initialization
        guess = cp.random.normal(0.01, 0.02, size=target.shape)
        velocity = cp.zeros_like(guess)
        lr = base_lr

        for i in range(steps):

            evolved = pde_forward(guess)
            diff = evolved - target
            loss = cp.linalg.norm(diff)

            if i % 500 == 0:
                print(f"[TBJA-PRO] restart={r+1}, step={i}, loss={float(loss):.6f}")

            # Mixed-norm gradient estimate
            grad_l1 = cp.sign(diff)
            grad_l2 = diff / (cp.linalg.norm(diff) + 1e-12)
            grad_inf = cp.clip(diff, -1, 1)

            grad = 0.4 * grad_l1 + 0.4 * grad_l2 + 0.2 * grad_inf

            # Momentum + adaptive LR
            velocity = momentum * velocity - lr * grad
            guess = guess + velocity

            # Inject tiny noise for basin escape
            guess += noise_scale * cp.random.normal(0, 1, size=guess.shape)

            # Anneal learning rate
            lr *= anneal

        best_loss = min(best_loss, float(loss))

    return best_loss


# ============================================================
# TEST RUNNER
# ============================================================

def run_tbja_scaled_pro():
    print("\n====================================================")
    print("    WAVELOCK — TBJA-PRO ADVERSARIAL INVERSION TEST  ")
    print("====================================================\n")

    # Use realistic dimensionality for a serious test
    kp = CurvatureKeyPair(n=8, seed=42)
    print(f"[INFO] ψ★ shape = {kp.psi_star.shape}")
    print(f"[INFO] Commitment = {kp.commitment}\n")

    pde_forward = pde_forward_factory(kp)

    # Run the professional adversarial inversion suite
    best_loss = tbja_pro_attempt(
        kp,
        pde_forward,
        steps=15000,
        restarts=10,
        base_lr=0.02,
        momentum=0.9,
        anneal=0.9995,
        noise_scale=0.003
    )

    print("\n===================== RESULTS =====================")
    print("Best adversarial loss:", best_loss)

    if best_loss > 10:
        print("[PASS] PDE remains one-way under TBJA-Pro attack.")
    else:
        print("[FAIL] Adversarial inversion reduced loss → investigate!")


if __name__ == "__main__":
    run_tbja_scaled_pro()
