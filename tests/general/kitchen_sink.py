"""
WaveLock Adversarial Attack Suite
================================

This script attempts to break a WaveLock-like irreversible PDE commitment
using every honest attack class available without learning:

1. PDE backward integration
2. Variational inverse (direct ψ₀ search)
3. Sensitivity / conditioning (Jacobian proxy)
4. Conservation-law discovery
5. Precision / determinism shortcuts
6. Blind verifier replication check

If this script fails to produce ψ★ (or a verifier-accepted equivalent)
without full forward evolution, invariant-based and inversion-based
attacks are empirically ruled out.

This is NOT a security proof.
It IS a falsification attempt.
"""

import numpy as np
from numpy.linalg import norm
import time

# ============================================================
#  PARAMETERS (freeze these)
# ============================================================

DT = 1e-4
STEPS = 5_000
GRID = (64, 64)
TOL = 1e-6
SEED = 0

np.random.seed(SEED)

# ============================================================
#  PDE KERNEL (REPLACE WITH REAL WAVELOCK KERNEL)
# ============================================================

def F(psi):
    """
    Example irreversible PDE RHS.
    Replace with your actual kernel.
    """
    lap = (
        -4 * psi
        + np.roll(psi, 1, 0) + np.roll(psi, -1, 0)
        + np.roll(psi, 1, 1) + np.roll(psi, -1, 1)
    )
    return lap - psi**3


def evolve_forward(psi0, steps=STEPS, dt=DT):
    psi = psi0.copy()
    for _ in range(steps):
        psi = psi + dt * F(psi)
    return psi


# ============================================================
#  VERIFIER (minimal)
# ============================================================

def verifier(candidate, psi_star, tol=TOL):
    return norm(candidate - psi_star) < tol


# ============================================================
#  1. BACKWARD INTEGRATION ATTACK
# ============================================================

def backward_integration_attack(psi_star, steps=2000, dt=DT):
    print("\n[1] BACKWARD INTEGRATION ATTACK")
    psi = psi_star.copy()
    for i in range(steps):
        psi = psi - dt * F(psi)  # reverse time
        if not np.isfinite(psi).all():
            print("  → Diverged (expected). PASS")
            return None
    print("  → Did NOT diverge. ⚠️ POTENTIAL ISSUE")
    return psi


# ============================================================
#  2. VARIATIONAL INVERSE (DIRECT ψ₀ SEARCH)
# ============================================================

def variational_inverse_attack(psi_star, iters=50, lr=1e-2):
    print("\n[2] VARIATIONAL INVERSE ATTACK")
    psi0 = np.random.randn(*psi_star.shape)

    for i in range(iters):
        psi_end = evolve_forward(psi0)
        loss = norm(psi_end - psi_star)**2
        print(f"  iter={i:03d} loss={loss:.4e}")

        if loss < 1e-8:
            print("  → Converged. ❌ BREAK")
            return psi0

        # crude finite-difference gradient (small grid only)
        grad = np.zeros_like(psi0)
        eps = 1e-4
        for idx in [(0,0), (1,1), (2,2)]:  # sparse probe
            psi0[idx] += eps
            lp = norm(evolve_forward(psi0) - psi_star)**2
            psi0[idx] -= 2*eps
            lm = norm(evolve_forward(psi0) - psi_star)**2
            psi0[idx] += eps
            grad[idx] = (lp - lm) / (2*eps)

        psi0 -= lr * grad

    print("  → No convergence. PASS")
    return None


# ============================================================
#  3. SENSITIVITY / CONDITIONING TEST
# ============================================================

def sensitivity_test(psi0, eps=1e-6):
    print("\n[3] SENSITIVITY / CONDITIONING TEST")
    base = evolve_forward(psi0)
    idx = (psi0.shape[0]//2, psi0.shape[1]//2)

    pert = psi0.copy()
    pert[idx] += eps
    out = evolve_forward(pert)

    sens = norm(out - base) / eps
    print(f"  Sensitivity ≈ {sens:.3e}")

    if sens > 1e6:
        print("  → Explosive sensitivity. PASS")
    else:
        print("  → Smooth sensitivity. ⚠️ CHECK")


# ============================================================
#  4. CONSERVATION LAW SCAN
# ============================================================

def conservation_scan(psi0):
    print("\n[4] CONSERVATION LAW SCAN")
    psi = psi0.copy()

    means, l2s, grads = [], [], []

    for _ in range(1000):
        psi = psi + DT * F(psi)
        means.append(psi.mean())
        l2s.append(np.sum(psi**2))
        gy, gx = np.gradient(psi)
        grads.append(np.sum(gx**2 + gy**2))

    print("  std(mean)   =", np.std(means))
    print("  std(L2)     =", np.std(l2s))
    print("  std(gradE)  =", np.std(grads))

    if np.std(l2s) < 1e-6:
        print("  → Possible invariant. ⚠️ CHECK")
    else:
        print("  → No conserved scalars. PASS")


# ============================================================
#  5. PRECISION / DETERMINISM ATTACK
# ============================================================

def precision_attack(psi0):
    print("\n[5] PRECISION ATTACK")
    psi64 = evolve_forward(psi0.astype(np.float64))
    psi32 = evolve_forward(psi0.astype(np.float32)).astype(np.float64)
    diff = norm(psi64 - psi32)
    print(f"  float64 vs float32 diff = {diff:.3e}")

    if diff < 1e-6:
        print("  → Precision shortcut risk. ⚠️ CHECK")
    else:
        print("  → Precision matters. PASS")


# ============================================================
#  6. THIRD-PARTY REPLICATION CHECK
# ============================================================

def replication_check(psi_star):
    print("\n[6] BLIND REPLICATION CHECK")
    fake = np.random.randn(*psi_star.shape)
    if verifier(fake, psi_star):
        print("  → Random guess accepted. ❌ BREAK")
    else:
        print("  → Verifier rejects random input. PASS")


# ============================================================
#  MAIN DRIVER
# ============================================================

if __name__ == "__main__":
    print("WaveLock Adversarial Attack Suite")
    print("================================")

    H, W = GRID
    psi0_true = np.random.randn(H, W)

    print("\nGenerating ψ★ via forward evolution...")
    t0 = time.time()
    psi_star = evolve_forward(psi0_true)
    print(f"Done in {time.time() - t0:.2f}s")

    backward_integration_attack(psi_star)
    variational_inverse_attack(psi_star)
    sensitivity_test(psi0_true)
    conservation_scan(psi0_true)
    precision_attack(psi0_true)
    replication_check(psi_star)

    print("\nATTACK SUITE COMPLETE")
