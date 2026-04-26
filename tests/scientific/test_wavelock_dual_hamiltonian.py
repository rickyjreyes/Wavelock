import os
import sys
import cupy as cp
from tqdm import tqdm

# ------------------------------------------------------------
# Locate repo root
# ------------------------------------------------------------

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from wavelock.chain.WaveLock import CurvatureKeyPair


# ------------------------------------------------------------
# Real WaveLock evolution wrapper
# ------------------------------------------------------------

def evolve_field(kp, psi, steps):
    psi = cp.asarray(psi, dtype=cp.float64)

    if hasattr(kp, "evolve"):
        return kp.evolve(psi, steps)

    if hasattr(kp, "_evolve"):
        return kp._evolve(psi)

    raise RuntimeError("WaveLock has no evolve() or _evolve()")


# ------------------------------------------------------------
# Dual Hamiltonian proxy operators
# ------------------------------------------------------------

def H1(f):
    return f - 0.1 * (cp.roll(f, -1, 0) - cp.roll(f, 1, 0))


def H2(f):
    return f - 0.1 * (cp.roll(f, -1, 1) - cp.roll(f, 1, 1))


# ------------------------------------------------------------
# Dual-Hamiltonian pseudo-inverse attack
# ------------------------------------------------------------

def dual_hamiltonian_attack(seed=123, n=6, steps=500, lr=0.03, T_pde=20):
    print(
        f"\n=== DUAL-HAMILTONIAN PSEUDO-INVERSE ATTACK "
        f"(n={n}, steps={steps}, T={T_pde}) ==="
    )

    # Real initialized WaveLock keypair
    kp = CurvatureKeyPair(n=n, seed=seed, test_mode=True)

    # Ground-truth forward map ψ0 → ψT
    cp.random.seed(seed)
    psi0_true = cp.random.standard_normal((n, n), dtype=cp.float64)
    psiT_target = evolve_field(kp, psi0_true, T_pde)

    # Attacker initialization
    psi_guess = cp.zeros_like(psi0_true)
    m = cp.array(0.0, dtype=cp.float64)

    eps = 1e-3

    for it in tqdm(range(steps)):
        psiT_guess = evolve_field(kp, psi_guess, T_pde)

        loss = cp.mean(
            (H1(psiT_guess) - H1(psiT_target)) ** 2
            + (H2(psiT_guess) - H2(psiT_target)) ** 2
        )

        # Scalar finite-difference gradient along global ψ direction
        psi_plus = psi_guess + eps
        psi_minus = psi_guess - eps

        psiT_plus = evolve_field(kp, psi_plus, T_pde)
        psiT_minus = evolve_field(kp, psi_minus, T_pde)

        Lp = cp.mean(
            (H1(psiT_plus) - H1(psiT_target)) ** 2
            + (H2(psiT_plus) - H2(psiT_target)) ** 2
        )

        Lm = cp.mean(
            (H1(psiT_minus) - H1(psiT_target)) ** 2
            + (H2(psiT_minus) - H2(psiT_target)) ** 2
        )

        grad = (Lp - Lm) / (2 * eps)

        # Broadcast scalar update over field
        m = 0.9 * m + 0.1 * grad
        psi_guess -= lr * m

        # Numerical safety
        psi_guess = cp.nan_to_num(psi_guess, nan=0.0, posinf=1e6, neginf=-1e6)
        psi_guess = cp.clip(psi_guess, -1e3, 1e3)

        if it % 100 == 0:
            print(f"[DualHam] iter={it:4d} loss={float(loss):.6f}")

    print("Final loss =", float(loss))
    return float(loss)


if __name__ == "__main__":
    dual_hamiltonian_attack()