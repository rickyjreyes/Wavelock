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

from wavelock.chain.WaveLock import CurvatureKeyPair, _curvature_functional


# ------------------------------------------------------------
# PDE wrapper
# ------------------------------------------------------------

def evolve_field(kp, psi, steps=None):
    psi = cp.asarray(psi, dtype=cp.float64)

    if steps is None:
        steps = getattr(kp, "T", getattr(kp, "n", 20))

    if hasattr(kp, "evolve"):
        return kp.evolve(psi, steps)

    if hasattr(kp, "_evolve"):
        return kp._evolve(psi)

    raise RuntimeError("WaveLock has no evolve() or _evolve()")


# ------------------------------------------------------------
# CURVATURE ATTACK
# ------------------------------------------------------------

def curvature_attack(seed=123, n=6, steps=500, lr=0.004, lam=0.1, T_pde=20):
    print(f"=== WCT CURVATURE ATTACK (n={n}, steps={steps}, T={T_pde}) ===")

    # Real initialized WaveLock keypair
    kp = CurvatureKeyPair(n=n, seed=seed, test_mode=True)

    # Target field
    rng = cp.random.default_rng(seed)
    psi0_true = rng.standard_normal((n, n), dtype=cp.float64)
    psiT_target = evolve_field(kp, psi0_true, T_pde)

    # Guess
    psi0_guess = cp.zeros_like(psi0_true)
    m = cp.zeros_like(psi0_guess)

    eps = 1e-3

    for it in tqdm(range(steps)):
        psiT_guess = evolve_field(kp, psi0_guess, T_pde)
        loss_main = cp.mean((psiT_guess - psiT_target) ** 2)

        E_grad, E_fb, E_ent, E_tot = _curvature_functional(psi0_guess)
        loss = loss_main + lam * E_tot

        # finite-difference inverse gradient proxy
        psi_plus = psi0_guess + eps
        psi_minus = psi0_guess - eps

        T_plus = evolve_field(kp, psi_plus, T_pde)
        T_minus = evolve_field(kp, psi_minus, T_pde)

        g_inv = (T_plus - T_minus) / (2 * eps)

        E_plus = _curvature_functional(psi_plus)[3]
        E_minus = _curvature_functional(psi_minus)[3]
        g_curv_scalar = (E_plus - E_minus) / (2 * eps)

        grad = g_inv + lam * g_curv_scalar

        # update
        m = 0.9 * m + 0.1 * grad
        psi0_guess -= lr * m

        # numerical safety
        psi0_guess = cp.nan_to_num(psi0_guess, nan=0.0, posinf=1e6, neginf=-1e6)
        psi0_guess = cp.clip(psi0_guess, -1e3, 1e3)

        if it % 100 == 0:
            print(
                f"[CURV] iter={it} "
                f"loss={float(loss):.5f} "
                f"main={float(loss_main):.5f} "
                f"E={float(E_tot):.5f}"
            )

    print("\n=== RESULT ===")
    print("Final loss:", float(loss))
    return float(loss)


if __name__ == "__main__":
    curvature_attack()