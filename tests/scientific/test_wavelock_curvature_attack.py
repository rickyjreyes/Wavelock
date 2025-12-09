import os, sys
import cupy as cp
from tqdm import tqdm

# ------------------------------------------------------------
# Locate real WaveLock (not archive)
# ------------------------------------------------------------
def find_chain(start):
    for root, dirs, files in os.walk(start):
        if root.endswith(os.path.join("wavelock", "chain")):
            if "WaveLock.py" in files:
                return root
    return None

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
CHAIN = find_chain(repo_root)

if CHAIN is None:
    raise RuntimeError("Could not find wavelock/chain/WaveLock.py")

sys.path.insert(0, CHAIN)

from WaveLock import CurvatureKeyPair, _curvature_functional


# ------------------------------------------------------------
# Detect PDE operator
# ------------------------------------------------------------
def get_pde_operator(kp):
    """
    Try available methods in real WaveLock.
    """
    if hasattr(kp, "evolve"):
        return lambda psi: kp.evolve(psi, kp.n)
    if hasattr(kp, "_evolve"):
        return lambda psi: kp._evolve(psi)
    raise RuntimeError("WaveLock has no evolve() or _evolve()")


# ------------------------------------------------------------
# Wrapper for PDE evolution
# ------------------------------------------------------------
def evolve_field(psi, kp, pde_step):
    psi = cp.asarray(psi, dtype=cp.float64)
    return pde_step(psi)


# ------------------------------------------------------------
# CURVATURE ATTACK
# ------------------------------------------------------------
def curvature_attack(seed=123, n=32, steps=500, lr=0.004, lam=0.1):

    print(f"=== WCT CURVATURE ATTACK (n={n}, steps={steps}) ===")

    # Build dummy KP without auto-evolve
    kp = CurvatureKeyPair.__new__(CurvatureKeyPair)
    kp.n = n
    kp.T = 20  # evolution depth (matches Wavelock defaults)

    pde_step = get_pde_operator(kp)

    # target field
    rng = cp.random.default_rng(seed)
    psi0_true = rng.standard_normal((n, n))
    psiT_target = evolve_field(psi0_true, kp, pde_step)

    # guess
    psi0_guess = cp.zeros_like(psi0_true)
    m = 0.0

    for it in tqdm(range(steps)):

        psiT_guess = evolve_field(psi0_guess, kp, pde_step)
        loss_main = cp.mean((psiT_guess - psiT_target)**2)

        # curvature term
        E_grad, E_fb, E_ent, E_tot = _curvature_functional(psi0_guess)

        loss = loss_main + lam * E_tot

        # finite difference gradient
        eps = 1e-3
        psi_plus  = psi0_guess + eps
        psi_minus = psi0_guess - eps

        T_plus  = evolve_field(psi_plus, kp, pde_step)
        T_minus = evolve_field(psi_minus, kp, pde_step)

        g_inv = (T_plus - T_minus) / (2 * eps)

        E_plus  = _curvature_functional(psi_plus)[3]
        E_minus = _curvature_functional(psi_minus)[3]
        g_curv = (E_plus - E_minus) / (2 * eps)

        grad = g_inv + lam * g_curv

        # update
        m = 0.9 * m + 0.1 * grad
        psi0_guess -= lr * m

        if it % 100 == 0:
            print(f"[CURV] iter={it} loss={float(loss):.5f} main={float(loss_main):.5f} E={float(E_tot):.5f}")

    print("\n=== RESULT ===")
    print("Final loss:", float(loss))
    return float(loss)


# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
if __name__ == "__main__":
    curvature_attack()
