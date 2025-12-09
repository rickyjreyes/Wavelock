import os, sys
import cupy as cp
from tqdm import tqdm

# ------------------------------------------------------------
# Locate real WaveLock (avoid accidental archive versions)
# ------------------------------------------------------------
def find_chain(start):
    for root, dirs, files in os.walk(start):
        if root.endswith(os.path.join("wavelock", "chain")) and "WaveLock.py" in files:
            return root
    return None

repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
CHAIN = find_chain(repo_root)

if CHAIN is None:
    raise RuntimeError("Could not find wavelock/chain/WaveLock.py")

if CHAIN not in sys.path:
    sys.path.insert(0, CHAIN)

from WaveLock import CurvatureKeyPair


# ------------------------------------------------------------
# PDE operator helper (reuse real WaveLock evolution)
# ------------------------------------------------------------
def get_pde_operator(kp):
    """
    Returns a callable F(psi) that applies the WaveLock PDE.
    Works whether WaveLock exposes evolve() or _evolve().
    """
    if hasattr(kp, "evolve"):
        # Standard interface: evolve(psi, n_steps)
        return lambda psi: kp.evolve(psi, kp.n)
    if hasattr(kp, "_evolve"):
        # Older/alt interface: _evolve(psi)
        return lambda psi: kp._evolve(psi)
    raise RuntimeError("WaveLock has no evolve() or _evolve()")


def evolve_field(psi, kp, pde_step):
    """One forward WaveLock evolution ψ0 → ψT."""
    psi = cp.asarray(psi, dtype=cp.float64)
    return pde_step(psi)


# ------------------------------------------------------------
# Dual "Hamiltonians" (directional transport operators)
# ------------------------------------------------------------
def H1(f):
    return f - 0.1 * (cp.roll(f, -1, 0) - cp.roll(f, 1, 0))


def H2(f):
    return f - 0.1 * (cp.roll(f, -1, 1) - cp.roll(f, 1, 1))


# ------------------------------------------------------------
# Dual-Hamiltonian pseudo-inverse attack
# ------------------------------------------------------------
def dual_hamiltonian_attack(seed=123, n=32, steps=500, lr=0.03, T_pde=20):
    """
    Try to recover ψ0 given ψT under the real WaveLock PDE,
    using two approximate commuting 'Hamiltonians' H1, H2 as
    constraints. Uses a scalar finite-difference gradient on
    the global loss (no backprop, no Jacobian → OOM-safe).
    """
    print(
        f"\n=== DUAL-HAMILTONIAN PSEUDO-INVERSE ATTACK "
        f"(n={n}, steps={steps}, T={T_pde}) ==="
    )

    # Build dummy CurvatureKeyPair WITHOUT calling __init__.
    # This avoids the heavy auto-evolution / recursion path
    # that was causing the OOM in the original test.
    kp = CurvatureKeyPair.__new__(CurvatureKeyPair)
    kp.n = n
    kp.T = T_pde  # depth hint for evolve(), if it uses T

    pde_step = get_pde_operator(kp)

    # Ground-truth forward map ψ0 → ψT
    cp.random.seed(seed)
    psi0_true = cp.random.standard_normal((n, n), dtype=cp.float64)
    psiT_target = evolve_field(psi0_true, kp, pde_step)

    # Attacker's initialization
    psi_guess = cp.zeros_like(psi0_true)
    m = 0.0  # scalar momentum

    for it in tqdm(range(steps)):
        psiT_guess = evolve_field(psi_guess, kp, pde_step)

        loss = cp.mean(
            (H1(psiT_guess) - H1(psiT_target)) ** 2
            + (H2(psiT_guess) - H2(psiT_target)) ** 2
        )

        # Finite-difference scalar gradient along the global ψ direction
        eps = 1e-3
        psi_plus = psi_guess + eps
        psi_minus = psi_guess - eps

        psiT_plus = evolve_field(psi_plus, kp, pde_step)
        psiT_minus = evolve_field(psi_minus, kp, pde_step)

        Lp = cp.mean(
            (H1(psiT_plus) - H1(psiT_target)) ** 2
            + (H2(psiT_plus) - H2(psiT_target)) ** 2
        )
        Lm = cp.mean(
            (H1(psiT_minus) - H1(psiT_target)) ** 2
            + (H2(psiT_minus) - H2(psiT_target)) ** 2
        )

        grad = (Lp - Lm) / (2 * eps)

        # Broadcast scalar update over the whole field
        m = 0.9 * m + 0.1 * grad
        psi_guess -= lr * m

        if it % 100 == 0:
            print(f"[DualHam] iter={it:4d} loss={float(loss):.6f}")

    print("Final loss =", float(loss))
    return float(loss)








if __name__ == "__main__":
    dual_hamiltonian_attack()
