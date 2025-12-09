import cupy as cp
from tqdm import tqdm

# ============================================================
# WCT-Like Curvature PDE (Nonlinear, Multi-Step)
# ============================================================

def laplacian(f):
    """5-point stencil Laplacian on a periodic 2D grid."""
    return (
        -4.0 * f
        + cp.roll(f, 1, axis=0)
        + cp.roll(f, -1, axis=0)
        + cp.roll(f, 1, axis=1)
        + cp.roll(f, -1, axis=1)
    )


def wct_pde_step(psi, dt=0.05):
    """
    WCT-style curvature PDE:
        ψ_{t+1} = ψ_t + dt * F(ψ_t)
    with
        F(ψ) = α Δψ - β (ψ^3 - ψ) - γ log(ψ^2 + ε)
    This closely mirrors the PDE form from the WaveLock paper.
    """
    alpha = 0.25
    beta  = 0.10
    gamma = 0.05
    eps   = 1e-6

    lap = laplacian(psi)
    nonlin = -beta * (psi**3 - psi)
    barrier = -gamma * cp.log(psi**2 + eps)

    return psi + dt * (alpha * lap + nonlin + barrier)


def wct_evolve(psi0, steps=40):
    """Evolve ψ0 forward under the WCT PDE for `steps` iterations."""
    psi = psi0.copy()
    for _ in range(steps):
        psi = wct_pde_step(psi)
    return psi


# ============================================================
# Curvature Functional (WCT-style)
# ============================================================

def curvature_energy(psi):
    """
    Simple curvature functional:
        E_curv = ⟨ |Δψ|^2 ⟩ + small L2 term
    This mimics curvature budget / mixing in WCT.
    """
    lap = laplacian(psi)
    E_lap = cp.mean(lap**2)
    E_l2  = cp.mean(psi**2)
    return E_lap + 1e-3 * E_l2


# ============================================================
# Mode Basis Construction (Low-Dimensional WCT Manifold)
# ============================================================

def make_mode_basis(N, K=8, seed=12345):
    """
    Build K orthonormal curvature modes in R^{N x N} using
    Gram–Schmidt on random fields. This defines a low-dimensional
    'curvature manifold' where we search for ψ0.
    """
    rng = cp.random.default_rng(seed)
    M = []
    for _ in range(K * 3):  # oversample to handle near-degenerate draws
        v = rng.standard_normal((N, N), dtype=cp.float64)
        v = v.reshape(-1)
        # Gram–Schmidt
        for u in M:
            v -= (cp.vdot(u, v) / cp.vdot(u, u)) * u
        nrm = cp.linalg.norm(v)
        if nrm < 1e-6:
            continue
        M.append(v / nrm)
        if len(M) >= K:
            break

    if len(M) < K:
        raise RuntimeError(f"Could only build {len(M)} modes, expected {K}")

    basis = cp.stack(M, axis=0)  # (K, N^2)
    return basis


def coeffs_to_field(a, basis, shape):
    """
    Map mode coefficients a (K,) to a full ψ field (shape = (N,N)):
        ψ0 = Σ_i a_i * mode_i
    """
    # basis: (K, N^2), a: (K,)
    flat = (a[:, None] * basis).sum(axis=0)
    return flat.reshape(shape)


# ============================================================
# WCT Mode–Basis TBJA Attack
# ============================================================

def wct_modes_tbja(
    seed=123,
    N=32,
    K=8,
    steps=600,
    T_pde=40,
    lr=0.05,
    lam_curv=1.0,
    eps_fd=1e-3,
):
    """
    WCT-style 'best effort' inversion:
    - True ψ0 sampled from N(0,1)
    - Target ψT = WCT_evolve(ψ0, T_pde)
    - ψ0_guess restricted to a K-dimensional curvature mode basis
    - Optimize coefficients a ∈ R^K to minimize:

        L(a) = || ψT_guess - ψT_target ||^2
               + λ_curv * (E_curv(ψ0_guess) - E_curv(ψ0_true))^2

    using finite-difference TBJA in mode space.
    """

    print(
        f"\n=== WCT MODES-TBJA START (N={N}, K={K}, "
        f"T_pde={T_pde}, steps={steps}) ==="
    )

    # --- Ground truth ---
    rng = cp.random.default_rng(seed)
    psi0_true = rng.standard_normal((N, N), dtype=cp.float64)
    psiT_target = wct_evolve(psi0_true, steps=T_pde)
    E_curv_true = curvature_energy(psi0_true)

    # --- Mode basis ---
    basis = make_mode_basis(N, K=K, seed=seed + 999)
    basis = basis.astype(cp.float64)

    # --- Coefficients initialization ---
    a = cp.zeros(K, dtype=cp.float64)
    m = cp.zeros_like(a)  # momentum (Adam-lite)

    best_loss = float("inf")
    best_a = None

    for it in tqdm(range(steps)):
        # Current guess ψ0 and forward evolution
        psi0_guess = coeffs_to_field(a, basis, psi0_true.shape)
        psiT_guess = wct_evolve(psi0_guess, steps=T_pde)

        # Main MSE loss
        diff = psiT_guess - psiT_target
        L_main = cp.mean(diff**2)

        # Curvature-matching penalty on ψ0
        E_curv_guess = curvature_energy(psi0_guess)
        L_curv = (E_curv_guess - E_curv_true) ** 2

        L_total = L_main + lam_curv * L_curv

        # Finite-difference gradient in mode space
        grad = cp.zeros_like(a)
        for i in range(K):
            ai_plus = a.copy()
            ai_minus = a.copy()
            ai_plus[i] += eps_fd
            ai_minus[i] -= eps_fd

            # ψ0_plus / ψ0_minus in mode manifold
            psi0_plus = coeffs_to_field(ai_plus, basis, psi0_true.shape)
            psi0_minus = coeffs_to_field(ai_minus, basis, psi0_true.shape)

            # Forward evolution
            psiT_plus = wct_evolve(psi0_plus, steps=T_pde)
            psiT_minus = wct_evolve(psi0_minus, steps=T_pde)

            # Loss_plus
            diff_plus = psiT_plus - psiT_target
            L_main_plus = cp.mean(diff_plus**2)
            E_curv_plus = curvature_energy(psi0_plus)
            L_curv_plus = (E_curv_plus - E_curv_true) ** 2
            L_plus = L_main_plus + lam_curv * L_curv_plus

            # Loss_minus
            diff_minus = psiT_minus - psiT_target
            L_main_minus = cp.mean(diff_minus**2)
            E_curv_minus = curvature_energy(psi0_minus)
            L_curv_minus = (E_curv_minus - E_curv_true) ** 2
            L_minus = L_main_minus + lam_curv * L_curv_minus

            grad[i] = (L_plus - L_minus) / (2.0 * eps_fd)

        # Adam-lite update in coefficient space
        beta = 0.9
        m = beta * m + (1.0 - beta) * grad
        a = a - lr * m

        if L_total < best_loss:
            best_loss = float(L_total)
            best_a = a.copy()

        if it % 50 == 0:
            print(
                f"[WCT-TBJA] iter={it:4d}  "
                f"L_main={float(L_main):.6f}  "
                f"L_curv={float(L_curv):.6f}  "
                f"L_total={float(L_total):.6f}"
            )

    # Final stats
    psi0_best = coeffs_to_field(best_a, basis, psi0_true.shape)
    # Flatten for correlation
    v_true = psi0_true.ravel()
    v_best = psi0_best.ravel()
    num = float(cp.dot(v_true, v_best))
    den = float(cp.linalg.norm(v_true) * cp.linalg.norm(v_best) + 1e-12)
    corr = num / den

    print("\n=== WCT MODES-TBJA RESULT ===")
    print("Best total loss :", best_loss)
    print("Final correlation(ψ0_best, ψ0_true) :", corr)
    return best_loss, corr








if __name__ == "__main__":
    wct_modes_tbja()
