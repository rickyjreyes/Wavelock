import os
import sys
import time

import numpy as np  # only used for final correlation on CPU printing

try:
    import cupy as cp
except ImportError as e:
    raise ImportError("This test requires CuPy (cupy). Please install cupy in your environment.") from e

from tqdm import tqdm


# ============================================================
# WaveLock curvature functional loader
# ============================================================

def _find_chain_root(start_dir):
    """
    Search upwards for wavelock/chain/WaveLock.py relative to tests/ directory.
    """
    for root, dirs, files in os.walk(start_dir):
        if root.endswith(os.path.join("wavelock", "chain")) and "WaveLock.py" in files:
            return root
    return None


def _load_curvature_functional():
    """
    Try very hard to import WaveLock._curvature_functional.
    Fallback to a simple gradient-based curvature if not found.
    """
    # 1) Search relative to repo root
    repo_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    chain_root = _find_chain_root(repo_root)

    if chain_root is not None:
        sys.path.insert(0, chain_root)
        try:
            from WaveLock import _curvature_functional  # type: ignore
            return _curvature_functional
        except Exception:
            pass

    # 2) Try package-style import
    try:
        from wavelock.chain.WaveLock import _curvature_functional  # type: ignore
        return _curvature_functional
    except Exception:
        pass

    # 3) Try local WaveLock.py
    try:
        from WaveLock import _curvature_functional  # type: ignore
        return _curvature_functional
    except Exception:
        pass

    # 4) Fallback definition (still WCT-flavored: gradient energy)
    def _fallback_curv(psi):
        psi = cp.asarray(psi, dtype=cp.float64)
        gx, gy = cp.gradient(psi)
        E_grad = cp.mean(gx * gx + gy * gy)
        E_fb = cp.array(0.0, dtype=cp.float64)
        E_ent = cp.array(0.0, dtype=cp.float64)
        E_tot = E_grad
        return E_grad, E_fb, E_ent, E_tot

    print("[TierOmega] WARNING: Using fallback curvature functional.")
    return _fallback_curv


_curvature_functional = _load_curvature_functional()


# ============================================================
# PDE backend: WCT-style surrogate (GPU-safe)
# ============================================================

DT = 0.10
ALPHA = 0.25
BETA = 0.05


def laplacian(psi):
    psi = cp.asarray(psi, dtype=cp.float64)
    return (
        -4.0 * psi
        + cp.roll(psi, 1, axis=0)
        + cp.roll(psi, -1, axis=0)
        + cp.roll(psi, 1, axis=1)
        + cp.roll(psi, -1, axis=1)
    )


def surrogate_pde_step(psi):
    """
    Lightweight WCT-flavored curvature PDE:
    - diffusive Laplacian
    - double-well nonlinearity
    """
    psi = cp.asarray(psi, dtype=cp.float64)
    lap = laplacian(psi)
    nonlin = -BETA * (psi**3 - psi)
    return psi + DT * (ALPHA * lap + nonlin)


def forward_pde(psi0, T):
    psi = cp.asarray(psi0, dtype=cp.float64)
    for _ in range(T):
        psi = surrogate_pde_step(psi)
    return psi


def curvature_energy(psi):
    psi = cp.asarray(psi, dtype=cp.float64)
    try:
        _, _, _, E_tot = _curvature_functional(psi)
        return cp.asnumpy(E_tot).item()
    except Exception:
        # fallback: just gradient energy
        gx, gy = cp.gradient(psi)
        E_tot = cp.mean(gx * gx + gy * gy)
        return float(cp.asnumpy(E_tot))


def mse(a, b):
    a = cp.asarray(a, dtype=cp.float64)
    b = cp.asarray(b, dtype=cp.float64)
    return cp.mean((a - b) ** 2)


def corr_coeff(a, b):
    """
    Pearson correlation coefficient on CPU (for reporting).
    """
    a = np.asarray(cp.asnumpy(a)).ravel()
    b = np.asarray(cp.asnumpy(b)).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    if denom == 0.0:
        return 0.0
    return float(a.dot(b) / denom)


# ============================================================
# Numerically safe curvature-loss helper (log-squashed)
# ============================================================

def safe_curvature_loss(E_guess: float, E_target: float, cap: float = 1e12) -> float:
    """
    Curvature loss in log-space for stability:
    - clamp energies to [-cap, cap]
    - map via sign(E) * log1p(|E|)
    - square the difference in this compressed space

    This keeps curvature terms on the same rough scale as MSE,
    even when raw energies are ~1e6–1e9.
    """
    def _norm_E(E: float) -> float:
        if not np.isfinite(E):
            return float(np.log1p(cap))
        E_clamped = float(np.clip(E, -cap, cap))
        return float(np.sign(E_clamped) * np.log1p(abs(E_clamped)))

    Eg = _norm_E(E_guess)
    Et = _norm_E(E_target)
    d = Eg - Et
    return d * d


# ============================================================
# Problem generator: forward map ψ0 -> ψT and curvature target
# ============================================================

def make_forward_problem(seed=123, n=32, T_pde=40):
    cp.random.seed(seed)
    psi0_true = cp.random.standard_normal((n, n), dtype=cp.float64)
    psiT_target = forward_pde(psi0_true, T_pde)
    E_target = curvature_energy(psiT_target)
    return psi0_true, psiT_target, E_target


# ============================================================
# Generic coefficient-basis descent engine
# ============================================================

def _build_random_modes(n, K, seed):
    cp.random.seed(seed)
    modes = []
    for k in range(K):
        v = cp.random.standard_normal((n, n), dtype=cp.float64)
        v = v - cp.mean(v)
        # Gram–Schmidt against previous modes
        for j in range(len(modes)):
            vj = modes[j]
            num = cp.vdot(vj.ravel(), v.ravel())
            den = cp.vdot(vj.ravel(), vj.ravel()) + 1e-12
            v = v - (num / den) * vj
        norm = cp.linalg.norm(v) + 1e-12
        v = v / norm
        modes.append(v)
    return cp.stack(modes, axis=0)  # (K, n, n)


def _coef_descent_agent(
    name,
    psiT_target,
    E_target,
    T_pde,
    n,
    K=8,
    steps=200,
    lr=0.03,
    lambda_curv=1e-3,
    lambda_reg=1e-4,
    seed=0,
):
    """
    Generic TBJA-style coefficient descent in a learned/random WCT-mode basis.
    """
    print(f"\n=== {name} (K={K}, n={n}, steps={steps}) ===")

    modes = _build_random_modes(n, K, seed=seed)
    coeff = cp.zeros((K,), dtype=cp.float64)
    velocity = cp.zeros_like(coeff)

    best_loss = float("inf")
    best_coeff = coeff.copy()

    eps = 1e-3

    for it in tqdm(range(steps), desc=name, leave=False):
        # Current reconstruction
        psi0_guess = cp.tensordot(coeff, modes, axes=1)
        psiT_guess = forward_pde(psi0_guess, T_pde)

        main = mse(psiT_guess, psiT_target)
        E_guess = curvature_energy(psiT_guess)
        curv_term = safe_curvature_loss(E_guess, E_target)
        reg_term = cp.mean(coeff * coeff)
        loss = main + lambda_curv * curv_term + lambda_reg * reg_term

        # Early exit on non-finite loss
        if not np.isfinite(float(loss)) or not np.isfinite(float(main)) or not np.isfinite(E_guess):
            print(
                f"[{name}] non-finite loss encountered at iter={it}: "
                f"L_total={float(loss)}, L_main={float(main)}, E_guess={E_guess}"
            )
            break

        # Finite-difference gradient w.r.t. coefficients
        grad = cp.zeros_like(coeff)
        for i in range(K):
            coeff[i] += eps
            psi0_p = cp.tensordot(coeff, modes, axes=1)
            psiT_p = forward_pde(psi0_p, T_pde)
            main_p = mse(psiT_p, psiT_target)
            E_p = curvature_energy(psiT_p)
            loss_p = main_p + lambda_curv * safe_curvature_loss(E_p, E_target) + lambda_reg * cp.mean(coeff * coeff)

            coeff[i] -= 2 * eps
            psi0_m = cp.tensordot(coeff, modes, axes=1)
            psiT_m = forward_pde(psi0_m, T_pde)
            main_m = mse(psiT_m, psiT_target)
            E_m = curvature_energy(psiT_m)
            loss_m = main_m + lambda_curv * safe_curvature_loss(E_m, E_target) + lambda_reg * cp.mean(coeff * coeff)

            coeff[i] += eps
            grad[i] = (loss_p - loss_m) / (2 * eps)

        # Momentum update
        velocity = 0.9 * velocity + 0.1 * grad
        coeff -= lr * velocity

        # Clamp coeffs to avoid blow-ups
        coeff = cp.clip(coeff, -1e3, 1e3)

        if float(loss) < best_loss:
            best_loss = float(loss)
            best_coeff = coeff.copy()

        if it % 50 == 0:
            print(
                f"[{name}] iter={it:4d}  "
                f"L_total={float(loss):.6f}  L_main={float(main):.6f}  "
                f"E_guess={E_guess:.6f}  curv_term={curv_term:.6e}"
            )

    psi0_best = cp.tensordot(best_coeff, modes, axes=1)
    return {
        "name": name,
        "psi0_best": psi0_best,
        "best_loss": best_loss,
    }


# ============================================================
# Multi-resolution chaotic envelope descent
# ============================================================

def _downsample(psi, factor):
    """
    Simple average pooling downsample.
    """
    n = psi.shape[0]
    assert n % factor == 0
    m = n // factor
    psi = psi.reshape(m, factor, m, factor)
    return psi.mean(axis=(1, 3))


def _upsample(psi_small, factor):
    """
    Nearest-neighbor upsample.
    """
    psi_small = cp.asarray(psi_small, dtype=cp.float64)
    m = psi_small.shape[0]
    psi = psi_small.repeat(factor, axis=0).repeat(factor, axis=1)
    return psi


def multires_envelope_attack(psiT_target, E_target, T_pde, n, steps=150, lr=0.05):
    """
    Multi-resolution chaotic envelope descent:
    - optimize a very coarse field
    - upsample to fine scale
    - enforce curvature similarity
    """
    print(f"\n=== Multi-Resolution Envelope Attack (n={n}, steps={steps}) ===")

    factor = 4
    m = n // factor

    cp.random.seed(2025)
    psi0_small = cp.random.standard_normal((m, m), dtype=cp.float64)
    velocity = cp.zeros_like(psi0_small)

    eps = 1e-3
    best_loss = float("inf")
    best_small = psi0_small.copy()

    for it in tqdm(range(steps), desc="MultiRes", leave=False):
        psi0_guess = _upsample(psi0_small, factor)
        psiT_guess = forward_pde(psi0_guess, T_pde)

        main = mse(psiT_guess, psiT_target)
        E_guess = curvature_energy(psiT_guess)
        curv_term = safe_curvature_loss(E_guess, E_target)
        loss = main + 1e-3 * curv_term

        if not np.isfinite(float(loss)) or not np.isfinite(float(main)) or not np.isfinite(E_guess):
            print(
                f"[MultiRes] non-finite loss encountered at iter={it}: "
                f"L_total={float(loss)}, L_main={float(main)}, E_guess={E_guess}"
            )
            break

        # Gradient via finite differences on coarse field
        grad = cp.zeros_like(psi0_small)
        for i in range(m):
            for j in range(m):
                psi0_small[i, j] += eps
                psiT_p = forward_pde(_upsample(psi0_small, factor), T_pde)
                main_p = mse(psiT_p, psiT_target)
                E_p = curvature_energy(psiT_p)
                loss_p = main_p + 1e-3 * safe_curvature_loss(E_p, E_target)

                psi0_small[i, j] -= 2 * eps
                psiT_m = forward_pde(_upsample(psi0_small, factor), T_pde)
                main_m = mse(psiT_m, psiT_target)
                E_m = curvature_energy(psiT_m)
                loss_m = main_m + 1e-3 * safe_curvature_loss(E_m, E_target)

                psi0_small[i, j] += eps
                grad[i, j] = (loss_p - loss_m) / (2 * eps)

        velocity = 0.9 * velocity + 0.1 * grad
        psi0_small -= lr * velocity

        # Clamp coarse field
        psi0_small = cp.clip(psi0_small, -10.0, 10.0)

        if float(loss) < best_loss:
            best_loss = float(loss)
            best_small = psi0_small.copy()

        if it % 20 == 0:
            print(
                f"[MultiRes] iter={it:4d}  L_total={float(loss):.6f}  "
                f"L_main={float(main):.6f}  E_guess={E_guess:.6f}"
            )

    psi0_best = _upsample(best_small, factor)
    return {
        "name": "MultiResEnvelope",
        "psi0_best": psi0_best,
        "best_loss": best_loss,
    }


# ============================================================
# NLS surrogate inverse operator (lighter, no insane finite diff)
# ============================================================

def _nls_step(phi, dt=0.05):
    """
    Simple cubic NLS step (split-step style, but crude).
    """
    phi = cp.asarray(phi, dtype=cp.complex128)
    lap = laplacian(phi.real) + 1j * laplacian(phi.imag)
    # linear part
    phi_lin = phi + 1j * dt * lap
    # nonlinear part
    amp2 = cp.abs(phi_lin) ** 2
    phi_nl = phi_lin * cp.exp(-1j * dt * amp2)
    return phi_nl


def nls_surrogate_inverse_attack(psiT_target, E_target, T_pde, n, steps=80, lr=0.01):
    """
    Try to recover ψ0 by embedding ψT into a complex NLS system and
    optimizing the initial complex field to match the observed ψT in its real part.

    This variant uses a cheap randomized gradient estimate instead of full finite differences.
    """
    print(f"\n=== NLS Surrogate Inverse Attack (n={n}, steps={steps}) ===")

    cp.random.seed(777)
    phi0 = (cp.random.standard_normal((n, n)) +
            1j * cp.random.standard_normal((n, n))) * 0.1
    velocity = cp.zeros_like(phi0)

    best_loss = float("inf")
    best_phi0 = phi0.copy()

    for it in tqdm(range(steps), desc="NLS", leave=False):
        phi = phi0
        # keep NLS depth modest to avoid blow-ups
        for _ in range(max(1, T_pde // 4)):
            phi = _nls_step(phi, dt=0.05)

        psiT_guess = phi.real.astype(cp.float64)
        main = mse(psiT_guess, psiT_target)
        E_guess = curvature_energy(psiT_guess)
        curv_term = safe_curvature_loss(E_guess, E_target)
        loss = main + 1e-3 * curv_term

        if not np.isfinite(float(loss)) or not np.isfinite(float(main)) or not np.isfinite(E_guess):
            print(
                f"[NLS] non-finite loss encountered at iter={it}: "
                f"L_total={float(loss)}, L_main={float(main)}, E_guess={E_guess}"
            )
            break

        # randomized pseudo-gradient in complex field
        noise = (cp.random.standard_normal(phi0.shape) +
                 1j * cp.random.standard_normal(phi0.shape))
        grad_est = noise * float(loss)

        velocity = 0.9 * velocity + 0.1 * grad_est
        phi0 -= lr * velocity

        # clamp NaNs / infinities in φ
        phi0 = cp.nan_to_num(phi0, nan=0.0, posinf=1e6, neginf=-1e6)

        if float(loss) < best_loss:
            best_loss = float(loss)
            best_phi0 = phi0.copy()

        if it % 20 == 0:
            print(
                f"[NLS] iter={it:4d}  L_total={float(loss):.6f}  "
                f"L_main={float(main):.6f}  E_guess={E_guess:.6f}"
            )

    # best reconstruction uses real part
    psi0_best = best_phi0.real.astype(cp.float64)
    return {
        "name": "NLSsurrogate",
        "psi0_best": psi0_best,
        "best_loss": best_loss,
    }


# ============================================================
# Simple topological mapper / homotopy-style random search
# ============================================================

def topological_mapper_attack(psiT_target, E_target, T_pde, n, steps=200, proposal_scale=0.1):
    """
    Very lightweight homotopy-style search:
    - maintain a current ψ0 guess
    - propose random topological perturbations (sign flips & blobs)
    - accept moves that decrease curvature-aware loss
    """
    print(f"\n=== Topological Mapper Homotopy Attack (n={n}, steps={steps}) ===")

    cp.random.seed(999)
    psi0 = cp.random.standard_normal((n, n), dtype=cp.float64) * 0.5

    def loss_fn(field):
        psiT_guess = forward_pde(field, T_pde)
        main = mse(psiT_guess, psiT_target)
        E_guess = curvature_energy(psiT_guess)
        curv_term = safe_curvature_loss(E_guess, E_target)
        total = float(main + 1e-3 * curv_term)
        if not np.isfinite(total):
            return 1e18
        return total

    current_loss = loss_fn(psi0)
    best_loss = current_loss
    best_field = psi0.copy()

    for it in tqdm(range(steps), desc="TopoMap", leave=False):
        # random blob
        blob = cp.zeros_like(psi0)
        num_blobs = np.random.randint(1, 4)
        for _ in range(num_blobs):
            cx = np.random.randint(0, n)
            cy = np.random.randint(0, n)
            radius = np.random.randint(1, max(2, n // 4))
            for i in range(max(0, cx - radius), min(n, cx + radius)):
                for j in range(max(0, cy - radius), min(n, cy + radius)):
                    dx = i - cx
                    dy = j - cy
                    if dx * dx + dy * dy <= radius * radius:
                        blob[i, j] = 1.0

        proposal = psi0 + proposal_scale * (2.0 * cp.random.rand(n, n) - 1.0) * blob
        proposal = cp.clip(proposal, -10.0, 10.0)
        proposal_loss = loss_fn(proposal)

        if proposal_loss < current_loss:
            psi0 = proposal
            current_loss = proposal_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_field = psi0.copy()

        if it % 20 == 0:
            print(f"[TopoMap] iter={it:4d}  current_loss={current_loss:.6f}  best_loss={best_loss:.6f}")

    return {
        "name": "TopologicalMapper",
        "psi0_best": best_field,
        "best_loss": best_loss,
    }


# ============================================================
# Tier-Ω Orchestrator
# ============================================================

def run_tier_omega(seed=123, n=32, T_pde=40):
    print("\n============================")
    print("   WAVELOCK TIER-Ω ATTACK   ")
    print("============================\n")

    psi0_true, psiT_target, E_target = make_forward_problem(seed=seed, n=n, T_pde=T_pde)

    agents = []

    # 1) Eigenmode bundle / coefficient descent
    agents.append(
        _coef_descent_agent(
            name="EigenmodeBundle",
            psiT_target=psiT_target,
            E_target=E_target,
            T_pde=T_pde,
            n=n,
            K=8,
            steps=200,   # was 300
            lr=0.03,
            lambda_curv=1e-3,
            lambda_reg=1e-4,
            seed=42,
        )
    )

    # 2) Multi-resolution chaotic envelope descent
    agents.append(
        multires_envelope_attack(
            psiT_target=psiT_target,
            E_target=E_target,
            T_pde=T_pde,
            n=n,
            steps=150,   # was 200
            lr=0.05,
        )
    )

    # 3) NLS surrogate inverse operator (lighter)
    agents.append(
        nls_surrogate_inverse_attack(
            psiT_target=psiT_target,
            E_target=E_target,
            T_pde=T_pde,
            n=n,
            steps=80,    # was 150 with huge inner loops
            lr=0.01,
        )
    )

    # 4) Topological mapper homotopy search
    agents.append(
        topological_mapper_attack(
            psiT_target=psiT_target,
            E_target=E_target,
            T_pde=T_pde,
            n=n,
            steps=200,
            proposal_scale=0.2,
        )
    )

    # ============================
    # Final ensemble summary
    # ============================
    print("\n=== TIER-OMEGA ENSEMBLE SUMMARY ===")

    best_global_loss = float("inf")
    best_global_agent = None
    best_global_psi0 = None

    for agent_result in agents:
        name = agent_result["name"]
        loss = float(agent_result["best_loss"])
        psi0_est = agent_result["psi0_best"]
        corr = corr_coeff(psi0_est, psi0_true)

        print(
            f"Agent: {name:20s}  "
            f"best_loss={loss:12.6f}  "
            f"corr(ψ0_est, ψ0_true)={corr:+.6f}"
        )

        if loss < best_global_loss:
            best_global_loss = loss
            best_global_agent = name
            best_global_psi0 = psi0_est

    if best_global_psi0 is not None:
        final_corr = corr_coeff(best_global_psi0, psi0_true)
    else:
        final_corr = 0.0

    print("\n=== TIER-OMEGA FINAL RESULT ===")
    print(f"Best agent        : {best_global_agent}")
    print(f"Best ensemble loss: {best_global_loss:.6f}")
    print(f"Final correlation : {final_corr:+.6f}")
    print("================================\n")









if __name__ == "__main__":
    # Default Tier-Ω configuration tuned for n=32 grid.
    # You can increase n or T_pde, but runtime will grow.
    run_tier_omega(seed=123, n=32, T_pde=40)
