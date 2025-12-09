import os
import sys
import time

import numpy as np
import cupy as cp
from tqdm import tqdm

# ============================================================
# Try to import Wavelock curvature functional (optional)
# ============================================================

def _load_curvature_functional():
    """
    Try to import WaveLock._curvature_functional.
    If that fails, fall back to a gradient-energy curvature.
    """
    # Try package-style import first
    try:
        from wavelock.chain.WaveLock import _curvature_functional  # type: ignore
        print("[Surrogate] Using WaveLock._curvature_functional()")
        return _curvature_functional
    except Exception:
        pass

    # Try local WaveLock.py
    try:
        from WaveLock import _curvature_functional  # type: ignore
        print("[Surrogate] Using local WaveLock._curvature_functional()")
        return _curvature_functional
    except Exception:
        pass

    # Fallback: WCT-ish curvature from gradients
    print("[Surrogate] WARNING: Using fallback gradient-based curvature functional")

    def _fallback_curvature(psi):
        psi = cp.asarray(psi, dtype=cp.float64)
        gx, gy = cp.gradient(psi)
        E_grad = cp.mean(gx * gx + gy * gy)
        E_fb = cp.array(0.0)
        E_ent = cp.array(0.0)
        E_tot = E_grad
        return E_grad, E_fb, E_ent, E_tot

    return _fallback_curvature


_curvature_functional = _load_curvature_functional()


# ============================================================
# WCT-style surrogate PDE backend (GPU-safe)
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
    Lightweight curvature-like PDE:
    - diffusive Laplacian
    - double-well nonlinearity
    """
    psi = cp.asarray(psi, dtype=cp.float64)
    lap = laplacian(psi)
    nonlin = -BETA * (psi**3 - psi)
    psi_next = psi + DT * (ALPHA * lap + nonlin)
    # sanitize to avoid runaway growth
    psi_next = cp.nan_to_num(psi_next, nan=0.0, posinf=1e6, neginf=-1e6)
    psi_next = cp.clip(psi_next, -1e3, 1e3)
    return psi_next


def forward_pde(psi0, T):
    psi = cp.asarray(psi0, dtype=cp.float64)
    for _ in range(T):
        psi = surrogate_pde_step(psi)
    return psi


def curvature_energy(psi):
    psi = cp.asarray(psi, dtype=cp.float64)
    try:
        _, _, _, E_tot = _curvature_functional(psi)
        return float(cp.asnumpy(E_tot))
    except Exception:
        gx, gy = cp.gradient(psi)
        E_grad = cp.mean(gx * gx + gy * gy)
        return float(cp.asnumpy(E_grad))


def mse(a, b):
    a = cp.asarray(a, dtype=cp.float64)
    b = cp.asarray(b, dtype=cp.float64)
    return cp.mean((a - b) ** 2)


def corr_coeff(a, b):
    """
    Pearson correlation on CPU for reporting.
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

    This keeps curvature terms on a similar scale as MSE.
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
# Forward problem: ψ0_true -> ψT_target
# ============================================================

def make_forward_problem(seed=123, n=32, T_pde=30):
    cp.random.seed(seed)
    psi0_true = cp.random.standard_normal((n, n), dtype=cp.float64)
    psiT_target = forward_pde(psi0_true, T_pde)
    E_target = curvature_energy(psiT_target)
    return psi0_true, psiT_target, E_target


# ============================================================
# 1. Curvature Resonance-Manifold Attack
# ============================================================

def curvature_resonance_attack(psiT_target, E_target, T_pde, n, steps=200, K=8):
    """
    Tries to tune coefficients in a random WCT-mode basis to match
    BOTH the forward PDE output and its curvature energy.
    """
    print("\n=== CURVATURE RESONANCE-MANIFOLD ATTACK ===")

    cp.random.seed(42)

    # random orthogonal-ish modes
    modes = []
    for k in range(K):
        v = cp.random.standard_normal((n, n), dtype=cp.float64)
        v = v - cp.mean(v)
        for j in range(len(modes)):
            vj = modes[j]
            num = cp.vdot(vj.ravel(), v.ravel())
            den = cp.vdot(vj.ravel(), vj.ravel()) + 1e-12
            v = v - (num / den) * vj
        v = v / (cp.linalg.norm(v) + 1e-12)
        modes.append(v)
    modes = cp.stack(modes, axis=0)  # (K, n, n)

    coeff = cp.zeros((K,), dtype=cp.float64)
    vel = cp.zeros_like(coeff)

    eps = 1e-3
    best_loss = float("inf")
    best_coeff = coeff.copy()

    for it in tqdm(range(steps), desc="CurvRes", leave=False):
        psi0_guess = cp.tensordot(coeff, modes, axes=1)
        # keep initial amplitudes bounded
        psi0_guess = cp.clip(psi0_guess, -10.0, 10.0)

        psiT_guess = forward_pde(psi0_guess, T_pde)

        if not bool(cp.isfinite(psiT_guess).all()):
            print(f"[CurvRes] non-finite ψ encountered at iter={it}, aborting.")
            break

        L_main = mse(psiT_guess, psiT_target)
        E_guess = curvature_energy(psiT_guess)
        L_curv = safe_curvature_loss(E_guess, E_target)
        loss = L_main + 1e-3 * L_curv

        if (not np.isfinite(float(loss))
                or not np.isfinite(float(L_main))
                or not np.isfinite(E_guess)):
            print(
                f"[CurvRes] non-finite loss at iter={it}: "
                f"L={float(loss)}, L_main={float(L_main)}, E_guess={E_guess}"
            )
            break

        # finite-diff gradient w.r.t coeffs
        grad = cp.zeros_like(coeff)
        for i in range(K):
            coeff[i] += eps
            psi0_p = cp.tensordot(coeff, modes, axes=1)
            psi0_p = cp.clip(psi0_p, -10.0, 10.0)
            psiT_p = forward_pde(psi0_p, T_pde)
            Lp_main = mse(psiT_p, psiT_target)
            Ep = curvature_energy(psiT_p)
            Lp = Lp_main + 1e-3 * safe_curvature_loss(Ep, E_target)

            coeff[i] -= 2 * eps
            psi0_m = cp.tensordot(coeff, modes, axes=1)
            psi0_m = cp.clip(psi0_m, -10.0, 10.0)
            psiT_m = forward_pde(psi0_m, T_pde)
            Lm_main = mse(psiT_m, psiT_target)
            Em = curvature_energy(psiT_m)
            Lm = Lm_main + 1e-3 * safe_curvature_loss(Em, E_target)

            coeff[i] += eps
            grad[i] = (Lp - Lm) / (2 * eps)

        vel = 0.9 * vel + 0.1 * grad
        coeff -= 0.03 * vel

        # clamp coeffs to avoid blow-ups
        coeff = cp.clip(coeff, -100.0, 100.0)

        if float(loss) < best_loss and np.isfinite(float(loss)):
            best_loss = float(loss)
            best_coeff = coeff.copy()

        if it % 50 == 0:
            print(
                f"[CurvRes] iter={it:4d}  "
                f"L={float(loss):.6f}  L_main={float(L_main):.6f}  "
                f"L_curv={L_curv:.6e}  E_guess={E_guess:.6f}"
            )

    psi0_best = cp.tensordot(best_coeff, modes, axes=1)
    psi0_best = cp.clip(psi0_best, -10.0, 10.0)
    return psi0_best, best_loss


# ============================================================
# 2. Multi-Resolution Chaotic Envelope Descent
# ============================================================

def _downsample(psi, factor):
    n = psi.shape[0]
    assert n % factor == 0
    m = n // factor
    psi = psi.reshape(m, factor, m, factor)
    return psi.mean(axis=(1, 3))


def _upsample(psi_small, factor):
    psi_small = cp.asarray(psi_small, dtype=cp.float64)
    m = psi_small.shape[0]
    psi = psi_small.repeat(factor, axis=0).repeat(factor, axis=1)
    return psi


def multires_attack(psiT_target, E_target, T_pde, n, steps=150):
    print("\n=== MULTI-RES CHAOTIC ENVELOPE DESCENT ===")

    factor = 4
    m = n // factor

    cp.random.seed(2025)
    psi0_small = cp.random.standard_normal((m, m), dtype=cp.float64)
    vel = cp.zeros_like(psi0_small)

    eps = 1e-3
    best_loss = float("inf")
    best_small = psi0_small.copy()

    for it in tqdm(range(steps), desc="MultiRes", leave=False):
        psi0_guess = _upsample(psi0_small, factor)
        psi0_guess = cp.clip(psi0_guess, -10.0, 10.0)

        psiT_guess = forward_pde(psi0_guess, T_pde)

        if not bool(cp.isfinite(psiT_guess).all()):
            print(f"[MultiRes] non-finite ψ encountered at iter={it}, aborting.")
            break

        L_main = mse(psiT_guess, psiT_target)
        E_guess = curvature_energy(psiT_guess)
        L_curv = safe_curvature_loss(E_guess, E_target)
        loss = L_main + 1e-3 * L_curv

        if (not np.isfinite(float(loss))
                or not np.isfinite(float(L_main))
                or not np.isfinite(E_guess)):
            print(
                f"[MultiRes] non-finite loss at iter={it}: "
                f"L={float(loss)}, L_main={float(L_main)}, E_guess={E_guess}"
            )
            break

        # finite-diff gradient on coarse grid
        grad = cp.zeros_like(psi0_small)
        for i in range(m):
            for j in range(m):
                psi0_small[i, j] += eps
                psiT_p = forward_pde(_upsample(psi0_small, factor), T_pde)
                Lp_main = mse(psiT_p, psiT_target)
                Ep = curvature_energy(psiT_p)
                Lp = Lp_main + 1e-3 * safe_curvature_loss(Ep, E_target)

                psi0_small[i, j] -= 2 * eps
                psiT_m = forward_pde(_upsample(psi0_small, factor), T_pde)
                Lm_main = mse(psiT_m, psiT_target)
                Em = curvature_energy(psiT_m)
                Lm = Lm_main + 1e-3 * safe_curvature_loss(Em, E_target)

                psi0_small[i, j] += eps
                grad[i, j] = (Lp - Lm) / (2 * eps)

        vel = 0.9 * vel + 0.1 * grad
        psi0_small -= 0.05 * vel

        # clamp coarse field
        psi0_small = cp.clip(psi0_small, -10.0, 10.0)

        if float(loss) < best_loss and np.isfinite(float(loss)):
            best_loss = float(loss)
            best_small = psi0_small.copy()

        if it % 20 == 0:
            print(
                f"[MultiRes] iter={it:4d}  "
                f"L={float(loss):.6f}  L_main={float(L_main):.6f}  "
                f"E_guess={E_guess:.6f}"
            )

    psi0_best = _upsample(best_small, factor)
    psi0_best = cp.clip(psi0_best, -10.0, 10.0)
    return psi0_best, best_loss


# ============================================================
# 3. Eigenmode Bundle Attack
# ============================================================

def eigenmode_bundle_attack(psiT_target, E_target, T_pde, n, steps=200, K=8):
    print("\n=== EIGENMODE BUNDLE ATTACK ===")

    cp.random.seed(7)
    modes = []
    for k in range(K):
        v = cp.random.standard_normal((n, n), dtype=cp.float64)
        v = v - cp.mean(v)
        for j in range(len(modes)):
            vj = modes[j]
            num = cp.vdot(vj.ravel(), v.ravel())
            den = cp.vdot(vj.ravel(), vj.ravel()) + 1e-12
            v = v - (num / den) * vj
        v = v / (cp.linalg.norm(v) + 1e-12)
        modes.append(v)
    modes = cp.stack(modes, axis=0)

    coeff = cp.zeros((K,), dtype=cp.float64)
    vel = cp.zeros_like(coeff)

    eps = 1e-3
    best_loss = float("inf")
    best_coeff = coeff.copy()

    for it in tqdm(range(steps), desc="EigenBundle", leave=False):
        psi0_guess = cp.tensordot(coeff, modes, axes=1)
        psi0_guess = cp.clip(psi0_guess, -10.0, 10.0)
        psiT_guess = forward_pde(psi0_guess, T_pde)

        if not bool(cp.isfinite(psiT_guess).all()):
            print(f"[EigenBundle] non-finite ψ encountered at iter={it}, aborting.")
            break

        L_main = mse(psiT_guess, psiT_target)
        loss = L_main  # pure modal TBJA here

        if not np.isfinite(float(loss)):
            print(f"[EigenBundle] non-finite loss at iter={it}: L={float(loss)}")
            break

        grad = cp.zeros_like(coeff)
        for i in range(K):
            coeff[i] += eps
            psi0_p = cp.tensordot(coeff, modes, axes=1)
            psi0_p = cp.clip(psi0_p, -10.0, 10.0)
            psiT_p = forward_pde(psi0_p, T_pde)
            Lp = mse(psiT_p, psiT_target)

            coeff[i] -= 2 * eps
            psi0_m = cp.tensordot(coeff, modes, axes=1)
            psi0_m = cp.clip(psi0_m, -10.0, 10.0)
            psiT_m = forward_pde(psi0_m, T_pde)
            Lm = mse(psiT_m, psiT_target)

            coeff[i] += eps
            grad[i] = (Lp - Lm) / (2 * eps)

        vel = 0.9 * vel + 0.1 * grad
        coeff -= 0.03 * vel
        coeff = cp.clip(coeff, -100.0, 100.0)

        if float(loss) < best_loss and np.isfinite(float(loss)):
            best_loss = float(loss)
            best_coeff = coeff.copy()

        if it % 50 == 0:
            print(f"[EigenBundle] iter={it:4d}  L={float(loss):.6f}")

    psi0_best = cp.tensordot(best_coeff, modes, axes=1)
    psi0_best = cp.clip(psi0_best, -10.0, 10.0)
    return psi0_best, best_loss


# ============================================================
# 4. NLS Surrogate Inverse Attack
# ============================================================

def _nls_step(phi, dt=0.05):
    phi = cp.asarray(phi, dtype=cp.complex128)
    lap = laplacian(phi.real) + 1j * laplacian(phi.imag)
    phi_lin = phi + 1j * dt * lap
    amp2 = cp.abs(phi_lin) ** 2
    phi_nl = phi_lin * cp.exp(-1j * dt * amp2)
    phi_nl = cp.nan_to_num(phi_nl, nan=0.0, posinf=1e6, neginf=-1e6)
    return phi_nl


def nls_attack(psiT_target, E_target, T_pde, n, steps=80):
    print("\n=== NLS SURROGATE INVERSE ATTACK ===")

    cp.random.seed(888)
    phi0 = (cp.random.standard_normal((n, n)) +
            1j * cp.random.standard_normal((n, n))) * 0.1
    vel = cp.zeros_like(phi0)

    eps = 5e-3
    best_loss = float("inf")
    best_phi0 = phi0.copy()

    for it in tqdm(range(steps), desc="NLS", leave=False):
        phi = phi0
        for _ in range(T_pde // 4):  # keep NLS depth modest
            phi = _nls_step(phi, dt=0.05)

        psiT_guess = phi.real.astype(cp.float64)
        psiT_guess = cp.clip(psiT_guess, -10.0, 10.0)

        if not bool(cp.isfinite(psiT_guess).all()):
            print(f"[NLS] non-finite ψ encountered at iter={it}, aborting.")
            break

        L_main = mse(psiT_guess, psiT_target)
        E_guess = curvature_energy(psiT_guess)
        L_curv = safe_curvature_loss(E_guess, E_target)
        loss = L_main + 1e-3 * L_curv

        if (not np.isfinite(float(loss))
                or not np.isfinite(float(L_main))
                or not np.isfinite(E_guess)):
            print(
                f"[NLS] non-finite loss at iter={it}: "
                f"L={float(loss)}, L_main={float(L_main)}, E_guess={E_guess}"
            )
            break

        # extremely cheap "gradient": random direction estimate
        noise = (cp.random.standard_normal(phi0.shape) +
                 1j * cp.random.standard_normal(phi0.shape))
        grad_est = noise * float(loss)

        vel = 0.9 * vel + 0.1 * grad_est
        phi0 -= 0.01 * vel

        # clamp NaNs / infinities in φ
        phi0 = cp.nan_to_num(phi0, nan=0.0, posinf=1e6, neginf=-1e6)

        if float(loss) < best_loss and np.isfinite(float(loss)):
            best_loss = float(loss)
            best_phi0 = phi0.copy()

        if it % 20 == 0:
            print(
                f"[NLS] iter={it:4d}  L={float(loss):.6f}  "
                f"L_main={float(L_main):.6f}  E_guess={E_guess:.6f}"
            )

    psi0_best = best_phi0.real.astype(cp.float64)
    psi0_best = cp.clip(psi0_best, -10.0, 10.0)
    return psi0_best, best_loss


# ============================================================
# 5. Topological Mapper Homotopy Attack
# ============================================================

def topological_mapper_attack(psiT_target, E_target, T_pde, n, steps=150):
    print("\n=== TOPOLOGICAL MAPPER HOMOTOPY ATTACK ===")

    cp.random.seed(999)
    psi0 = cp.random.standard_normal((n, n), dtype=cp.float64) * 0.5

    def loss_fn(field):
        psiT_guess = forward_pde(field, T_pde)
        if not bool(cp.isfinite(psiT_guess).all()):
            return 1e18
        L_main = mse(psiT_guess, psiT_target)
        E_guess = curvature_energy(psiT_guess)
        L_curv = safe_curvature_loss(E_guess, E_target)
        total = float(L_main + 1e-3 * L_curv)
        if not np.isfinite(total):
            return 1e18
        return total

    current_loss = loss_fn(psi0)
    best_loss = current_loss
    best_field = psi0.copy()

    for it in tqdm(range(steps), desc="TopoMap", leave=False):
        new = psi0.copy()
        # random blobs / sign flips
        for _ in range(np.random.randint(1, 4)):
            cx = np.random.randint(0, n)
            cy = np.random.randint(0, n)
            radius = np.random.randint(1, max(2, n // 4))
            for i in range(max(0, cx - radius), min(n, cx + radius)):
                for j in range(max(0, cy - radius), min(n, cy + radius)):
                    dx = i - cx
                    dy = j - cy
                    if dx * dx + dy * dy <= radius * radius:
                        new[i, j] *= -1.0

        new = cp.clip(new, -10.0, 10.0)
        proposal_loss = loss_fn(new)
        if proposal_loss < current_loss:
            psi0 = new
            current_loss = proposal_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_field = psi0.copy()

        if it % 20 == 0:
            print(
                f"[TopoMap] iter={it:4d}  current={current_loss:.6f}  "
                f"best={best_loss:.6f}"
            )

    return best_field, best_loss


# ============================================================
# 6. Phase-Space Adjoint Injection (pseudo-Hamiltonian)
# ============================================================

def phase_space_attack(psiT_target, E_target, T_pde, n, steps=120):
    """
    Treat (psi, p) as phase-space:
    psi_{t+1} = psi_t + dt * p_t
    p_{t+1}   = p_t - dt * grad_psi H(psi)
    where H ~ curvature energy.
    We try to optimize initial (psi0, p0) to match psiT_target after
    a few pseudo-Hamiltonian steps.
    """
    print("\n=== PHASE-SPACE ADJOINT INJECTION ATTACK ===")

    cp.random.seed(31415)
    psi0 = cp.random.standard_normal((n, n), dtype=cp.float64) * 0.1
    p0 = cp.zeros_like(psi0)
    v_psi = cp.zeros_like(psi0)
    v_p = cp.zeros_like(p0)

    dt = 0.1
    eps = 1e-3
    best_loss = float("inf")
    best_psi0 = psi0.copy()
    best_p0 = p0.copy()

    for it in tqdm(range(steps), desc="PhaseSpace", leave=False):
        psi = psi0
        p = p0
        # small number of phase-space steps
        for _ in range(8):
            H_grad = laplacian(psi)  # cheap proxy for ∂H/∂psi
            psi = psi + dt * p
            p = p - dt * H_grad
            psi = cp.nan_to_num(psi, nan=0.0, posinf=1e6, neginf=-1e6)
            p = cp.nan_to_num(p, nan=0.0, posinf=1e6, neginf=-1e6)
            psi = cp.clip(psi, -1e3, 1e3)
            p = cp.clip(p, -1e3, 1e3)

        psiT_guess = forward_pde(psi, T_pde // 4)

        if not bool(cp.isfinite(psiT_guess).all()):
            print(f"[PhaseSpace] non-finite ψ encountered at iter={it}, aborting.")
            break

        L_main = mse(psiT_guess, psiT_target)
        E_guess = curvature_energy(psiT_guess)
        L_curv = safe_curvature_loss(E_guess, E_target)
        loss = L_main + 1e-3 * L_curv

        if (not np.isfinite(float(loss))
                or not np.isfinite(float(L_main))
                or not np.isfinite(E_guess)):
            print(
                f"[PhaseSpace] non-finite loss at iter={it}: "
                f"L={float(loss)}, L_main={float(L_main)}, E_guess={E_guess}"
            )
            break

        # randomized pseudo-gradient in phase-space
        noise_psi = cp.random.standard_normal(psi0.shape)
        noise_p = cp.random.standard_normal(p0.shape)
        grad_psi = noise_psi * float(loss)
        grad_p = noise_p * float(loss)

        v_psi = 0.9 * v_psi + 0.1 * grad_psi
        v_p = 0.9 * v_p + 0.1 * grad_p

        psi0 -= 0.02 * v_psi
        p0 -= 0.02 * v_p

        psi0 = cp.nan_to_num(psi0, nan=0.0, posinf=1e6, neginf=-1e6)
        p0 = cp.nan_to_num(p0, nan=0.0, posinf=1e6, neginf=-1e6)
        psi0 = cp.clip(psi0, -1e3, 1e3)
        p0 = cp.clip(p0, -1e3, 1e3)

        if float(loss) < best_loss and np.isfinite(float(loss)):
            best_loss = float(loss)
            best_psi0 = psi0.copy()
            best_p0 = p0.copy()

        if it % 20 == 0:
            print(
                f"[PhaseSpace] iter={it:4d}  L={float(loss):.6f}  "
                f"L_main={float(L_main):.6f}  E_guess={E_guess:.6f}"
            )

    return best_psi0, best_loss


# ============================================================
# Orchestrator: run all attacks & summarize
# ============================================================

def run_all_attacks(seed=123, n=32, T_pde=30):
    print("\n====================================")
    print("   WAVELOCK WCT SURROGATE ATTACKS   ")
    print("====================================\n")

    psi0_true, psiT_target, E_target = make_forward_problem(seed, n, T_pde)

    results = []

    # each attack returns (psi0_est, loss)
    psi0_est, loss = curvature_resonance_attack(psiT_target, E_target, T_pde, n)
    results.append(("CurvatureResonance", psi0_est, loss))

    psi0_est, loss = multires_attack(psiT_target, E_target, T_pde, n)
    results.append(("MultiResEnvelope", psi0_est, loss))

    psi0_est, loss = eigenmode_bundle_attack(psiT_target, E_target, T_pde, n)
    results.append(("EigenmodeBundle", psi0_est, loss))

    psi0_est, loss = nls_attack(psiT_target, E_target, T_pde, n)
    results.append(("NLSsurrogate", psi0_est, loss))

    psi0_est, loss = topological_mapper_attack(psiT_target, E_target, T_pde, n)
    results.append(("TopologicalMapper", psi0_est, loss))

    psi0_est, loss = phase_space_attack(psiT_target, E_target, T_pde, n)
    results.append(("PhaseSpaceAdjoint", psi0_est, loss))

    print("\n=== ATTACK SUMMARY ===")
    best_global = None
    best_loss = float("inf")
    best_corr = 0.0

    for name, psi_est, L in results:
        c = corr_coeff(psi_est, psi0_true)
        print(f"{name:22s}  loss={float(L):10.6f}  corr(ψ0_est, ψ0_true)={c:+.6f}")
        if float(L) < best_loss:
            best_loss = float(L)
            best_global = name
            best_corr = c

    print("\n=== FINAL RESULT ===")
    print(f"Best attack       : {best_global}")
    print(f"Best loss         : {best_loss:.6f}")
    print(f"Best correlation  : {best_corr:+.6f}")
    print("====================================\n")





if __name__ == "__main__":
    # Tuned for n=32, modest depth. Safe on a single GPU.
    run_all_attacks(seed=123, n=32, T_pde=30)
