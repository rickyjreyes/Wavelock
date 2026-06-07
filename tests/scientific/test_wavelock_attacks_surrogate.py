#!/usr/bin/env python3
# test_wavelock_attacks_surrogate.py
# ============================================================
# WaveLock WCT surrogate attack benchmark
#
# Updated version:
# - no environment commands required
# - internal fast/standard/deep profiles
# - replaces expensive full finite-difference gradients with SPSA-style
#   two-evaluation stochastic gradients where appropriate
# - adds per-attack timing and structured summaries
# - adds RISK_METRICS_BEGIN/END for run_benchmarks.py
# - makes "danger" explicit: only a meaningful inverse recovery is danger
# - keeps losses/correlations visible for audit instead of relying on prints
#
# The old version could run ~90 minutes because MultiRes used full finite
# differences over the coarse grid at every step. This version keeps the same
# attack families but makes the default profile much faster.
# ============================================================

import os
import sys
import time
import json
import traceback
from pathlib import Path

import numpy as np
import cupy as cp

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ============================================================
# CONFIG — edit here only
# ============================================================

PROFILE = "standard"

PROFILE_CONFIG = {
    "fast": {
        "seed": 123,
        "n": 32,
        "T_pde": 30,
        "curvres_steps": 60,
        "curvres_K": 8,
        "multires_steps": 50,
        "multires_factor": 4,
        "eigen_steps": 60,
        "eigen_K": 8,
        "nls_steps": 40,
        "topo_steps": 60,
        "phase_steps": 50,
    },
    "standard": {
        "seed": 123,
        "n": 32,
        "T_pde": 30,
        "curvres_steps": 120,
        "curvres_K": 10,
        "multires_steps": 100,
        "multires_factor": 4,
        "eigen_steps": 120,
        "eigen_K": 10,
        "nls_steps": 70,
        "topo_steps": 100,
        "phase_steps": 80,
    },
    "deep": {
        "seed": 123,
        "n": 32,
        "T_pde": 30,
        "curvres_steps": 250,
        "curvres_K": 12,
        "multires_steps": 200,
        "multires_factor": 4,
        "eigen_steps": 250,
        "eigen_K": 12,
        "nls_steps": 120,
        "topo_steps": 200,
        "phase_steps": 160,
    },
}

CORR_DANGER_THRESHOLD = 0.90
LOSS_DANGER_THRESHOLD = 1e-6
MAX_SECONDS = None

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

THIS_FILE = Path(__file__).resolve()
try:
    REPO_ROOT = THIS_FILE.parents[2]
except IndexError:
    REPO_ROOT = Path.cwd()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_curvature_functional():
    try:
        from wavelock.chain.WaveLock import _curvature_functional  # type: ignore
        print("[Surrogate] Using WaveLock._curvature_functional()")
        return _curvature_functional
    except Exception:
        pass

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

DT = 0.10
ALPHA = 0.25
BETA = 0.05


def sync_gpu():
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def finite_guard(x, label):
    arr = cp.asarray(x)
    if bool(cp.any(~cp.isfinite(arr)).get()):
        raise FloatingPointError(f"non-finite values detected in {label}")


def laplacian(psi):
    psi = cp.asarray(psi, dtype=cp.float64)
    return (
        -4.0 * psi
        + cp.roll(psi, 1, axis=-2)
        + cp.roll(psi, -1, axis=-2)
        + cp.roll(psi, 1, axis=-1)
        + cp.roll(psi, -1, axis=-1)
    )


def surrogate_pde_step(psi):
    psi = cp.asarray(psi, dtype=cp.float64)
    lap = laplacian(psi)
    nonlin = -BETA * (psi**3 - psi)
    psi_next = psi + DT * (ALPHA * lap + nonlin)
    psi_next = cp.nan_to_num(psi_next, nan=0.0, posinf=1e6, neginf=-1e6)
    psi_next = cp.clip(psi_next, -1e3, 1e3)
    return psi_next


def forward_pde(psi0, T):
    psi = cp.asarray(psi0, dtype=cp.float64)
    for _ in range(int(T)):
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
    a = np.asarray(cp.asnumpy(a)).ravel()
    b = np.asarray(cp.asnumpy(b)).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    if denom == 0.0:
        return 0.0
    return float(a.dot(b) / denom)


def safe_curvature_loss(E_guess: float, E_target: float, cap: float = 1e12) -> float:
    def _norm_E(E: float) -> float:
        if not np.isfinite(E):
            return float(np.log1p(cap))
        E_clamped = float(np.clip(E, -cap, cap))
        return float(np.sign(E_clamped) * np.log1p(abs(E_clamped)))

    d = _norm_E(E_guess) - _norm_E(E_target)
    return d * d


def total_loss_for_initial(psi0, psiT_target, E_target, T_pde, curvature_weight=1e-3):
    psi0 = cp.clip(cp.asarray(psi0, dtype=cp.float64), -10.0, 10.0)
    psiT_guess = forward_pde(psi0, T_pde)
    finite_guard(psiT_guess, "psiT_guess")
    L_main = float(mse(psiT_guess, psiT_target).get())
    E_guess = curvature_energy(psiT_guess)
    L_curv = safe_curvature_loss(E_guess, E_target)
    total = L_main + curvature_weight * L_curv
    if not np.isfinite(total):
        return 1e18, L_main, E_guess
    return float(total), float(L_main), float(E_guess)


def make_forward_problem(seed=123, n=32, T_pde=30):
    cp.random.seed(int(seed))
    psi0_true = cp.random.standard_normal((n, n), dtype=cp.float64)
    psiT_target = forward_pde(psi0_true, T_pde)
    E_target = curvature_energy(psiT_target)
    return psi0_true, psiT_target, E_target


def progress_iter(range_obj, desc):
    if tqdm is None:
        return range_obj
    return tqdm(range_obj, desc=desc, leave=False)


def timed_attack(name, fn, *args, **kwargs):
    print(f"\n=== {name} ===")
    t0 = time.time()
    try:
        psi0_est, loss = fn(*args, **kwargs)
        sync_gpu()
        runtime = time.time() - t0
        return {
            "name": name,
            "psi0_est": psi0_est,
            "loss": float(loss),
            "runtime_seconds": float(runtime),
            "crashed": False,
        }
    except Exception as e:
        runtime = time.time() - t0
        print(f"[CRASH] {name}: {repr(e)}")
        traceback.print_exc()
        return {
            "name": name,
            "psi0_est": None,
            "loss": float("inf"),
            "runtime_seconds": float(runtime),
            "crashed": True,
            "error": repr(e),
        }


def make_orthogonal_modes(n, K, seed):
    cp.random.seed(int(seed))
    modes = []
    for _ in range(K):
        v = cp.random.standard_normal((n, n), dtype=cp.float64)
        v = v - cp.mean(v)
        for vj in modes:
            num = cp.vdot(vj.ravel(), v.ravel())
            den = cp.vdot(vj.ravel(), vj.ravel()) + 1e-12
            v = v - (num / den) * vj
        v = v / (cp.linalg.norm(v) + 1e-12)
        modes.append(v)
    return cp.stack(modes, axis=0)


def _upsample(psi_small, factor):
    psi_small = cp.asarray(psi_small, dtype=cp.float64)
    return psi_small.repeat(factor, axis=0).repeat(factor, axis=1)


def curvature_resonance_attack(psiT_target, E_target, T_pde, n, steps=120, K=10):
    modes = make_orthogonal_modes(n, K, seed=42)
    coeff = cp.zeros((K,), dtype=cp.float64)
    vel = cp.zeros_like(coeff)
    best_loss = float("inf")
    best_coeff = coeff.copy()
    a = 0.08
    c = 0.04

    for it in progress_iter(range(int(steps)), "CurvRes"):
        psi0_guess = cp.tensordot(coeff, modes, axes=1)
        loss, L_main, E_guess = total_loss_for_initial(psi0_guess, psiT_target, E_target, T_pde)

        if loss < best_loss:
            best_loss = loss
            best_coeff = coeff.copy()

        delta = cp.random.choice(cp.array([-1.0, 1.0], dtype=cp.float64), size=coeff.shape)
        ck = c / ((it + 1) ** 0.101)
        ak = a / ((it + 1) ** 0.602)

        loss_p, _, _ = total_loss_for_initial(cp.tensordot(coeff + ck * delta, modes, axes=1), psiT_target, E_target, T_pde)
        loss_m, _, _ = total_loss_for_initial(cp.tensordot(coeff - ck * delta, modes, axes=1), psiT_target, E_target, T_pde)
        grad = ((loss_p - loss_m) / (2.0 * ck)) * delta

        vel = 0.9 * vel + 0.1 * grad
        coeff = cp.clip(coeff - ak * vel, -100.0, 100.0)

        if it % 25 == 0:
            print(f"[CurvRes] iter={it:4d} L={loss:.6f} L_main={L_main:.6f} E={E_guess:.6f}")

    psi0_best = cp.clip(cp.tensordot(best_coeff, modes, axes=1), -10.0, 10.0)
    return psi0_best, best_loss


def multires_attack(psiT_target, E_target, T_pde, n, steps=100, factor=4):
    m = n // factor
    cp.random.seed(2025)
    psi0_small = cp.random.standard_normal((m, m), dtype=cp.float64)
    vel = cp.zeros_like(psi0_small)
    best_loss = float("inf")
    best_small = psi0_small.copy()
    a = 0.08
    c = 0.03

    for it in progress_iter(range(int(steps)), "MultiRes"):
        psi0_guess = cp.clip(_upsample(psi0_small, factor), -10.0, 10.0)
        loss, L_main, E_guess = total_loss_for_initial(psi0_guess, psiT_target, E_target, T_pde)

        if loss < best_loss:
            best_loss = loss
            best_small = psi0_small.copy()

        delta = cp.random.choice(cp.array([-1.0, 1.0], dtype=cp.float64), size=psi0_small.shape)
        ck = c / ((it + 1) ** 0.101)
        ak = a / ((it + 1) ** 0.602)

        loss_p, _, _ = total_loss_for_initial(_upsample(psi0_small + ck * delta, factor), psiT_target, E_target, T_pde)
        loss_m, _, _ = total_loss_for_initial(_upsample(psi0_small - ck * delta, factor), psiT_target, E_target, T_pde)
        grad = ((loss_p - loss_m) / (2.0 * ck)) * delta

        vel = 0.9 * vel + 0.1 * grad
        psi0_small = cp.clip(psi0_small - ak * vel, -10.0, 10.0)

        if it % 20 == 0:
            print(f"[MultiRes] iter={it:4d} L={loss:.6f} L_main={L_main:.6f} E={E_guess:.6f}")

    psi0_best = cp.clip(_upsample(best_small, factor), -10.0, 10.0)
    return psi0_best, best_loss


def eigenmode_bundle_attack(psiT_target, E_target, T_pde, n, steps=120, K=10):
    modes = make_orthogonal_modes(n, K, seed=7)
    coeff = cp.zeros((K,), dtype=cp.float64)
    vel = cp.zeros_like(coeff)
    best_loss = float("inf")
    best_coeff = coeff.copy()
    a = 0.08
    c = 0.04

    for it in progress_iter(range(int(steps)), "EigenBundle"):
        psi0_guess = cp.tensordot(coeff, modes, axes=1)
        loss, L_main, _ = total_loss_for_initial(psi0_guess, psiT_target, E_target, T_pde, curvature_weight=0.0)

        if loss < best_loss:
            best_loss = loss
            best_coeff = coeff.copy()

        delta = cp.random.choice(cp.array([-1.0, 1.0], dtype=cp.float64), size=coeff.shape)
        ck = c / ((it + 1) ** 0.101)
        ak = a / ((it + 1) ** 0.602)

        loss_p, _, _ = total_loss_for_initial(cp.tensordot(coeff + ck * delta, modes, axes=1), psiT_target, E_target, T_pde, curvature_weight=0.0)
        loss_m, _, _ = total_loss_for_initial(cp.tensordot(coeff - ck * delta, modes, axes=1), psiT_target, E_target, T_pde, curvature_weight=0.0)
        grad = ((loss_p - loss_m) / (2.0 * ck)) * delta

        vel = 0.9 * vel + 0.1 * grad
        coeff = cp.clip(coeff - ak * vel, -100.0, 100.0)

        if it % 25 == 0:
            print(f"[EigenBundle] iter={it:4d} L={loss:.6f} L_main={L_main:.6f}")

    psi0_best = cp.clip(cp.tensordot(best_coeff, modes, axes=1), -10.0, 10.0)
    return psi0_best, best_loss


def _nls_step(phi, dt=0.05):
    phi = cp.asarray(phi, dtype=cp.complex128)
    lap = laplacian(phi.real) + 1j * laplacian(phi.imag)
    phi_lin = phi + 1j * dt * lap
    amp2 = cp.abs(phi_lin) ** 2
    phi_nl = phi_lin * cp.exp(-1j * dt * amp2)
    return cp.nan_to_num(phi_nl, nan=0.0, posinf=1e6, neginf=-1e6)


def nls_attack(psiT_target, E_target, T_pde, n, steps=70):
    cp.random.seed(888)
    phi0 = (cp.random.standard_normal((n, n)) + 1j * cp.random.standard_normal((n, n))) * 0.1
    vel = cp.zeros_like(phi0)
    best_loss = float("inf")
    best_phi0 = phi0.copy()

    for it in progress_iter(range(int(steps)), "NLS"):
        phi = phi0
        for _ in range(max(1, T_pde // 4)):
            phi = _nls_step(phi, dt=0.05)

        psiT_guess = cp.clip(phi.real.astype(cp.float64), -10.0, 10.0)
        L_main = float(mse(psiT_guess, psiT_target).get())
        E_guess = curvature_energy(psiT_guess)
        loss = L_main + 1e-3 * safe_curvature_loss(E_guess, E_target)

        if loss < best_loss and np.isfinite(loss):
            best_loss = loss
            best_phi0 = phi0.copy()

        noise = (cp.random.standard_normal(phi0.shape) + 1j * cp.random.standard_normal(phi0.shape))
        grad_est = noise * float(loss)
        vel = 0.9 * vel + 0.1 * grad_est
        phi0 = cp.nan_to_num(phi0 - 0.01 * vel, nan=0.0, posinf=1e6, neginf=-1e6)

        if it % 20 == 0:
            print(f"[NLS] iter={it:4d} L={loss:.6f} L_main={L_main:.6f} E={E_guess:.6f}")

    return cp.clip(best_phi0.real.astype(cp.float64), -10.0, 10.0), best_loss


def topological_mapper_attack(psiT_target, E_target, T_pde, n, steps=100):
    cp.random.seed(999)
    psi0 = cp.random.standard_normal((n, n), dtype=cp.float64) * 0.5

    def loss_fn(field):
        loss, _, _ = total_loss_for_initial(field, psiT_target, E_target, T_pde)
        return loss

    current_loss = loss_fn(psi0)
    best_loss = current_loss
    best_field = psi0.copy()
    rng = np.random.default_rng(999)

    yy, xx = cp.meshgrid(cp.arange(n), cp.arange(n), indexing="ij")

    for it in progress_iter(range(int(steps)), "TopoMap"):
        new = psi0.copy()
        for _ in range(int(rng.integers(1, 4))):
            cx = int(rng.integers(0, n))
            cy = int(rng.integers(0, n))
            radius = int(rng.integers(1, max(2, n // 4)))
            mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius
            new = cp.where(mask, -new, new)

        new = cp.clip(new, -10.0, 10.0)
        proposal_loss = loss_fn(new)

        if proposal_loss < current_loss:
            psi0 = new
            current_loss = proposal_loss
            if current_loss < best_loss:
                best_loss = current_loss
                best_field = psi0.copy()

        if it % 20 == 0:
            print(f"[TopoMap] iter={it:4d} current={current_loss:.6f} best={best_loss:.6f}")

    return best_field, best_loss


def phase_space_attack(psiT_target, E_target, T_pde, n, steps=80):
    cp.random.seed(31415)
    psi0 = cp.random.standard_normal((n, n), dtype=cp.float64) * 0.1
    p0 = cp.zeros_like(psi0)
    v_psi = cp.zeros_like(psi0)
    v_p = cp.zeros_like(p0)
    dt = 0.1
    best_loss = float("inf")
    best_psi0 = psi0.copy()

    for it in progress_iter(range(int(steps)), "PhaseSpace"):
        psi = psi0
        p = p0
        for _ in range(8):
            H_grad = laplacian(psi)
            psi = psi + dt * p
            p = p - dt * H_grad
            psi = cp.clip(cp.nan_to_num(psi, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)
            p = cp.clip(cp.nan_to_num(p, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)

        psiT_guess = forward_pde(psi, max(1, T_pde // 4))
        L_main = float(mse(psiT_guess, psiT_target).get())
        E_guess = curvature_energy(psiT_guess)
        loss = L_main + 1e-3 * safe_curvature_loss(E_guess, E_target)

        if loss < best_loss and np.isfinite(loss):
            best_loss = loss
            best_psi0 = psi0.copy()

        noise_psi = cp.random.standard_normal(psi0.shape)
        noise_p = cp.random.standard_normal(p0.shape)
        v_psi = 0.9 * v_psi + 0.1 * noise_psi * float(loss)
        v_p = 0.9 * v_p + 0.1 * noise_p * float(loss)

        psi0 = cp.clip(cp.nan_to_num(psi0 - 0.02 * v_psi, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)
        p0 = cp.clip(cp.nan_to_num(p0 - 0.02 * v_p, nan=0.0, posinf=1e6, neginf=-1e6), -1e3, 1e3)

        if it % 20 == 0:
            print(f"[PhaseSpace] iter={it:4d} L={loss:.6f} L_main={L_main:.6f} E={E_guess:.6f}")

    return best_psi0, best_loss


def run_all_attacks(cfg):
    seed = int(cfg["seed"])
    n = int(cfg["n"])
    T_pde = int(cfg["T_pde"])

    print("\n====================================")
    print("   WAVELOCK WCT SURROGATE ATTACKS   ")
    print("====================================")
    print(f"[INFO] Profile: {PROFILE}")
    print(f"[INFO] Config: {json.dumps(cfg, indent=2, sort_keys=True)}\n")

    t0 = time.time()
    psi0_true, psiT_target, E_target = make_forward_problem(seed, n, T_pde)

    attacks = [
        timed_attack("CurvatureResonance", curvature_resonance_attack, psiT_target, E_target, T_pde, n, steps=cfg["curvres_steps"], K=cfg["curvres_K"]),
        timed_attack("MultiResEnvelope", multires_attack, psiT_target, E_target, T_pde, n, steps=cfg["multires_steps"], factor=cfg["multires_factor"]),
        timed_attack("EigenmodeBundle", eigenmode_bundle_attack, psiT_target, E_target, T_pde, n, steps=cfg["eigen_steps"], K=cfg["eigen_K"]),
        timed_attack("NLSsurrogate", nls_attack, psiT_target, E_target, T_pde, n, steps=cfg["nls_steps"]),
        timed_attack("TopologicalMapper", topological_mapper_attack, psiT_target, E_target, T_pde, n, steps=cfg["topo_steps"]),
        timed_attack("PhaseSpaceAdjoint", phase_space_attack, psiT_target, E_target, T_pde, n, steps=cfg["phase_steps"]),
    ]

    elapsed = time.time() - t0
    print("\n=== ATTACK SUMMARY ===")

    summaries = []
    best_name = None
    best_loss = float("inf")
    best_corr = 0.0
    best_abs_corr = 0.0
    crash_count = 0

    for item in attacks:
        name = item["name"]
        loss = float(item.get("loss", float("inf")))
        crashed = bool(item.get("crashed", False))
        crash_count += int(crashed)

        if item.get("psi0_est") is not None:
            c = corr_coeff(item["psi0_est"], psi0_true)
        else:
            c = 0.0

        abs_c = abs(c)

        print(f"{name:22s} loss={loss:12.6g} corr(psi0_est, psi0_true)={c:+.6f}")

        if loss < best_loss:
            best_loss = loss
            best_name = name
            best_corr = c
            best_abs_corr = abs_c

        summaries.append({
            "name": name,
            "loss": loss,
            "corr": float(c),
            "abs_corr": float(abs_c),
            "runtime_seconds": float(item.get("runtime_seconds", 0.0)),
            "crashed": crashed,
            "error": item.get("error"),
        })

    learned_inverse = (best_abs_corr >= CORR_DANGER_THRESHOLD) or (best_loss <= LOSS_DANGER_THRESHOLD)
    danger_total = int(learned_inverse) + int(crash_count)

    metrics = {
        "test": "attacks_surrogate",
        "status": "danger_detected" if danger_total else "pass",
        "profile": PROFILE,
        "matched": bool(learned_inverse),
        "learned_inverse": bool(learned_inverse),
        "collisions": 0,
        "forgeries": 0,
        "false_accepts": 0,
        "accepted": 0,
        "nan_detected": False,
        "crashes": int(crash_count),
        "danger_total": int(danger_total),
        "best_attack": best_name,
        "best_loss": float(best_loss),
        "best_corr": float(best_corr),
        "best_abs_corr": float(best_abs_corr),
        "corr_danger_threshold": float(CORR_DANGER_THRESHOLD),
        "loss_danger_threshold": float(LOSS_DANGER_THRESHOLD),
        "elapsed_seconds": float(elapsed),
        "n": int(n),
        "T_pde": int(T_pde),
        "attacks": summaries,
    }

    print("\n=== FINAL RESULT ===")
    print(f"Best attack       : {best_name}")
    print(f"Best loss         : {best_loss:.6g}")
    print(f"Best correlation  : {best_corr:+.6f}")
    print(f"Learned inverse   : {learned_inverse}")
    print(f"Danger total      : {danger_total}")
    print("====================================\n")

    print("RISK_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("RISK_METRICS_END")

    return danger_total == 0


def main():
    if PROFILE not in PROFILE_CONFIG:
        print(f"[ERROR] Unknown PROFILE={PROFILE!r}. Valid profiles: {sorted(PROFILE_CONFIG)}")
        return False

    return run_all_attacks(PROFILE_CONFIG[PROFILE])


if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print("[FATAL] Surrogate attack benchmark crashed before metrics completion:", repr(e))
        traceback.print_exc()
        sys.exit(1)
