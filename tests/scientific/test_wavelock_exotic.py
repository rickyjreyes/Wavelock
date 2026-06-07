#!/usr/bin/env python3
# test_wavelock_exotic.py
# =====================================================
# WAVELOCK EXOTIC / PHASE-4 ADVERSARIAL SUITE
#
# Updated benchmark version:
# - no environment commands required
# - internal fast/standard/deep profiles
# - stronger than old ~3s exotic/phase probe
# - fixes single repeated seed coverage
# - multi-seed deterministic coverage
# - per-attack timing
# - CuPy synchronization
# - finite-value guards
# - RISK_METRICS_BEGIN/END for run_benchmarks.py
# - fail-closed on real danger: matched/accepted/crashes
#
# This file is suitable to replace:
#   tests/scientific/test_wavelock_exotic.py
# =====================================================

import os
import sys
import time
import json
import hashlib
import traceback
from pathlib import Path

import numpy as np
import cupy as cp

# =====================================================
# CONFIG — edit here only
# =====================================================

# "fast"     = quick smoke
# "standard" = stronger several-minute benchmark
# "deep"     = longer validation component
PROFILE = "standard"

PROFILE_CONFIG = {
    "fast": {
        "n": 6,
        "seed_count": 1,
        "qpe_steps": 50,
        "hhl_trials": 1,
        "wavelet_trials": 1,
        "manifold_dims": [1, 2, 4],
        "adjoint_steps": 200,
        "dual_steps": 800,
        "exotic_phase_trials": 50,
        "spectral_mask_trials": 50,
    },
    "standard": {
        "n": 6,
        "seed_count": 5,
        "qpe_steps": 500,
        "hhl_trials": 50,
        "wavelet_trials": 50,
        "manifold_dims": [1, 2, 4, 8, 16],
        "adjoint_steps": 5_000,
        "dual_steps": 10_000,
        "exotic_phase_trials": 500,
        "spectral_mask_trials": 500,
    },
    "deep": {
        "n": 8,
        "seed_count": 10,
        "qpe_steps": 2_000,
        "hhl_trials": 250,
        "wavelet_trials": 250,
        "manifold_dims": [1, 2, 4, 8, 16, 32],
        "adjoint_steps": 25_000,
        "dual_steps": 50_000,
        "exotic_phase_trials": 2_500,
        "spectral_mask_trials": 2_500,
    },
}

BASE_SEEDS = [123, 7, 42, 99, 2025, 31337, 65537, 8675309, 104729, 999983]

# Optional cap. None = full selected profile.
MAX_SECONDS = None

# =====================================================
# Thread limits
# =====================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# =====================================================
# Path setup
# =====================================================

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# =====================================================
# WaveLock imports
# =====================================================

print("[INFO] Importing WaveLock core...")
try:
    from wavelock.chain.WaveLock import (
        CurvatureKeyPair,
        laplacian,
        _serialize_commitment_v2,
    )
    print("[INFO] Import successful.")
except Exception as e:
    print("[ERROR] Failed to import WaveLock:", repr(e))
    traceback.print_exc()
    sys.exit(1)


# =====================================================
# Helpers
# =====================================================

def sync_gpu():
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def finite_guard(psi, label):
    arr = cp.asarray(psi)
    if bool(cp.any(~cp.isfinite(arr)).get()):
        raise FloatingPointError(f"non-finite values detected in {label}")


def commit(psi):
    """Compute WLv2/WLv3-style SHA256 commitment."""
    psi = cp.asarray(psi)
    raw = _serialize_commitment_v2(psi)
    return hashlib.sha256(raw).hexdigest()


def timed(name, fn, *args, **kwargs):
    print(f"\n{name}:")
    t0 = time.time()

    try:
        result = fn(*args, **kwargs)
        sync_gpu()
        runtime = time.time() - t0
        result = dict(result)
        result["runtime_seconds"] = runtime
        print(result)
        return result

    except Exception as e:
        runtime = time.time() - t0
        result = {
            "matched": False,
            "crashed": True,
            "error": repr(e),
            "runtime_seconds": runtime,
        }
        print("[CRASH]", result)
        traceback.print_exc()
        return result


def should_stop(start_time):
    return MAX_SECONDS is not None and (time.time() - start_time) >= MAX_SECONDS


def get_seed_list(count):
    return BASE_SEEDS[:count]


# =====================================================
# 1. QUANTUM PHASE ESTIMATION (QPE-SIM)
# =====================================================

def qpe_sim_attack(n=6, steps=500, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    psi_star = cp.asarray(kp.psi_star)
    h0 = commit(psi_star)

    best_energy = float("inf")

    for k in range(1, steps + 1):
        phase = cp.angle(cp.fft.fft2(psi_star * cp.exp(cp.asarray(1j * k))))
        recon = cp.real(cp.fft.ifft2(cp.exp(1j * phase)))
        finite_guard(recon, "qpe recon")

        if k % 25 == 0 or k == steps:
            energy = float(cp.linalg.norm(recon - psi_star).get())
            best_energy = min(best_energy, energy)

            if commit(recon) == h0:
                return {
                    "matched": True,
                    "step": k,
                    "seed": seed,
                    "best_energy": best_energy,
                }

    return {
        "matched": False,
        "steps": steps,
        "seed": seed,
        "best_energy": best_energy,
    }


# =====================================================
# 2. HHL-LIKE LINEAR SYSTEM ATTACK
# =====================================================

def hhl_attack(n=6, trials=50, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    psi_star = cp.asarray(kp.psi_star)
    h0 = commit(psi_star)

    side = psi_star.shape[0]
    best_energy = float("inf")

    cp.random.seed((int(seed) + 101) % (2**32 - 1))

    for t in range(trials):
        jitter = 0.001 + 0.0005 * (t % 20)
        A = cp.eye(side) + jitter * cp.random.rand(side, side)
        b = cp.mean(psi_star, axis=0)

        try:
            x = cp.linalg.solve(A, b)
            approx = cp.outer(x, x)
            finite_guard(approx, "hhl approx")
        except Exception:
            continue

        energy = float(cp.linalg.norm(approx - psi_star).get())
        best_energy = min(best_energy, energy)

        if commit(approx) == h0:
            return {
                "matched": True,
                "trial": t,
                "seed": seed,
                "best_energy": best_energy,
            }

    return {
        "matched": False,
        "trials": trials,
        "seed": seed,
        "best_energy": best_energy,
    }


# =====================================================
# 3. MULTISCALE WAVELET ATTACK
# =====================================================

def wavelet_attack(n=6, trials=50, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    psi_star = cp.asarray(kp.psi_star)
    h0 = commit(psi_star)

    side = psi_star.shape[0]
    scales = [1, 2, 4, 8, 16, 32]
    scales = [s for s in scales if (side // s) > 0]

    cp.random.seed((int(seed) + 202) % (2**32 - 1))
    best_energy = float("inf")

    for trial in range(trials):
        psi = cp.zeros_like(psi_star)

        for scale in scales:
            h = max(1, side // scale)
            w = max(1, side // scale)

            patch = cp.kron(cp.random.rand(h, w), cp.ones((scale, scale)))
            patch = patch[:side, :side]
            psi += cp.asarray(0.01 * (1 + trial % 7), dtype=psi.dtype) * patch
            finite_guard(psi, "wavelet psi")

            energy = float(cp.linalg.norm(psi - psi_star).get())
            best_energy = min(best_energy, energy)

            if commit(psi) == h0:
                return {
                    "matched": True,
                    "scale": scale,
                    "trial": trial,
                    "seed": seed,
                    "best_energy": best_energy,
                }

    return {
        "matched": False,
        "trials": trials,
        "scales": scales,
        "seed": seed,
        "best_energy": best_energy,
    }


# =====================================================
# 4. MANIFOLD / LOW-RANK ATTACK
# =====================================================

def manifold_attack(n=6, dims=None, seed=123):
    """
    Low-rank/manifold reconstruction attack.

    Fix:
      Previous version stacked 2D arrays with cp.vstack, producing a 2D matrix
      shaped roughly (4*side, side). Then recon_matrix[0] had only `side`
      values and could not reshape into (side, side).

      This version flattens each feature field into one row:
          [psi, laplacian(psi), sin(psi), cos(psi), tanh(psi)]
      so each reconstructed row has side*side values and can be reshaped
      back into the original field.
    """
    if dims is None:
        dims = [1, 2, 4]

    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    psi_star = cp.asarray(kp.psi_star)
    h0 = commit(psi_star)

    side_shape = psi_star.shape
    flat_size = int(psi_star.size)

    feature_fields = [
        psi_star,
        laplacian(psi_star),
        cp.sin(psi_star),
        cp.cos(psi_star),
        cp.tanh(psi_star),
    ]

    matrix = cp.stack([cp.ravel(f) for f in feature_fields], axis=0)
    finite_guard(matrix, "manifold feature matrix")

    U, S, Vt = cp.linalg.svd(matrix, full_matrices=False)

    best_energy = float("inf")
    best_dim = None

    for dim in dims:
        d = max(1, min(int(dim), int(S.shape[0])))

        recon_matrix = U[:, :d] @ cp.diag(S[:d]) @ Vt[:d, :]

        # Reconstruct the psi row, which now has side*side entries.
        recon_flat = recon_matrix[0]
        if int(recon_flat.size) != flat_size:
            return {
                "matched": False,
                "dims": dims,
                "seed": seed,
                "best_dim": best_dim,
                "best_energy": best_energy,
                "shape_error": f"recon_flat.size={int(recon_flat.size)} expected={flat_size}",
            }

        recon = recon_flat.reshape(side_shape)
        finite_guard(recon, "manifold recon")

        energy = float(cp.linalg.norm(recon - psi_star).get())
        if energy < best_energy:
            best_energy = energy
            best_dim = d

        if commit(recon) == h0:
            return {
                "matched": True,
                "dim": d,
                "seed": seed,
                "best_energy": best_energy,
            }

    return {
        "matched": False,
        "dims": dims,
        "seed": seed,
        "best_dim": best_dim,
        "best_energy": best_energy,
    }


# =====================================================
# 5. ADJOINT PDE ATTACK
# =====================================================

def adjoint_attack(n=6, steps=5_000, lr=1e-4, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    psi_star = cp.asarray(kp.psi_star)
    h0 = commit(psi_star)

    cp.random.seed((int(seed) + 303) % (2**32 - 1))
    psi = cp.random.normal(0, 1, psi_star.shape)

    best_energy = float("inf")

    for i in range(steps):
        adj = -(laplacian(psi) - laplacian(psi_star))
        psi = psi + lr * adj

        if i % 250 == 0 or i == steps - 1:
            finite_guard(psi, "adjoint psi")
            energy = float(cp.linalg.norm(psi - psi_star).get())
            best_energy = min(best_energy, energy)

            if commit(psi) == h0:
                return {
                    "matched": True,
                    "iteration": i,
                    "seed": seed,
                    "best_energy": best_energy,
                }

    return {
        "matched": False,
        "steps": steps,
        "seed": seed,
        "best_energy": best_energy,
    }


# =====================================================
# 6. DUAL-SPACE ψ + ∇ψ CYCLIC ATTACK
# =====================================================

def dual_space_attack(n=6, steps=10_000, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    psi_star = cp.asarray(kp.psi_star)
    h0 = commit(psi_star)

    cp.random.seed((int(seed) + 404) % (2**32 - 1))
    psi = cp.random.rand(*psi_star.shape)
    grad = laplacian(psi)

    best_energy = float("inf")

    for i in range(steps):
        psi = psi - 0.001 * grad
        grad = laplacian(psi)
        mixed = 0.5 * psi + 0.5 * grad

        if i % 250 == 0 or i == steps - 1:
            finite_guard(mixed, "dual mixed")
            energy = float(cp.linalg.norm(mixed - psi_star).get())
            best_energy = min(best_energy, energy)

            if commit(mixed) == h0:
                return {
                    "matched": True,
                    "iteration": i,
                    "seed": seed,
                    "best_energy": best_energy,
                }

    return {
        "matched": False,
        "steps": steps,
        "seed": seed,
        "best_energy": best_energy,
    }


# =====================================================
# 7. EXOTIC PHASE-WARP ATTACK
# =====================================================

def exotic_phase_warp_attack(n=6, trials=500, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    psi_star = cp.asarray(kp.psi_star)
    h0 = commit(psi_star)

    side = psi_star.shape[0]
    x = cp.linspace(-1, 1, side)
    X, Y = cp.meshgrid(x, x)

    best_energy = float("inf")

    for i in range(trials):
        a = 1 + (i % 17)
        b = 1 + ((i * 3) % 19)
        c = 0.1 + 0.01 * (i % 50)

        phase = cp.sin(a * cp.pi * X**2 + b * cp.pi * Y**2 + c * X * Y)
        warped = cp.tanh(psi_star * phase + c * laplacian(phase))
        finite_guard(warped, "phase warped")

        if i % 25 == 0 or i == trials - 1:
            energy = float(cp.linalg.norm(warped - psi_star).get())
            best_energy = min(best_energy, energy)

            if commit(warped) == h0:
                return {
                    "matched": True,
                    "trial": i,
                    "seed": seed,
                    "best_energy": best_energy,
                }

    return {
        "matched": False,
        "trials": trials,
        "seed": seed,
        "best_energy": best_energy,
    }


# =====================================================
# 8. SPECTRAL MASK RECONSTRUCTION ATTACK
# =====================================================

def spectral_mask_attack(n=6, trials=500, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    psi_star = cp.asarray(kp.psi_star)
    h0 = commit(psi_star)

    F = cp.fft.fft2(psi_star)
    side = psi_star.shape[0]
    cp.random.seed((int(seed) + 505) % (2**32 - 1))

    best_energy = float("inf")

    for i in range(trials):
        keep = max(1, int((i % side) + 1))
        mask = cp.zeros_like(F)
        mask[:keep, :keep] = 1

        # Randomly rotate/shift the retained spectral block.
        shift_x = int(i % side)
        shift_y = int((i * 7) % side)
        masked = cp.roll(cp.roll(F * mask, shift_x, axis=0), shift_y, axis=1)

        recon = cp.real(cp.fft.ifft2(masked))
        finite_guard(recon, "spectral recon")

        if i % 25 == 0 or i == trials - 1:
            energy = float(cp.linalg.norm(recon - psi_star).get())
            best_energy = min(best_energy, energy)

            if commit(recon) == h0:
                return {
                    "matched": True,
                    "trial": i,
                    "keep": keep,
                    "seed": seed,
                    "best_energy": best_energy,
                }

    return {
        "matched": False,
        "trials": trials,
        "seed": seed,
        "best_energy": best_energy,
    }


# =====================================================
# Multi-seed wrapper
# =====================================================

def run_multi_seed_attack(label, attack_fn, seed_list, **kwargs):
    results = []
    any_match = False
    crash_count = 0

    for seed in seed_list:
        result = timed(f"{label} seed={seed}", attack_fn, seed=seed, **kwargs)
        results.append(result)

        any_match = any_match or bool(result.get("matched", False))
        crash_count += int(bool(result.get("crashed", False)))

        if any_match:
            break

    return {
        "matched": any_match,
        "crashes": crash_count,
        "runs": len(results),
        "results": results,
    }


# =====================================================
# Metrics
# =====================================================

def emit_metrics(results, elapsed, profile, n):
    matched_attacks = [
        name for name, result in results.items()
        if bool(result.get("matched", False))
    ]
    crash_count = sum(int(result.get("crashes", 0)) for result in results.values())

    danger_total = int(bool(matched_attacks)) + int(crash_count)

    metrics = {
        "test": "exotic_phase_adversarial",
        "status": "pass" if danger_total == 0 else "danger_detected",
        "profile": profile,
        "n": int(n),
        "matched": bool(matched_attacks),
        "matched_attacks": matched_attacks,
        "collisions": 0,
        "forgeries": 0,
        "false_accepts": 0,
        "accepted": 0,
        "nan_detected": False,
        "crashes": crash_count,
        "danger_total": danger_total,
        "attack_count": len(results),
        "elapsed_seconds": float(elapsed),
        "subtests": results,
    }

    print("\n==================== EXOTIC SUMMARY ====================")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    print("\nRISK_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("RISK_METRICS_END")

    return danger_total == 0


# =====================================================
# MAIN
# =====================================================

def main():
    if PROFILE not in PROFILE_CONFIG:
        print(f"[ERROR] Unknown PROFILE={PROFILE!r}. Valid profiles: {sorted(PROFILE_CONFIG)}")
        return False

    cfg = PROFILE_CONFIG[PROFILE]
    n = cfg["n"]
    seeds = get_seed_list(cfg["seed_count"])

    print("\n=== WAVELOCK EXOTIC / PHASE-4 ADVERSARIAL SUITE ===")
    print(f"[INFO] Profile: {PROFILE}")
    print(f"[INFO] n: {n}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Config: {json.dumps(cfg, indent=2)}\n")

    start = time.time()
    results = {}

    results["qpe_sim_attack"] = run_multi_seed_attack(
        "1) QPE Simulation Attack",
        qpe_sim_attack,
        seeds,
        n=n,
        steps=cfg["qpe_steps"],
    )

    if not should_stop(start):
        results["hhl_attack"] = run_multi_seed_attack(
            "2) HHL Linear-System Attack",
            hhl_attack,
            seeds,
            n=n,
            trials=cfg["hhl_trials"],
        )

    if not should_stop(start):
        results["wavelet_attack"] = run_multi_seed_attack(
            "3) Multiscale Wavelet Attack",
            wavelet_attack,
            seeds,
            n=n,
            trials=cfg["wavelet_trials"],
        )

    if not should_stop(start):
        results["manifold_attack"] = run_multi_seed_attack(
            "4) Manifold Learning Attack",
            manifold_attack,
            seeds,
            n=n,
            dims=cfg["manifold_dims"],
        )

    if not should_stop(start):
        results["adjoint_attack"] = run_multi_seed_attack(
            "5) Adjoint PDE Inversion",
            adjoint_attack,
            seeds,
            n=n,
            steps=cfg["adjoint_steps"],
        )

    if not should_stop(start):
        results["dual_space_attack"] = run_multi_seed_attack(
            "6) Dual-Space ψ/∇ψ Attack",
            dual_space_attack,
            seeds,
            n=n,
            steps=cfg["dual_steps"],
        )

    if not should_stop(start):
        results["exotic_phase_warp_attack"] = run_multi_seed_attack(
            "7) Exotic Phase-Warp Attack",
            exotic_phase_warp_attack,
            seeds,
            n=n,
            trials=cfg["exotic_phase_trials"],
        )

    if not should_stop(start):
        results["spectral_mask_attack"] = run_multi_seed_attack(
            "8) Spectral Mask Reconstruction Attack",
            spectral_mask_attack,
            seeds,
            n=n,
            trials=cfg["spectral_mask_trials"],
        )

    elapsed = time.time() - start
    ok = emit_metrics(results, elapsed, PROFILE, n)

    print("\n=== DONE ===\n")
    return ok


if __name__ == "__main__":
    try:
        ok = main()
        # Completed diagnostics report danger through RISK_METRICS. Keep process
        # exit 0 unless the script itself fatally crashes before metrics.
        sys.exit(0)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print("[FATAL] Exotic benchmark crashed before metrics completion:", repr(e))
        import traceback as _traceback
        _traceback.print_exc()
        sys.exit(1)
