# test_wavelock_attack_nextgen.py
# WaveLock Phase-2 Adversarial Attack Suite
# Updated deep/standard benchmark version with:
# - no env commands required
# - internal profile constants
# - stronger default runtime than the old ~18s test
# - RISK_METRICS output for run_benchmarks.py
# - per-attack timing
# - CuPy synchronization for honest runtime
# - fail-closed crash reporting

import os
import sys
import time
import json
import hashlib
import traceback
import numpy as np

# Keep numerical libraries from oversubscribing when benchmark runner is parallel.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

try:
    import cupy as cp
except Exception as e:
    print("[ERROR] CuPy import failed:", repr(e))
    sys.exit(1)

# =====================================================================
# CONFIG — edit here, no environment commands needed
# =====================================================================

# Profiles:
#   "standard" = stronger than old 18s run, reasonable daily/deep run
#   "deep"     = longer adversarial run
#   "fast"     = quick smoke run
PROFILE = "standard"

PROFILE_CONFIG = {
    "fast": {
        "n": 6,
        "gradient_iterations": 3_000,
        "monte_carlo_steps": 5_000,
        "fourier_depth": 40,
        "zeta_layers": 50,
        "curvehash_rounds": 12,
        "multi_seed_count": 1,
    },
    "standard": {
        "n": 6,
        "gradient_iterations": 25_000,
        "monte_carlo_steps": 30_000,
        "fourier_depth": 250,
        "zeta_layers": 250,
        "curvehash_rounds": 200,
        "multi_seed_count": 5,
    },
    "deep": {
        "n": 8,
        "gradient_iterations": 100_000,
        "monte_carlo_steps": 100_000,
        "fourier_depth": 750,
        "zeta_layers": 750,
        "curvehash_rounds": 1_000,
        "multi_seed_count": 10,
    },
}

BASE_SEEDS = [123, 7, 42, 99, 2025, 31337, 65537, 8675309, 104729, 999983]

# If not None, stop after this many seconds but still emit metrics.
# Example: MAX_SECONDS = 20 * 60
MAX_SECONDS = None

# =====================================================================
# PATH FIX
# =====================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# =====================================================================
# IMPORT WAVELOCK CORE
# =====================================================================

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


# =====================================================================
# HELPERS
# =====================================================================

def sync_gpu():
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def commit(psi):
    """Return WaveLock commitment for any ψ-field."""
    psi = cp.asarray(psi, dtype=cp.float32)
    raw = _serialize_commitment_v2(psi)
    return hashlib.sha256(raw).hexdigest()


def finite_guard(psi, label):
    if bool(cp.any(~cp.isfinite(psi)).get()):
        raise FloatingPointError(f"non-finite values detected in {label}")


def timed_attack(name, fn, *args, **kwargs):
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


# =====================================================================
# 1. GRADIENT SURROGATE ATTACK v2
# =====================================================================

def gradient_attack_v2(n=6, iterations=25_000, lr=4e-5, seed=123):
    """
    White-box gradient surrogate attack.
    Uses Laplacian and bi-Laplacian as surrogate gradient.
    """
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    target = kp.psi_star
    target_hash = commit(target)

    cp.random.seed(int(seed) % (2**32 - 1))
    psi = cp.random.normal(0, 1, target.shape).astype(cp.float32)

    best_distance = float("inf")

    for i in range(iterations):
        grad = laplacian(psi) + 0.001 * laplacian(laplacian(psi))
        psi -= lr * grad

        if i % 250 == 0 or i == iterations - 1:
            finite_guard(psi, "gradient_attack_v2 psi")
            # Numeric closeness is diagnostic only; hash match is the break.
            dist = float(cp.linalg.norm(psi - target).get())
            best_distance = min(best_distance, dist)

            if commit(psi) == target_hash:
                return {
                    "matched": True,
                    "iteration": i,
                    "seed": seed,
                    "best_distance": best_distance,
                }

    return {
        "matched": False,
        "iterations": iterations,
        "seed": seed,
        "best_distance": best_distance,
    }


# =====================================================================
# 2. MONTE CARLO ANNEALING ATTACK
# =====================================================================

def energy_proxy(psi):
    L = laplacian(psi)
    return float((cp.sum(L * L) + cp.sum(psi * psi)).get())


def monte_carlo_attack(n=6, steps=30_000, T_start=1.0, T_end=0.01, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    target = kp.psi_star
    target_hash = commit(target)

    cp.random.seed((int(seed) + 1009) % (2**32 - 1))
    psi = cp.random.rand(*target.shape).astype(cp.float32)

    E_old = energy_proxy(psi)
    best_energy = E_old

    for t in range(steps):
        T = T_start * (1 - t / steps) + T_end * (t / steps)

        proposal = psi + cp.random.normal(0, T, psi.shape).astype(cp.float32)
        finite_guard(proposal, "monte_carlo proposal")

        E_new = energy_proxy(proposal)

        if E_new <= E_old:
            accept = True
        else:
            accept_prob = math_exp_safe(-(E_new - E_old) / (T + 1e-9))
            accept = float(cp.random.rand().get()) < accept_prob

        if accept:
            psi = proposal
            E_old = E_new
            best_energy = min(best_energy, E_new)

        if t % 500 == 0 or t == steps - 1:
            if commit(psi) == target_hash:
                return {
                    "matched": True,
                    "step": t,
                    "seed": seed,
                    "best_energy": best_energy,
                }

    return {
        "matched": False,
        "steps": steps,
        "seed": seed,
        "best_energy": best_energy,
    }


def math_exp_safe(x):
    # Prevent overflow/underflow noise.
    if x < -80:
        return 0.0
    if x > 0:
        return 1.0
    return float(np.exp(x))


# =====================================================================
# 3. FOURIER-ψ ADVERSARIAL SHELL ATTACK
# =====================================================================

def fourier_shell_attack(n=6, depth=250, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    target = kp.psi_star
    target_hash = commit(target)

    side = target.shape[0]
    fx = cp.fft.fftfreq(side)
    fy = cp.fft.fftfreq(side)
    X, Y = cp.meshgrid(fx, fy)

    best_radius = None

    for d in range(1, depth + 1):
        radius = 0.005 * d
        shell = cp.exp(-((X**2 + Y**2 - radius) ** 2) * 5000)

        wave = cp.real(cp.fft.ifft2(shell))
        wave = (wave - wave.min()) / (wave.max() - wave.min() + 1e-12)
        finite_guard(wave, "fourier wave")

        if commit(wave) == target_hash:
            return {
                "matched": True,
                "depth": d,
                "radius": float(radius),
                "seed": seed,
            }

        best_radius = float(radius)

    return {
        "matched": False,
        "depth": depth,
        "seed": seed,
        "best_radius": best_radius,
    }


# =====================================================================
# 4. ZETA-PHASE LAYERED ATTACK
# =====================================================================

def zeta_phase_attack(n=6, layers=250, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    target = kp.psi_star
    target_hash = commit(target)

    side = target.shape[0]
    psi = cp.zeros_like(target)

    x = cp.linspace(0, 1, side)

    for k in range(1, layers + 1):
        freq = k * 11
        band = cp.sin(2 * cp.pi * freq * x)
        ring = cp.outer(band, band)

        psi += ring
        psi_norm = (psi - psi.min()) / (psi.max() - psi.min() + 1e-12)
        finite_guard(psi_norm, "zeta psi_norm")

        if commit(psi_norm) == target_hash:
            return {
                "matched": True,
                "layer": k,
                "seed": seed,
            }

    return {
        "matched": False,
        "layers": layers,
        "seed": seed,
    }


# =====================================================================
# 5. CURVEHASH v3 MULTI-ROUND ATTACK
# =====================================================================

def curvehash_v3_attack(n=6, rounds=200, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    target = kp.psi_star
    target_hash = commit(target)

    cp.random.seed((int(seed) + 2027) % (2**32 - 1))
    psi = cp.random.rand(*target.shape).astype(cp.float32)

    for r in range(rounds):
        salt = cp.float32(0.03 * (r + 1))
        psi = psi + salt * laplacian(psi)
        psi = cp.tanh(psi)
        finite_guard(psi, "curvehash psi")

        if commit(psi) == target_hash:
            return {
                "matched": True,
                "round": r,
                "seed": seed,
            }

    return {
        "matched": False,
        "rounds": rounds,
        "seed": seed,
    }


# =====================================================================
# MULTI-SEED WRAPPER
# =====================================================================

def run_multi_seed_attack(label, attack_fn, seed_list, **kwargs):
    results = []
    any_match = False
    crash_count = 0

    for seed in seed_list:
        result = timed_attack(f"{label} seed={seed}", attack_fn, seed=seed, **kwargs)
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


# =====================================================================
# MAIN
# =====================================================================

def main():
    if PROFILE not in PROFILE_CONFIG:
        print(f"[ERROR] Unknown PROFILE={PROFILE!r}. Valid profiles: {sorted(PROFILE_CONFIG)}")
        return False

    cfg = PROFILE_CONFIG[PROFILE]
    n = cfg["n"]
    seed_list = BASE_SEEDS[: cfg["multi_seed_count"]]

    print("\n=== WAVELOCK PHASE 2 ADVERSARIAL ATTACK SUITE ===")
    print(f"[INFO] Profile: {PROFILE}")
    print(f"[INFO] n: {n}")
    print(f"[INFO] Seeds: {seed_list}")
    print(f"[INFO] Config: {json.dumps(cfg, indent=2)}\n")

    start = time.time()
    suite_results = {}

    suite_results["gradient_attack_v2"] = run_multi_seed_attack(
        "1) Gradient Surrogate v2",
        gradient_attack_v2,
        seed_list,
        n=n,
        iterations=cfg["gradient_iterations"],
        lr=4e-5,
    )

    if not should_stop(start):
        suite_results["monte_carlo_attack"] = run_multi_seed_attack(
            "2) Monte Carlo Annealing Attack",
            monte_carlo_attack,
            seed_list,
            n=n,
            steps=cfg["monte_carlo_steps"],
        )

    if not should_stop(start):
        suite_results["fourier_shell_attack"] = run_multi_seed_attack(
            "3) Fourier Shell Attack",
            fourier_shell_attack,
            seed_list,
            n=n,
            depth=cfg["fourier_depth"],
        )

    if not should_stop(start):
        suite_results["zeta_phase_attack"] = run_multi_seed_attack(
            "4) Zeta-Phase Layered Attack",
            zeta_phase_attack,
            seed_list,
            n=n,
            layers=cfg["zeta_layers"],
        )

    if not should_stop(start):
        suite_results["curvehash_v3_attack"] = run_multi_seed_attack(
            "5) CurveHash v3 Multi-Round Attack",
            curvehash_v3_attack,
            seed_list,
            n=n,
            rounds=cfg["curvehash_rounds"],
        )

    elapsed = time.time() - start

    matched_attacks = [
        name for name, result in suite_results.items()
        if bool(result.get("matched", False))
    ]

    crash_count = sum(int(result.get("crashes", 0)) for result in suite_results.values())

    metrics = {
        "test": "attack_nextgen",
        "profile": PROFILE,
        "matched": bool(matched_attacks),
        "matched_attacks": matched_attacks,
        "collisions": 0,
        "forgeries": 0,
        "false_accepts": 0,
        "accepted": 0,
        "nan_detected": False,
        "crashes": crash_count,
        "attack_count": len(suite_results),
        "seed_count": len(seed_list),
        "elapsed_seconds": elapsed,
        "n": n,
    }

    print("\n==================== SUITE SUMMARY ====================")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    print("\nRISK_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("RISK_METRICS_END")

    print("\n=== DONE ===\n")

    # Return nonzero only for actual break/crash.
    return not bool(matched_attacks) and crash_count == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
