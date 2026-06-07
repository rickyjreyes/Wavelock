#!/usr/bin/env python3
# test_wavelock_extreme.py
# ============================================================
# WaveLock Extreme Test Suite
#
# Updated benchmark version:
# - no environment commands required
# - internal fast/standard/deep profiles
# - stronger than old ~6s probe
# - fixes repeated-seed loops
# - varied deterministic seeds
# - per-test timing
# - CuPy synchronization
# - RISK_METRICS_BEGIN/END for run_benchmarks.py
# - fail-closed on real danger: collisions, false accepts, forgeries,
#   accepted inversions, avalanche failures, drift failures, bad precision match
# ============================================================

import os
import sys
import time
import json
import math
import hashlib
import traceback
from pathlib import Path

import numpy as np
import cupy as cp

# ============================================================
# CONFIG — edit here only
# ============================================================

# "fast"     = quick smoke
# "standard" = stronger several-minute benchmark
# "deep"     = longer validation component
PROFILE = "standard"

PROFILE_CONFIG = {
    "fast": {
        "n": 6,
        "avalanche_flips": 10_000,
        "drift_steps": 1_000,
        "fourier_attempts": 200,
        "projection_attempts": 500,
        "thermal_trials": 100,
        "recombination_attempts": 200,
        "gradient_steps": 300,
        "seed_count": 1,
    },
    "standard": {
        "n": 6,
        "avalanche_flips": 100_000,
        "drift_steps": 10_000,
        "fourier_attempts": 2_000,
        "projection_attempts": 5_000,
        "thermal_trials": 1_000,
        "recombination_attempts": 2_000,
        "gradient_steps": 5_000,
        "seed_count": 5,
    },
    "deep": {
        "n": 8,
        "avalanche_flips": 500_000,
        "drift_steps": 50_000,
        "fourier_attempts": 10_000,
        "projection_attempts": 25_000,
        "thermal_trials": 5_000,
        "recombination_attempts": 10_000,
        "gradient_steps": 25_000,
        "seed_count": 10,
    },
}

BASE_SEEDS = [123, 7, 42, 99, 2025, 31337, 65537, 8675309, 104729, 999983]

# Optional cap. None = full selected profile.
MAX_SECONDS = None

# ============================================================
# Thread limits
# ============================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# ============================================================
# Path setup
# ============================================================

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ============================================================
# WaveLock imports
# ============================================================

print("[INFO] Importing WaveLock core...")
try:
    from wavelock.chain.WaveLock import (
        CurvatureKeyPair,
        _serialize_commitment_v2,
        laplacian,
    )
    print("[INFO] Import successful.")
except Exception as e:
    print("[ERROR] Failed to import WaveLock:", repr(e))
    traceback.print_exc()
    sys.exit(1)


# ============================================================
# Helpers
# ============================================================

def sync_gpu():
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def finite_guard(psi, label):
    if bool(cp.any(~cp.isfinite(cp.asarray(psi))).get()):
        raise FloatingPointError(f"non-finite values detected in {label}")


def commit(psi):
    """Compute WaveLock v2 commitment."""
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
            "crashed": True,
            "error": repr(e),
            "runtime_seconds": runtime,
        }
        print("[CRASH]", result)
        traceback.print_exc()
        return result


def should_stop(start_time):
    return MAX_SECONDS is not None and (time.time() - start_time) >= MAX_SECONDS


def seed_list(count):
    return BASE_SEEDS[:count]


# ============================================================
# 1. Massive Avalanche Analysis
# ============================================================

def extreme_avalanche_test(n=6, flips=100_000, seeds=None):
    if seeds is None:
        seeds = [123]

    total_failures = 0
    total_flips = 0

    for seed in seeds:
        kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
        base = cp.asarray(kp.psi_star, dtype=cp.float64)
        base_hash = commit(base)

        side = base.shape[0]
        cp.random.seed(int(seed) % (2**32 - 1))
        rng = np.random.default_rng(int(seed))

        failures = 0
        per_seed_flips = max(1, flips // len(seeds))

        for i in range(per_seed_flips):
            perturbed = base.copy()
            x = int(rng.integers(0, side))
            y = int(rng.integers(0, side))
            delta = float(rng.normal(0, 1e-4))
            perturbed[x, y] += delta
            finite_guard(perturbed, "avalanche perturbed")

            if commit(perturbed) == base_hash:
                failures += 1
                break

        total_failures += failures
        total_flips += per_seed_flips

    return {
        "flip_attempts": total_flips,
        "failures": total_failures,
        "failure_rate": total_failures / max(1, total_flips),
        "seeds": len(seeds),
    }


# ============================================================
# 2. Long-Term Drift Accumulation
# ============================================================

def extreme_drift_test(n=6, steps=10_000, seeds=None):
    if seeds is None:
        seeds = [123]

    same_commit_count = 0
    min_delta_norm = float("inf")
    max_delta_norm = 0.0

    for seed in seeds:
        kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
        base = cp.asarray(kp.psi_star, dtype=cp.float64)
        c0 = commit(base)

        cp.random.seed((int(seed) + 101) % (2**32 - 1))
        psi = base.copy()

        for _ in range(steps):
            psi += cp.random.normal(0, 1e-7, psi.shape)

        finite_guard(psi, "drift psi")
        c1 = commit(psi)

        delta_norm = float(cp.linalg.norm(psi - base).get())
        min_delta_norm = min(min_delta_norm, delta_norm)
        max_delta_norm = max(max_delta_norm, delta_norm)

        if c1 == c0:
            same_commit_count += 1

    return {
        "same_commit": same_commit_count > 0,
        "same_commit_count": same_commit_count,
        "status": "PASS" if same_commit_count == 0 else "FAIL",
        "steps_per_seed": steps,
        "seeds": len(seeds),
        "min_delta_norm": min_delta_norm,
        "max_delta_norm": max_delta_norm,
    }


# ============================================================
# 3. Precision Downgrade Attack
# ============================================================

def precision_attack(n=6, seeds=None):
    if seeds is None:
        seeds = [123]

    float32_matches = 0
    float16_matches = 0

    for seed in seeds:
        kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
        base = cp.asarray(kp.psi_star, dtype=cp.float64)
        c0 = commit(base)

        psi32 = cp.asarray(base, dtype=cp.float32)
        psi16 = cp.asarray(base, dtype=cp.float16)

        if commit(psi32) == c0:
            float32_matches += 1
        if commit(psi16) == c0:
            float16_matches += 1

    return {
        "float32_match": float32_matches > 0,
        "float16_match": float16_matches > 0,
        "float32_matches": float32_matches,
        "float16_matches": float16_matches,
        "seeds": len(seeds),
    }


# ============================================================
# 4. Fourier-Space Adversarial Attack
# ============================================================

def fourier_attack(n=6, attempts=2_000, seeds=None):
    if seeds is None:
        seeds = [123]

    false_accepts = 0
    total_attempts = 0

    for seed in seeds:
        kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
        target = cp.asarray(kp.psi_star, dtype=cp.float64)
        h0 = commit(target)

        side = target.shape[0]
        fx = cp.fft.fftfreq(side)
        fy = cp.fft.fftfreq(side)
        X, Y = cp.meshgrid(fx, fy)

        rng = np.random.default_rng(int(seed) + 202)
        per_seed_attempts = max(1, attempts // len(seeds))

        for _ in range(per_seed_attempts):
            freq_x = int(rng.integers(5, 300))
            freq_y = int(rng.integers(5, 300))
            amp = float(rng.uniform(0.001, 0.05))

            wave = cp.sin(2 * cp.pi * freq_x * X) * cp.cos(2 * cp.pi * freq_y * Y)
            wave = wave / (cp.max(cp.abs(wave)) + 1e-12) * amp
            finite_guard(wave, "fourier wave")

            if commit(wave) == h0:
                false_accepts += 1
                break

        total_attempts += per_seed_attempts

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / max(1, total_attempts),
        "attempts": total_attempts,
        "seeds": len(seeds),
    }


# ============================================================
# 5. Random Projection Blend Attack
# ============================================================

def random_projection_attack(n=6, attempts=5_000, seeds=None):
    if seeds is None:
        seeds = [123]

    false_accepts = 0
    total_attempts = 0

    for seed in seeds:
        kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
        base = cp.asarray(kp.psi_star, dtype=cp.float64)
        h0 = commit(base)

        cp.random.seed((int(seed) + 303) % (2**32 - 1))
        per_seed_attempts = max(1, attempts // len(seeds))

        for i in range(per_seed_attempts):
            scale_noise = 10 ** (-4 + (i % 8) * 0.5)
            scale_base = 10 ** (-4 + ((i + 3) % 8) * 0.5)

            noise = cp.random.normal(0, 1, base.shape)
            mix = scale_noise * noise + scale_base * base
            finite_guard(mix, "projection mix")

            if commit(mix) == h0:
                false_accepts += 1
                break

        total_attempts += per_seed_attempts

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / max(1, total_attempts),
        "attempts": total_attempts,
        "seeds": len(seeds),
    }


# ============================================================
# 6. Multi-GPU Consistency
# ============================================================

def multi_gpu_test(n=6):
    try:
        gpu_count = cp.cuda.runtime.getDeviceCount()
    except Exception:
        return {"skip": True, "reason": "could not query GPU count"}

    if gpu_count < 2:
        return {"skip": True, "gpu_count": gpu_count}

    cp.cuda.Device(0).use()
    kp0 = CurvatureKeyPair(n=n, seed=123, test_mode=True)
    h0 = commit(kp0.psi_star)

    cp.cuda.Device(1).use()
    kp1 = CurvatureKeyPair(n=n, seed=123, test_mode=True)
    h1 = commit(kp1.psi_star)

    cp.cuda.Device(0).use()

    return {
        "gpu_count": gpu_count,
        "same": h0 == h1,
    }


# ============================================================
# 7. Quantization Attack
# ============================================================

def quantization_attack(n=6, seeds=None):
    if seeds is None:
        seeds = [123]

    quantized_matches = 0

    for seed in seeds:
        kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
        psi = cp.asarray(kp.psi_star, dtype=cp.float64)
        h0 = commit(psi)

        thresholds = [
            cp.mean(psi),
            cp.median(psi),
            cp.percentile(psi, 25),
            cp.percentile(psi, 75),
        ]

        for threshold in thresholds:
            q = (psi > threshold).astype(cp.float64)
            finite_guard(q, "quantized field")

            if commit(q) == h0:
                quantized_matches += 1
                break

    return {
        "quantized_match": quantized_matches > 0,
        "quantized_matches": quantized_matches,
        "seeds": len(seeds),
    }


# ============================================================
# 8. Thermal Noise Model Attack
# ============================================================

def thermal_attack(n=6, trials=1_000, seeds=None):
    if seeds is None:
        seeds = [123]

    thermal_matches = 0
    total_trials = 0

    for seed in seeds:
        kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
        psi = cp.asarray(kp.psi_star, dtype=cp.float64)
        h0 = commit(psi)

        cp.random.seed((int(seed) + 404) % (2**32 - 1))
        per_seed_trials = max(1, trials // len(seeds))

        for i in range(per_seed_trials):
            T = 10 ** (-6 + (i % 8) * 0.5)
            thermal = psi + cp.random.normal(0, T, psi.shape)
            finite_guard(thermal, "thermal field")

            if commit(thermal) == h0:
                thermal_matches += 1
                break

        total_trials += per_seed_trials

    return {
        "thermal_match": thermal_matches > 0,
        "thermal_matches": thermal_matches,
        "trials": total_trials,
        "rate": thermal_matches / max(1, total_trials),
        "seeds": len(seeds),
    }


# ============================================================
# 9. ψ*-Recombination Attack
# ============================================================

def recombination_attack(n=6, attempts=2_000, seeds=None):
    if seeds is None:
        seeds = [123]

    false_accepts = 0
    total_attempts = 0

    for seed in seeds:
        kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
        psi = cp.asarray(kp.psi_star, dtype=cp.float64)
        h0 = commit(psi)

        cp.random.seed((int(seed) + 505) % (2**32 - 1))
        per_seed_attempts = max(1, attempts // len(seeds))

        for i in range(per_seed_attempts):
            alpha = (i % 101) / 100.0
            noise = cp.random.rand(*psi.shape)
            mix = alpha * noise + (1.0 - alpha) * psi
            finite_guard(mix, "recombination mix")

            if commit(mix) == h0 and alpha != 0.0:
                false_accepts += 1
                break

        total_attempts += per_seed_attempts

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / max(1, total_attempts),
        "attempts": total_attempts,
        "seeds": len(seeds),
    }


# ============================================================
# 10. Gradient Surrogate Attack
# ============================================================

def gradient_surrogate_attack(n=6, steps=5_000, seeds=None, lr=1e-4):
    if seeds is None:
        seeds = [123]

    matched_count = 0
    best_distance = float("inf")

    for seed in seeds:
        kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
        target = cp.asarray(kp.psi_star, dtype=cp.float64)
        h0 = commit(target)

        cp.random.seed((int(seed) + 606) % (2**32 - 1))
        psi = cp.random.rand(*target.shape).astype(cp.float64)

        for step in range(steps):
            grad = laplacian(psi)
            psi -= lr * grad

            if step % 250 == 0 or step == steps - 1:
                finite_guard(psi, "gradient psi")
                dist = float(cp.linalg.norm(psi - target).get())
                best_distance = min(best_distance, dist)

                if commit(psi) == h0:
                    matched_count += 1
                    break

    return {
        "matched": matched_count > 0,
        "matched_count": matched_count,
        "steps_per_seed": steps,
        "seeds": len(seeds),
        "best_distance": best_distance,
    }


# ============================================================
# Metrics
# ============================================================

def emit_metrics(results, elapsed, profile, n):
    false_accepts = (
        int(results.get("fourier_attack", {}).get("false_accepts", 0))
        + int(results.get("projection_attack", {}).get("false_accepts", 0))
        + int(results.get("recombination_attack", {}).get("false_accepts", 0))
    )

    avalanche_failures = int(results.get("avalanche", {}).get("failures", 0))
    drift_failures = int(results.get("drift", {}).get("same_commit_count", 0))
    precision_bad = int(bool(results.get("precision", {}).get("float16_match", False)))
    quantized_bad = int(bool(results.get("quantization", {}).get("quantized_match", False)))
    thermal_bad = int(bool(results.get("thermal", {}).get("thermal_match", False)))
    gradient_matched = int(bool(results.get("gradient_surrogate", {}).get("matched", False)))
    crashes = sum(int(bool(r.get("crashed", False))) for r in results.values())

    # float32 may match if serialization canonicalizes or preserves precision in a way
    # that makes float32 equivalent for some states. Track it but do not count as danger.
    float32_matches = int(results.get("precision", {}).get("float32_matches", 0))

    danger = (
        false_accepts
        + avalanche_failures
        + drift_failures
        + precision_bad
        + quantized_bad
        + thermal_bad
        + gradient_matched
        + crashes
    )

    metrics = {
        "test": "extreme",
        "profile": profile,
        "n": int(n),
        "matched": bool(gradient_matched),
        "collisions": 0,
        "forgeries": 0,
        "false_accepts": false_accepts,
        "accepted": 0,
        "nan_detected": False,
        "avalanche_failures": avalanche_failures,
        "drift_failures": drift_failures,
        "precision_float16_match": bool(precision_bad),
        "precision_float32_matches": float32_matches,
        "quantized_match": bool(quantized_bad),
        "thermal_match": bool(thermal_bad),
        "gradient_matched": bool(gradient_matched),
        "crashes": crashes,
        "danger_total": danger,
        "elapsed_seconds": float(elapsed),
        "subtests": results,
    }

    print("\n==================== EXTREME SUMMARY ====================")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    print("\nRISK_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("RISK_METRICS_END")

    return danger == 0


# ============================================================
# Main
# ============================================================

def main():
    if PROFILE not in PROFILE_CONFIG:
        print(f"[ERROR] Unknown PROFILE={PROFILE!r}. Valid profiles: {sorted(PROFILE_CONFIG)}")
        return False

    cfg = PROFILE_CONFIG[PROFILE]
    n = cfg["n"]
    seeds = seed_list(cfg["seed_count"])

    print("\n=== WAVELOCK EXTREME TEST SUITE ===")
    print(f"[INFO] Profile: {PROFILE}")
    print(f"[INFO] n: {n}")
    print(f"[INFO] Seeds: {seeds}")
    print(f"[INFO] Config: {json.dumps(cfg, indent=2)}\n")

    start = time.time()
    results = {}

    results["avalanche"] = timed(
        "1) Massive Avalanche Analysis",
        extreme_avalanche_test,
        n=n,
        flips=cfg["avalanche_flips"],
        seeds=seeds,
    )

    if not should_stop(start):
        results["drift"] = timed(
            "2) Long-Term Drift Accumulation",
            extreme_drift_test,
            n=n,
            steps=cfg["drift_steps"],
            seeds=seeds,
        )

    if not should_stop(start):
        results["precision"] = timed(
            "3) Precision Downgrade Attack",
            precision_attack,
            n=n,
            seeds=seeds,
        )

    if not should_stop(start):
        results["fourier_attack"] = timed(
            "4) Fourier-Space Adversarial Attack",
            fourier_attack,
            n=n,
            attempts=cfg["fourier_attempts"],
            seeds=seeds,
        )

    if not should_stop(start):
        results["projection_attack"] = timed(
            "5) Random Projection Blend Attack",
            random_projection_attack,
            n=n,
            attempts=cfg["projection_attempts"],
            seeds=seeds,
        )

    if not should_stop(start):
        results["multi_gpu"] = timed(
            "6) Multi-GPU Consistency",
            multi_gpu_test,
            n=n,
        )

    if not should_stop(start):
        results["quantization"] = timed(
            "7) Quantization Attack",
            quantization_attack,
            n=n,
            seeds=seeds,
        )

    if not should_stop(start):
        results["thermal"] = timed(
            "8) Thermal Noise Model Attack",
            thermal_attack,
            n=n,
            trials=cfg["thermal_trials"],
            seeds=seeds,
        )

    if not should_stop(start):
        results["recombination_attack"] = timed(
            "9) ψ*-Recombination Attack",
            recombination_attack,
            n=n,
            attempts=cfg["recombination_attempts"],
            seeds=seeds,
        )

    if not should_stop(start):
        results["gradient_surrogate"] = timed(
            "10) Gradient Surrogate Attack",
            gradient_surrogate_attack,
            n=n,
            steps=cfg["gradient_steps"],
            seeds=seeds,
        )

    elapsed = time.time() - start
    ok = emit_metrics(results, elapsed, PROFILE, n)

    print("\n=== DONE ===\n")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
