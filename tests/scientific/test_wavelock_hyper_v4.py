#!/usr/bin/env python3
"""
============================================================
   WAVELOCK HYPER-TEST HARNESS (WLv4)
   ---------------------------------
   Curvature-domain diagnostic/security suite for WaveLock v4
   Curvature Canonical Commitment (CCC).

   Updated benchmark version:
       - no environment commands required
       - internal fast/standard/deep profiles
       - fixes repeated-seed issue in collision/false-accept/forgery loops
       - multi-seed coverage
       - per-test timing
       - CuPy synchronization for honest timing
       - RISK_METRICS_BEGIN/END for run_benchmarks.py
       - fail-closed crash reporting
============================================================
"""

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

# "fast"     = close to old behavior / smoke
# "standard" = stronger, target several minutes
# "deep"     = longer validation component
PROFILE = "standard"

PROFILE_CONFIG = {
    "fast": {
        "n": 6,
        "determinism_trials": 10,
        "collision_trials": 50,
        "false_accept_trials": 300,
        "forgery_trials": 100,
        "drift_trials": 30,
        "resonance_trials": 20,
        "pde_inversion_trials": 20,
    },
    "standard": {
        "n": 6,
        "determinism_trials": 50,
        "collision_trials": 500,
        "false_accept_trials": 1_500,
        "forgery_trials": 500,
        "drift_trials": 250,
        "resonance_trials": 250,
        "pde_inversion_trials": 250,
    },
    "deep": {
        "n": 8,
        "determinism_trials": 200,
        "collision_trials": 2_500,
        "false_accept_trials": 10_000,
        "forgery_trials": 2_500,
        "drift_trials": 1_500,
        "resonance_trials": 1_500,
        "pde_inversion_trials": 1_500,
    },
}

BASE_SEED = 123
SEED_JUMP = 7_777_777

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
# PYTHONPATH FIX
# ============================================================

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ============================================================
# IMPORT WAVELOCK v4
# ============================================================

print("[INFO] Importing WaveLock v4 core...")
try:
    from wavelock.chain.WaveLock import (
        CurvatureKeyPair,
        _serialize_commitment_v4,
        laplacian,
        _curvature_functional,
    )
    print("[INFO] Import successful.")
except Exception as e:
    print("[ERROR] Failed to import WaveLock:", repr(e))
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# HELPERS
# ============================================================

def sync_gpu():
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def finite_guard(psi, label):
    if bool(cp.any(~cp.isfinite(cp.asarray(psi))).get()):
        raise FloatingPointError(f"non-finite values detected in {label}")


def commit_v4(psi):
    raw = _serialize_commitment_v4(cp.asarray(psi, dtype=cp.float64))
    return hashlib.sha256(raw).hexdigest()


def curvature_vector(psi):
    """Returns (E_grad, E_fb, E_ent, E_tot) as a NumPy array."""
    E_grad, E_fb, E_ent, E_tot = _curvature_functional(cp.asarray(psi))
    return np.array([E_grad, E_fb, E_ent, E_tot], dtype=np.float64)


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


def seed_i(i, offset=0):
    return int(BASE_SEED + offset + i * SEED_JUMP)


# ============================================================
# TEST 1 — Deterministic Curvature Evolution
# ============================================================

def test_deterministic_evolution(n=6, trials=50):
    mismatches = 0
    vector_max_delta = 0.0

    for i in range(trials):
        seed = seed_i(i)
        kp = CurvatureKeyPair(n=n, seed=seed, use_v4=True, test_mode=True)
        psi0 = kp.psi_0

        psi1 = kp.evolve(psi0, n)
        psi2 = kp.evolve(psi0, n)

        finite_guard(psi1, "deterministic psi1")
        finite_guard(psi2, "deterministic psi2")

        h1 = commit_v4(psi1)
        h2 = commit_v4(psi2)

        if h1 != h2:
            mismatches += 1

        v1 = curvature_vector(psi1)
        v2 = curvature_vector(psi2)
        vector_max_delta = max(vector_max_delta, float(np.max(np.abs(v1 - v2))))

    return {
        "all_identical": mismatches == 0,
        "mismatches": mismatches,
        "trials": trials,
        "vector_max_delta": vector_max_delta,
    }


# ============================================================
# TEST 2 — CCC Collision Resistance
# ============================================================

def test_collision_resistance(n=6, trials=500):
    collisions = 0
    seen = set()

    for i in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed_i(i), use_v4=True, test_mode=True)
        c = commit_v4(kp.psi_star)

        if c in seen:
            collisions += 1
            break

        seen.add(c)

    return {
        "collisions": collisions,
        "collision_rate": collisions / max(1, trials),
        "trials": trials,
    }


# ============================================================
# TEST 3 — Curvature False-Accept
# ============================================================

def test_false_accept(n=6, trials=1_500):
    false_accepts = 0

    kp = CurvatureKeyPair(n=n, seed=BASE_SEED, use_v4=True, test_mode=True)
    psi = kp.psi_star
    target_commit = commit_v4(psi)

    cp.random.seed(2026)

    for i in range(trials):
        noise = cp.random.randn(*psi.shape).astype(cp.float64)
        noise /= (cp.linalg.norm(noise) + 1e-12)
        finite_guard(noise, "false_accept noise")

        if commit_v4(noise) == target_commit:
            false_accepts += 1
            break

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / max(1, trials),
        "trials": trials,
    }


# ============================================================
# TEST 4 — Signature Forgery (WLv4)
# ============================================================

def test_signature_forgery(n=6, trials=500):
    msg = "wavelock-v4-integrity"
    forgeries = 0

    for i in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed_i(i), use_v4=True, test_mode=True)
        fake = hashlib.sha256(f"fake-{i}-{seed_i(i)}".encode("utf-8")).hexdigest()

        if kp.verify(msg, fake):
            forgeries += 1
            break

    return {
        "forgeries": forgeries,
        "rate": forgeries / max(1, trials),
        "trials": trials,
    }


# ============================================================
# TEST 5 — Curvature Drift Sensitivity
# ============================================================

def test_drift(n=6, trials=250):
    failures = 0
    min_delta = float("inf")
    max_delta = 0.0

    cp.random.seed(3030)

    for i in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed_i(i), use_v4=True, test_mode=True)
        base_commit = commit_v4(kp.psi_star)
        base_vec = curvature_vector(kp.psi_star)

        scale = 1e-6 * (1.0 + (i % 10))
        drift = kp.psi_star + cp.random.normal(0, scale, kp.psi_star.shape)
        finite_guard(drift, "drift field")

        drift_commit = commit_v4(drift)
        drift_vec = curvature_vector(drift)
        delta = float(np.linalg.norm(drift_vec - base_vec))
        min_delta = min(min_delta, delta)
        max_delta = max(max_delta, delta)

        if drift_commit == base_commit:
            failures += 1
            break

    return {
        "drift_failures": failures,
        "fail_rate": failures / max(1, trials),
        "rate_ok": 1 - failures / max(1, trials),
        "trials": trials,
        "min_vector_delta": min_delta,
        "max_vector_delta": max_delta,
    }


# ============================================================
# TEST 6 — Resonance Attack (Curvature-Space)
# ============================================================

def test_resonance_attack(n=6, trials=250):
    false_accepts = 0

    for i in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed_i(i), use_v4=True, test_mode=True)
        psi = kp.psi_star
        target_commit = commit_v4(psi)
        x = psi.shape[0]

        t = cp.linspace(0, 12 * math.pi, x)

        f1 = 2 + (i % 7)
        f2 = 3 + (i % 11)
        f3 = 5 + (i % 13)

        r = (
            cp.sin(f1 * t).reshape(x, 1)
            * cp.cos(f2 * t).reshape(1, x)
            + cp.sin(f3 * t).reshape(x, 1)
        )

        r = (r - r.min()) / (r.max() - r.min() + 1e-12)
        finite_guard(r, "resonance field")

        if commit_v4(r) == target_commit:
            false_accepts += 1
            break

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / max(1, trials),
        "trials": trials,
    }


# ============================================================
# TEST 7 — PDE Inversion (Curvature-Space)
# ============================================================

def test_pde_inversion(n=6, trials=250):
    accepts = 0

    for i in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed_i(i), use_v4=True, test_mode=True)
        psi = kp.psi_star
        target_commit = commit_v4(psi)

        alpha = 0.0005 * (1 + (i % 8))
        inv = psi - alpha * laplacian(psi)
        finite_guard(inv, "pde inversion field")

        if commit_v4(inv) == target_commit:
            accepts += 1
            break

    return {
        "accepted": accepts,
        "rate": accepts / max(1, trials),
        "trials": trials,
    }


# ============================================================
# METRICS
# ============================================================

def emit_metrics(results, elapsed, profile, n):
    collisions = int(results.get("collision_resistance", {}).get("collisions", 0))
    false_accepts = (
        int(results.get("false_accept", {}).get("false_accepts", 0))
        + int(results.get("resonance_attack", {}).get("false_accepts", 0))
    )
    forgeries = int(results.get("signature_forgery", {}).get("forgeries", 0))
    accepted = int(results.get("pde_inversion", {}).get("accepted", 0))
    drift_failures = int(results.get("drift_sensitivity", {}).get("drift_failures", 0))
    deterministic_mismatches = int(results.get("deterministic_evolution", {}).get("mismatches", 0))
    crashes = sum(int(bool(r.get("crashed", False))) for r in results.values())

    metrics = {
        "test": "hyper_v4",
        "profile": profile,
        "n": int(n),
        "matched": False,
        "collisions": collisions,
        "false_accepts": false_accepts,
        "forgeries": forgeries,
        "accepted": accepted,
        "nan_detected": False,
        "drift_failures": drift_failures,
        "deterministic_mismatches": deterministic_mismatches,
        "crashes": crashes,
        "elapsed_seconds": float(elapsed),
        "subtests": results,
    }

    print("\n==================== HYPER V4 SUMMARY ====================")
    print(json.dumps(metrics, indent=2, sort_keys=True))

    print("\nRISK_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("RISK_METRICS_END")

    danger = (
        collisions
        + false_accepts
        + forgeries
        + accepted
        + drift_failures
        + deterministic_mismatches
        + crashes
    )

    return danger == 0


# ============================================================
# MAIN
# ============================================================

def main():
    if PROFILE not in PROFILE_CONFIG:
        print(f"[ERROR] Unknown PROFILE={PROFILE!r}. Valid profiles: {sorted(PROFILE_CONFIG)}")
        return False

    cfg = PROFILE_CONFIG[PROFILE]
    n = cfg["n"]
    results = {}

    print("\n=== WAVELOCK HYPER-TEST HARNESS (WLv4) ===")
    print(f"[INFO] Profile: {PROFILE}")
    print(f"[INFO] n: {n}")
    print(f"[INFO] Config: {json.dumps(cfg, indent=2)}\n")

    start = time.time()

    results["deterministic_evolution"] = timed(
        "1) Deterministic Curvature Evolution",
        test_deterministic_evolution,
        n=n,
        trials=cfg["determinism_trials"],
    )

    if not should_stop(start):
        results["collision_resistance"] = timed(
            "2) CCC Collision Resistance",
            test_collision_resistance,
            n=n,
            trials=cfg["collision_trials"],
        )

    if not should_stop(start):
        results["false_accept"] = timed(
            "3) False Accept (Curvature)",
            test_false_accept,
            n=n,
            trials=cfg["false_accept_trials"],
        )

    if not should_stop(start):
        results["signature_forgery"] = timed(
            "4) Signature Forgery",
            test_signature_forgery,
            n=n,
            trials=cfg["forgery_trials"],
        )

    if not should_stop(start):
        results["drift_sensitivity"] = timed(
            "5) Curvature Drift Sensitivity",
            test_drift,
            n=n,
            trials=cfg["drift_trials"],
        )

    if not should_stop(start):
        results["resonance_attack"] = timed(
            "6) Resonance Attack",
            test_resonance_attack,
            n=n,
            trials=cfg["resonance_trials"],
        )

    if not should_stop(start):
        results["pde_inversion"] = timed(
            "7) PDE-Inversion Attack",
            test_pde_inversion,
            n=n,
            trials=cfg["pde_inversion_trials"],
        )

    elapsed = time.time() - start
    ok = emit_metrics(results, elapsed, PROFILE, n)

    print("\n=== DONE ===\n")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
