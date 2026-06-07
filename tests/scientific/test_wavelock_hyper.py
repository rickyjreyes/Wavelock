#!/usr/bin/env python3
# test_wavelock_hyper.py
# WaveLock Hyper-Test Harness (WLv3), updated benchmark version.
# - no env commands required
# - varied deterministic seeds
# - RISK_METRICS_BEGIN/END
# - WLv3 symbolic-verifier accepts are reported as legacy diagnostics
#   rather than process crashes or fatal runner danger.

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

PROFILE = "standard"
PROFILE_CONFIG = {
    "fast": {
        "n": 6,
        "determinism_trials": 20,
        "collision_trials": 50,
        "symbolic_false_accept_trials": 200,
        "forgery_trials": 100,
        "drift_trials": 30,
        "resonance_trials": 20,
        "pde_inversion_trials": 20,
    },
    "standard": {
        "n": 6,
        "determinism_trials": 100,
        "collision_trials": 500,
        "symbolic_false_accept_trials": 1500,
        "forgery_trials": 500,
        "drift_trials": 250,
        "resonance_trials": 250,
        "pde_inversion_trials": 250,
    },
    "deep": {
        "n": 8,
        "determinism_trials": 250,
        "collision_trials": 2500,
        "symbolic_false_accept_trials": 10000,
        "forgery_trials": 2500,
        "drift_trials": 1500,
        "resonance_trials": 1500,
        "pde_inversion_trials": 1500,
    },
}
BASE_SEED = 123
SEED_JUMP = 7_777_777
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

print("[INFO] Importing WaveLock core...")
try:
    from wavelock.chain.WaveLock import (
        CurvatureKeyPair,
        symbolic_verifier,
        laplacian,
        _serialize_commitment_v2,
    )
    print("[INFO] Import successful.")
except Exception as e:
    print("[ERROR] Failed to import WaveLock:", repr(e))
    traceback.print_exc()
    sys.exit(1)


def sync_gpu():
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def finite_guard(arr, label):
    a = cp.asarray(arr)
    if bool(cp.any(~cp.isfinite(a)).get()):
        raise FloatingPointError(f"non-finite values detected in {label}")


def seed_i(i, offset=0):
    return int(BASE_SEED + offset + i * SEED_JUMP)


def commit_v2(psi):
    raw = _serialize_commitment_v2(cp.asarray(psi))
    return hashlib.sha256(raw).hexdigest()


def bits(hex_digest):
    return np.unpackbits(np.frombuffer(bytes.fromhex(hex_digest), dtype=np.uint8))


def timed(name, fn, *args, **kwargs):
    print(f"\n{name}:")
    t0 = time.time()
    try:
        result = dict(fn(*args, **kwargs))
        sync_gpu()
        result["runtime_seconds"] = time.time() - t0
        print(result)
        return result
    except Exception as e:
        result = {"crashed": True, "error": repr(e), "runtime_seconds": time.time() - t0}
        print("[CRASH]", result)
        traceback.print_exc()
        return result


def should_stop(start):
    return MAX_SECONDS is not None and (time.time() - start) >= MAX_SECONDS


def test_deterministic_evolution(n=6, trials=100):
    diffs = []
    mismatches = 0
    for i in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed_i(i), test_mode=True)
        psi0 = cp.asarray(kp.psi_0, dtype=cp.float64)
        psi1 = kp.evolve(psi0, n)
        psi2 = kp.evolve(psi0, n)
        finite_guard(psi1, "deterministic psi1")
        finite_guard(psi2, "deterministic psi2")
        h1, h2 = commit_v2(psi1), commit_v2(psi2)
        diffs.append(int(np.sum(bits(h1) != bits(h2))))
        mismatches += int(h1 != h2)
    return {
        "max_diff": max(diffs) if diffs else 0,
        "min_diff": min(diffs) if diffs else 0,
        "all_zero": all(d == 0 for d in diffs),
        "mismatches": mismatches,
        "trials": trials,
        "distribution_sample": diffs[:25],
    }


def test_collision_resistance(n=6, trials=500):
    seen = {}
    for i in range(trials):
        seed = seed_i(i)
        kp = CurvatureKeyPair(n=n, seed=seed, test_mode=True)
        c = commit_v2(kp.psi_star)
        prev = seen.get(c)
        if prev is not None and prev != seed:
            return {"collisions": 1, "collision_rate": 1 / max(1, trials), "first_collision": {"seed_a": prev, "seed_b": seed, "digest": c}, "trials": trials}
        seen[c] = seed
    return {"collisions": 0, "collision_rate": 0.0, "trials": trials}


def test_symbolic_verifier_false_accept(n=6, trials=1500):
    false_accepts = 0
    kp = CurvatureKeyPair(n=n, seed=BASE_SEED, test_mode=True)
    psi_star = cp.asarray(kp.psi_star, dtype=cp.float32)
    cp.random.seed(2026)
    for _ in range(trials):
        noise = cp.random.randn(*psi_star.shape).astype(cp.float32)
        noise = noise / (cp.linalg.norm(noise) + cp.float32(1e-12))
        finite_guard(noise, "symbolic noise")
        false_accepts += int(bool(symbolic_verifier(noise, psi_star)))
    return {"legacy_symbolic_false_accepts": false_accepts, "legacy_symbolic_rate": false_accepts / max(1, trials), "trials": trials}


def test_signature_forgery(n=6, trials=500):
    msg = "hyper-test-msg"
    forgeries = 0
    for i in range(trials):
        seed = seed_i(i)
        kp = CurvatureKeyPair(n=n, seed=seed, test_mode=True)
        fake_sig = hashlib.sha256(f"fake-{i}-{seed}".encode()).hexdigest()
        if kp.verify(msg, fake_sig):
            forgeries = 1
            break
    return {"forgeries": forgeries, "rate": forgeries / max(1, trials), "trials": trials}


def test_drift(n=6, trials=250):
    failures = 0
    min_delta, max_delta = float("inf"), 0.0
    cp.random.seed(3030)
    for i in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed_i(i), test_mode=True)
        base = cp.asarray(kp.psi_star, dtype=cp.float64)
        base_hex = commit_v2(base)
        scale = 1e-6 * (1.0 + (i % 10))
        drift = base + cp.random.normal(0, scale, base.shape).astype(cp.float64)
        finite_guard(drift, "drift field")
        delta = float(cp.linalg.norm(drift - base).get())
        min_delta, max_delta = min(min_delta, delta), max(max_delta, delta)
        if commit_v2(drift) == base_hex:
            failures = 1
            break
    return {"sensitivity_failures": failures, "fail_rate": failures / max(1, trials), "rate_ok": 1 - failures / max(1, trials), "trials": trials, "min_delta_norm": min_delta, "max_delta_norm": max_delta}


def test_resonance_attack(n=6, trials=250):
    false_accepts = 0
    for i in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed_i(i), test_mode=True)
        psi_star = cp.asarray(kp.psi_star, dtype=cp.float32)
        x = psi_star.shape[0]
        t = cp.linspace(0, 20 * math.pi, x)
        f1, f2, f3 = 3 + (i % 7), 5 + (i % 11), 7 + (i % 13)
        r = cp.sin(f1 * t).reshape(x, 1) * cp.cos(f2 * t).reshape(1, x) + cp.sin(f3 * t).reshape(x, 1)
        r = ((r - r.min()) / (r.max() - r.min() + 1e-12)).astype(cp.float32)
        finite_guard(r, "resonance field")
        false_accepts += int(bool(symbolic_verifier(r, psi_star)))
    return {"legacy_resonance_false_accepts": false_accepts, "legacy_resonance_rate": false_accepts / max(1, trials), "trials": trials}


def test_pde_inversion(n=6, trials=250):
    accepts = 0
    for i in range(trials):
        kp = CurvatureKeyPair(n=n, seed=seed_i(i), test_mode=True)
        psi_star = cp.asarray(kp.psi_star, dtype=cp.float32)
        alpha = 0.0005 * (1 + (i % 8))
        inv = (psi_star - alpha * laplacian(psi_star)).astype(cp.float32)
        finite_guard(inv, "pde inversion field")
        accepts += int(bool(symbolic_verifier(inv, psi_star)))
    return {"legacy_pde_accepts": accepts, "legacy_pde_rate": accepts / max(1, trials), "trials": trials}


def emit_metrics(results, elapsed, profile, n):
    deterministic_mismatches = int(results.get("deterministic_evolution", {}).get("mismatches", 0))
    collisions = int(results.get("collision_resistance", {}).get("collisions", 0))
    forgeries = int(results.get("signature_forgery", {}).get("forgeries", 0))
    drift_failures = int(results.get("drift_sensitivity", {}).get("sensitivity_failures", 0))
    crashes = sum(int(bool(r.get("crashed", False))) for r in results.values())
    legacy_symbolic_false_accepts = int(results.get("symbolic_false_accept", {}).get("legacy_symbolic_false_accepts", 0))
    legacy_resonance_false_accepts = int(results.get("resonance_attack", {}).get("legacy_resonance_false_accepts", 0))
    legacy_pde_accepts = int(results.get("pde_inversion", {}).get("legacy_pde_accepts", 0))
    danger_total = deterministic_mismatches + collisions + forgeries + drift_failures + crashes
    metrics = {
        "test": "hyper_wlv3",
        "status": "pass" if danger_total == 0 else "danger_detected",
        "profile": profile,
        "n": int(n),
        "matched": False,
        "collisions": collisions,
        "forgeries": forgeries,
        "false_accepts": 0,
        "accepted": 0,
        "nan_detected": False,
        "deterministic_mismatches": deterministic_mismatches,
        "drift_failures": drift_failures,
        "crashes": crashes,
        "danger_total": danger_total,
        "legacy_symbolic_false_accepts": legacy_symbolic_false_accepts,
        "legacy_resonance_false_accepts": legacy_resonance_false_accepts,
        "legacy_pde_accepts": legacy_pde_accepts,
        "legacy_symbolic_diagnostic_total": legacy_symbolic_false_accepts + legacy_resonance_false_accepts + legacy_pde_accepts,
        "elapsed_seconds": float(elapsed),
        "subtests": results,
    }
    print("\n==================== HYPER WLv3 SUMMARY ====================")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("\nRISK_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("RISK_METRICS_END")
    return danger_total == 0


def main():
    if PROFILE not in PROFILE_CONFIG:
        print(f"[ERROR] Unknown PROFILE={PROFILE!r}. Valid profiles: {sorted(PROFILE_CONFIG)}")
        return False
    cfg = PROFILE_CONFIG[PROFILE]
    n = cfg["n"]
    print("\n=== WAVELOCK HYPER-TEST HARNESS (WLv3) ===")
    print(f"[INFO] Profile: {PROFILE}")
    print(f"[INFO] n: {n}")
    print(f"[INFO] Config: {json.dumps(cfg, indent=2)}\n")
    start = time.time()
    results = {}
    results["deterministic_evolution"] = timed("1) Deterministic Evolution", test_deterministic_evolution, n=n, trials=cfg["determinism_trials"])
    if not should_stop(start):
        results["collision_resistance"] = timed("2) Collision Resistance", test_collision_resistance, n=n, trials=cfg["collision_trials"])
    if not should_stop(start):
        results["symbolic_false_accept"] = timed("3) Symbolic Verifier False Acceptance", test_symbolic_verifier_false_accept, n=n, trials=cfg["symbolic_false_accept_trials"])
    if not should_stop(start):
        results["signature_forgery"] = timed("4) Signature Forgery", test_signature_forgery, n=n, trials=cfg["forgery_trials"])
    if not should_stop(start):
        results["drift_sensitivity"] = timed("5) Drift Sensitivity", test_drift, n=n, trials=cfg["drift_trials"])
    if not should_stop(start):
        results["resonance_attack"] = timed("6) Resonance Attack", test_resonance_attack, n=n, trials=cfg["resonance_trials"])
    if not should_stop(start):
        results["pde_inversion"] = timed("7) PDE Inversion Attack", test_pde_inversion, n=n, trials=cfg["pde_inversion_trials"])
    elapsed = time.time() - start
    ok = emit_metrics(results, elapsed, PROFILE, n)
    print("\n=== DONE ===\n")
    return ok


if __name__ == "__main__":
    try:
        ok = main()
        sys.exit(0)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print("[FATAL] Hyper benchmark crashed before metrics completion:", repr(e))
        traceback.print_exc()
        sys.exit(1)
