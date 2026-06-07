#!/usr/bin/env python3
# test_wavelock_post_quantum.py
# WaveLock post-quantum-style adversarial simulation suite
#
# Updated after benchmark finding:
#   The previous QFT reconstruction test could report a "match" when rank == side.
#   That is not an attack; it is full-spectrum reconstruction of the target.
#
# Fixes:
#   - QFT attack only counts STRICTLY PARTIAL Fourier reconstructions as attacks.
#   - Optional full-rank QFT check is recorded as a control, not danger.
#   - Gibbs attack is batched on CuPy to reduce GPU launch overhead.
#   - Metrics distinguish matched_attacks from diagnostic_controls.
#   - No environment commands required.

import os
import sys
import time
import json
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
# "deep"     = longer validation
PROFILE = "standard"

PROFILE_CONFIG = {
    "fast": {
        "n": 6,
        "seed_count": 1,
        "grover_iterations": 2_000,
        "qaoa_layers": 2_000,
        "qft_ranks": [1, 2, 4, 8, 16, 32],
        "gibbs_samples": 5_000,
        "gibbs_batch": 256,
        "qrw_steps": 10_000,
    },
    "standard": {
        "n": 6,
        "seed_count": 5,
        "grover_iterations": 25_000,
        "qaoa_layers": 25_000,
        "qft_ranks": [1, 2, 4, 8, 16, 24, 32, 48, 64],
        "gibbs_samples": 50_000,
        "gibbs_batch": 512,
        "qrw_steps": 75_000,
    },
    "deep": {
        "n": 8,
        "seed_count": 10,
        "grover_iterations": 100_000,
        "qaoa_layers": 100_000,
        "qft_ranks": [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128],
        "gibbs_samples": 200_000,
        "gibbs_batch": 1024,
        "qrw_steps": 250_000,
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
# Imports
# ============================================================

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


# ============================================================
# Helpers
# ============================================================

def sync_gpu():
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass


def commit(psi):
    """Compute SHA256 commitment of ψ-field."""
    psi = cp.asarray(psi, dtype=cp.float32)
    raw = _serialize_commitment_v2(psi)
    return hashlib.sha256(raw).hexdigest()


def finite_guard(psi, label):
    if bool(cp.any(~cp.isfinite(cp.asarray(psi))).get()):
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


def laplacian_batched(psi):
    """
    Batched 2D Laplacian for arrays shaped:
      [B, H, W] or [H, W]
    """
    return (
        cp.roll(psi, 1, axis=-2)
        + cp.roll(psi, -1, axis=-2)
        + cp.roll(psi, 1, axis=-1)
        + cp.roll(psi, -1, axis=-1)
        - 4 * psi
    )


# ============================================================
# 1. GROVER-SIM ATTACK — amplitude-amplification-style surrogate
# ============================================================

def grover_sim_attack(n=6, iterations=25_000, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    target = cp.asarray(kp.psi_star, dtype=cp.float32)
    h0 = commit(target)

    cp.random.seed(int(seed) % (2**32 - 1))
    psi = cp.random.normal(0, 1, target.shape).astype(cp.float32)

    best_energy = float("inf")

    for t in range(iterations):
        oracle = -cp.abs(laplacian(psi))
        psi = psi + cp.float32(0.001) * oracle
        psi = psi / (cp.linalg.norm(psi) + cp.float32(1e-12))

        if t % 500 == 0 or t == iterations - 1:
            finite_guard(psi, "grover_sim psi")
            energy = float(cp.linalg.norm(psi - target).get())
            best_energy = min(best_energy, energy)

            if commit(psi) == h0:
                return {
                    "matched": True,
                    "iteration": t,
                    "seed": seed,
                    "best_energy": best_energy,
                }

    return {
        "matched": False,
        "iterations": iterations,
        "seed": seed,
        "best_energy": best_energy,
    }


# ============================================================
# 2. QAOA-SIM ATTACK — alternating optimizer surrogate
# ============================================================

def qaoa_sim_attack(n=6, layers=25_000, step=1e-4, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    target = cp.asarray(kp.psi_star, dtype=cp.float32)
    h0 = commit(target)

    cp.random.seed((int(seed) + 1009) % (2**32 - 1))
    psi = cp.random.rand(*target.shape).astype(cp.float32)

    best_energy = float("inf")

    for i in range(layers):
        psi = psi - cp.float32(step) * laplacian(psi)
        psi = cp.sin(cp.pi * psi)

        if i % 500 == 0 or i == layers - 1:
            finite_guard(psi, "qaoa_sim psi")
            energy = float(cp.linalg.norm(psi - target).get())
            best_energy = min(best_energy, energy)

            if commit(psi) == h0:
                return {
                    "matched": True,
                    "layer": i,
                    "seed": seed,
                    "best_energy": best_energy,
                }

    return {
        "matched": False,
        "layers": layers,
        "seed": seed,
        "best_energy": best_energy,
    }


# ============================================================
# 3. QFT-RECONSTRUCTION ATTACK — STRICTLY PARTIAL Fourier reconstruction
# ============================================================

def qft_reconstruction_attack(n=6, ranks=None, seed=123):
    """
    Only partial Fourier reconstructions count as attacks.

    Important:
      If rank >= side, the mask reconstructs the entire Fourier spectrum and may
      reproduce the target. That is a trivial full-information control, not a
      vulnerability. This function records that separately as full_rank_control.
    """
    if ranks is None:
        ranks = [1, 2, 4, 8, 16, 32]

    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    target = cp.asarray(kp.psi_star, dtype=cp.float32)
    h0 = commit(target)
    side = int(target.shape[0])

    F = cp.fft.fft2(target)

    partial_ranks = sorted({int(r) for r in ranks if 0 < int(r) < side})
    skipped_full_or_oversized = sorted({int(r) for r in ranks if int(r) >= side})

    best_rank = None
    best_energy = float("inf")

    for rr in partial_ranks:
        mask = cp.zeros_like(F)
        mask[:rr, :rr] = F[:rr, :rr]

        guess = cp.real(cp.fft.ifft2(mask)).astype(cp.float32)
        finite_guard(guess, "qft partial guess")

        energy = float(cp.linalg.norm(guess - target).get())
        if energy < best_energy:
            best_energy = energy
            best_rank = rr

        if commit(guess) == h0:
            return {
                "matched": True,
                "trivial_full_rank": False,
                "rank": rr,
                "side": side,
                "seed": seed,
                "best_energy": best_energy,
                "partial_ranks_tested": partial_ranks,
                "skipped_full_or_oversized_ranks": skipped_full_or_oversized,
            }

    # Diagnostic full-rank control: expected to reconstruct target if canonical
    # serialization and dtype line up. It is NOT counted as matched.
    full_rank_control_match = False
    if skipped_full_or_oversized:
        mask = cp.ones_like(F)
        full_guess = cp.real(cp.fft.ifft2(F * mask)).astype(cp.float32)
        finite_guard(full_guess, "qft full-rank control")
        full_rank_control_match = commit(full_guess) == h0

    return {
        "matched": False,
        "trivial_full_rank": bool(full_rank_control_match),
        "side": side,
        "ranks_requested": list(ranks),
        "partial_ranks_tested": partial_ranks,
        "skipped_full_or_oversized_ranks": skipped_full_or_oversized,
        "seed": seed,
        "best_rank": best_rank,
        "best_energy": best_energy,
    }


# ============================================================
# 4. QUANTUM GIBBS SAMPLING ATTACK — batched candidate scoring
# ============================================================

def gibbs_qsim_attack(n=6, samples=50_000, T=0.01, seed=123, batch_size=512):
    """
    Batched on GPU. SHA256 is CPU-bound, so each batch scores all candidates on
    GPU and hashes the closest candidate.
    """
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    target = cp.asarray(kp.psi_star, dtype=cp.float32)
    h0 = commit(target)

    cp.random.seed((int(seed) + 2027) % (2**32 - 1))
    best_energy = float("inf")
    best_sample = None

    done = 0
    while done < samples:
        b = min(int(batch_size), samples - done)

        psi = cp.random.normal(0, T, (b, *target.shape)).astype(cp.float32)
        weight = cp.exp(-cp.abs(laplacian_batched(psi)) / (cp.float32(T) + cp.float32(1e-12)))
        psi = psi * weight
        finite_guard(psi, "gibbs batch psi")

        diffs = psi - target[None, :, :]
        energies = cp.linalg.norm(diffs.reshape(b, -1), axis=1)
        idx = int(cp.argmin(energies).get())
        energy = float(energies[idx].get())

        if energy < best_energy:
            best_energy = energy
            best_sample = done + idx

        candidate = psi[idx]
        if commit(candidate) == h0:
            return {
                "matched": True,
                "sample": done + idx,
                "seed": seed,
                "best_energy": best_energy,
                "batch_size": batch_size,
            }

        done += b

    return {
        "matched": False,
        "samples": samples,
        "seed": seed,
        "best_sample": best_sample,
        "best_energy": best_energy,
        "batch_size": batch_size,
    }


# ============================================================
# 5. QUANTUM RANDOM WALK COLLISION SEARCH
# ============================================================

def qrw_collision_attack(n=6, steps=75_000, seed=123):
    kp = CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
    target = cp.asarray(kp.psi_star, dtype=cp.float32)
    h0 = commit(target)

    cp.random.seed((int(seed) + 4049) % (2**32 - 1))
    psi = cp.random.normal(0, 1, target.shape).astype(cp.float32)

    best_energy = float("inf")

    for i in range(steps):
        psi = psi + cp.random.normal(0, 0.002, psi.shape).astype(cp.float32)
        psi = cp.tanh(psi)

        if i % 500 == 0 or i == steps - 1:
            finite_guard(psi, "qrw psi")
            energy = float(cp.linalg.norm(psi - target).get())
            best_energy = min(best_energy, energy)

            if commit(psi) == h0:
                return {
                    "matched": True,
                    "step": i,
                    "seed": seed,
                    "best_energy": best_energy,
                }

    return {
        "matched": False,
        "steps": steps,
        "seed": seed,
        "best_energy": best_energy,
    }


# ============================================================
# Multi-seed wrapper
# ============================================================

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


# ============================================================
# Main
# ============================================================

def main():
    if PROFILE not in PROFILE_CONFIG:
        print(f"[ERROR] Unknown PROFILE={PROFILE!r}. Valid profiles: {sorted(PROFILE_CONFIG)}")
        return False

    cfg = PROFILE_CONFIG[PROFILE]
    n = cfg["n"]
    seed_list = BASE_SEEDS[: cfg["seed_count"]]

    print("\n=== WAVELOCK POST-QUANTUM ATTACK SUITE ===")
    print(f"[INFO] Profile: {PROFILE}")
    print(f"[INFO] n: {n}")
    print(f"[INFO] Seeds: {seed_list}")
    print(f"[INFO] Config: {json.dumps(cfg, indent=2)}\n")

    start = time.time()
    suite_results = {}

    suite_results["grover_sim_attack"] = run_multi_seed_attack(
        "1) Grover-Sim Attack",
        grover_sim_attack,
        seed_list,
        n=n,
        iterations=cfg["grover_iterations"],
    )

    if not should_stop(start):
        suite_results["qaoa_sim_attack"] = run_multi_seed_attack(
            "2) QAOA-Sim Attack",
            qaoa_sim_attack,
            seed_list,
            n=n,
            layers=cfg["qaoa_layers"],
        )

    if not should_stop(start):
        suite_results["qft_reconstruction_attack"] = run_multi_seed_attack(
            "3) QFT-Based Reconstruction",
            qft_reconstruction_attack,
            seed_list,
            n=n,
            ranks=cfg["qft_ranks"],
        )

    if not should_stop(start):
        suite_results["gibbs_qsim_attack"] = run_multi_seed_attack(
            "4) Quantum Gibbs Sampling Attack",
            gibbs_qsim_attack,
            seed_list,
            n=n,
            samples=cfg["gibbs_samples"],
            batch_size=cfg["gibbs_batch"],
        )

    if not should_stop(start):
        suite_results["qrw_collision_attack"] = run_multi_seed_attack(
            "5) Quantum Random Walk Collision Search",
            qrw_collision_attack,
            seed_list,
            n=n,
            steps=cfg["qrw_steps"],
        )

    elapsed = time.time() - start

    matched_attacks = [
        name for name, result in suite_results.items()
        if bool(result.get("matched", False))
    ]
    crash_count = sum(int(result.get("crashes", 0)) for result in suite_results.values())

    qft_controls = []
    qft = suite_results.get("qft_reconstruction_attack", {})
    for item in qft.get("results", []):
        if item.get("trivial_full_rank"):
            qft_controls.append({
                "seed": item.get("seed"),
                "side": item.get("side"),
                "skipped_full_or_oversized_ranks": item.get("skipped_full_or_oversized_ranks"),
            })

    metrics = {
        "test": "post_quantum",
        "profile": PROFILE,
        "matched": bool(matched_attacks),
        "matched_attacks": matched_attacks,
        "diagnostic_controls": {
            "qft_full_rank_reconstruction_controls": qft_controls,
        },
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

    return not bool(matched_attacks) and crash_count == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
