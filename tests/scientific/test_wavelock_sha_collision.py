#!/usr/bin/env python3
# test_wavelock_sha_collision.py
# ============================================================
# WaveLock SHA256-lifted collision cluster
#
# Updated/turbo exact version:
# - no environment commands required
# - exact SHA256 digest comparison, not a fingerprint shortcut
# - ProcessPoolExecutor with per-worker initializer
# - imports WaveLock once per worker instead of once per chunk
# - returns packed digest blobs instead of Python lists of bytes
# - parent reconstructs seed mapping exactly
# - redundant local duplicate dictionary removed by default
# - RISK_METRICS_BEGIN/END for run_benchmarks.py
#
# Why faster than previous optimized version:
#   Previous version still imported WaveLock inside every chunk and returned
#   list[bytes]. This version imports once per process and returns one packed
#   bytes blob per chunk: digest0||digest1||...||digestN.
# ============================================================

import os
import sys
import time
import json
import hashlib
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
# CONFIG — edit here only
# ============================================================

PROFILE = "standard"

PROFILE_CONFIG = {
    "fast": {
        "n": 6,
        "seed_max": 25_000,
        "workers": 4,
        "chunk_size": 5_000,
    },
    "standard": {
        "n": 6,
        "seed_max": 100_000,
        "workers": 6,
        "chunk_size": 10_000,
    },
    "deep": {
        "n": 6,
        "seed_max": 1_000_000,
        "workers": 8,
        "chunk_size": 25_000,
    },
}

DIGEST_SIZE = 32
MAX_SECONDS = None
LOCAL_DUP_CHECK = False
PROGRESS_EVERY_CHUNKS = 1

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
try:
    REPO_ROOT = THIS_FILE.parents[2]
except IndexError:
    REPO_ROOT = Path.cwd()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

LOG_DIR = REPO_ROOT / "scripts" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_PATH = LOG_DIR / "sha_collision_progress.json"

# ============================================================
# Worker globals initialized once per process
# ============================================================

_WORKER_READY = False
_CurvatureKeyPair = None
_serialize_commitment_v2 = None


def worker_init(repo_root_str):
    """
    Runs once per worker process. This avoids re-importing WaveLock for every
    chunk submitted to the same worker.
    """
    global _WORKER_READY, _CurvatureKeyPair, _serialize_commitment_v2

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from wavelock.chain.WaveLock import CurvatureKeyPair, _serialize_commitment_v2 as serializer

    _CurvatureKeyPair = CurvatureKeyPair
    _serialize_commitment_v2 = serializer
    _WORKER_READY = True


# ============================================================
# Worker
# ============================================================

def worker_range(args):
    """
    Compute exact SHA256 digests for seeds [start, stop).

    Returns packed digest blob:
        digest_blob = digest(seed=start) || digest(start+1) || ...

    Parent maps digest offset back to seed exactly.
    """
    start, stop, n = args
    t0 = time.time()

    try:
        if not _WORKER_READY:
            worker_init(str(REPO_ROOT))

        local_seen = {} if LOCAL_DUP_CHECK else None
        parts = []
        append = parts.append

        for seed in range(start, stop):
            kp = _CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
            raw = _serialize_commitment_v2(kp.psi_star)
            digest = hashlib.sha256(raw).digest()

            if LOCAL_DUP_CHECK:
                prev = local_seen.get(digest)
                if prev is not None and prev != seed:
                    return {
                        "ok": False,
                        "start": start,
                        "stop": stop,
                        "count": seed - start + 1,
                        "digest_blob": b"",
                        "collision": {
                            "digest_hex": digest.hex(),
                            "seed_a": int(prev),
                            "seed_b": int(seed),
                            "scope": "local",
                        },
                        "runtime": time.time() - t0,
                        "error": None,
                    }
                local_seen[digest] = seed

            append(digest)

        return {
            "ok": True,
            "start": start,
            "stop": stop,
            "count": stop - start,
            "digest_blob": b"".join(parts),
            "collision": None,
            "runtime": time.time() - t0,
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "start": start,
            "stop": stop,
            "count": 0,
            "digest_blob": b"",
            "collision": None,
            "runtime": time.time() - t0,
            "error": repr(e) + "\n" + traceback.format_exc(),
        }


# ============================================================
# Metrics/progress
# ============================================================

def emit_metrics(status, completed, seed_max, collisions, elapsed, rate, profile, n, workers, chunk_size):
    crashes = 1 if status in {"worker_crash", "runner_crash"} else 0
    danger_total = int(collisions) + crashes

    metrics = {
        "test": "sha_collision_cluster",
        "profile": profile,
        "status": status,
        "matched": False,
        "collisions": int(collisions),
        "forgeries": 0,
        "false_accepts": 0,
        "accepted": 0,
        "nan_detected": False,
        "crashes": crashes,
        "danger_total": danger_total,
        "trials_completed": int(completed),
        "target_trials": int(seed_max),
        "elapsed_seconds": float(elapsed),
        "rate_per_second": float(rate),
        "n": int(n),
        "workers": int(workers),
        "chunk_size": int(chunk_size),
        "exact_sha256": True,
        "packed_digest_blob": True,
        "worker_initializer": True,
        "local_dup_check": bool(LOCAL_DUP_CHECK),
    }

    print("\nRISK_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("RISK_METRICS_END")


def write_progress(done, seed_max, elapsed, rate, chunks_done, total_chunks, profile, n, workers, chunk_size):
    data = {
        "done": int(done),
        "target": int(seed_max),
        "elapsed_seconds": float(elapsed),
        "rate_per_second": float(rate),
        "chunks_done": int(chunks_done),
        "total_chunks": int(total_chunks),
        "profile": profile,
        "n": int(n),
        "workers": int(workers),
        "chunk_size": int(chunk_size),
        "timestamp": time.time(),
    }

    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def iter_digest_blob(digest_blob):
    if len(digest_blob) % DIGEST_SIZE != 0:
        raise ValueError(
            f"digest blob length {len(digest_blob)} is not multiple of {DIGEST_SIZE}"
        )

    for offset in range(0, len(digest_blob), DIGEST_SIZE):
        yield digest_blob[offset:offset + DIGEST_SIZE]


# ============================================================
# Main collision cluster
# ============================================================

def run_sha_collision_cluster():
    if PROFILE not in PROFILE_CONFIG:
        print(f"[ERROR] Unknown PROFILE={PROFILE!r}. Valid profiles: {sorted(PROFILE_CONFIG)}")
        return False

    cfg = PROFILE_CONFIG[PROFILE]
    n = int(cfg["n"])
    seed_max = int(cfg["seed_max"])
    workers = int(cfg["workers"])
    chunk_size = int(cfg["chunk_size"])

    chunks = [
        (start, min(start + chunk_size, seed_max), n)
        for start in range(0, seed_max, chunk_size)
    ]

    print("\n=== SHA256 Collision Cluster ===")
    print(f"[INFO] Profile: {PROFILE}")
    print(f"[INFO] n: {n}")
    print(f"[INFO] seed range: [0, {seed_max:,})")
    print(f"[INFO] workers: {workers}")
    print(f"[INFO] chunk_size: {chunk_size:,}")
    print(f"[INFO] chunks: {len(chunks)}")
    print(f"[INFO] progress: {PROGRESS_PATH}")
    print("[INFO] Mode: exact SHA256 digest comparison")
    print("[INFO] Transport: packed digest blobs")
    print("[INFO] Worker import: initializer once per process")

    if not chunks:
        print("[ERROR] No chunks to process.")
        return False

    global_table = {}
    completed = 0
    chunks_done = 0
    collisions = []
    start_time = time.time()

    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=worker_init,
            initargs=(str(REPO_ROOT),),
        ) as executor:
            futures = [executor.submit(worker_range, chunk) for chunk in chunks]

            for fut in as_completed(futures):
                result = fut.result()
                chunks_done += 1

                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0.0

                if not result["ok"]:
                    if result.get("collision"):
                        collisions.append(result["collision"])
                        print("\n!!! LOCAL COLLISION FOUND !!!")
                        print(json.dumps(result["collision"], indent=2))
                        emit_metrics("collision", completed, seed_max, len(collisions), elapsed, rate, PROFILE, n, workers, chunk_size)
                    else:
                        print("\n[ERROR] Worker failure.")
                        print(f"Chunk: {result['start']:,} to {result['stop']:,}")
                        print(result.get("error"))
                        emit_metrics("worker_crash", completed, seed_max, len(collisions), elapsed, rate, PROFILE, n, workers, chunk_size)
                    return False

                start_seed = int(result["start"])
                digest_blob = result["digest_blob"]
                expected_count = int(result["count"])
                actual_count = len(digest_blob) // DIGEST_SIZE

                if actual_count != expected_count:
                    print(f"[ERROR] digest count mismatch: actual={actual_count} expected={expected_count}")
                    emit_metrics("runner_crash", completed, seed_max, len(collisions), elapsed, rate, PROFILE, n, workers, chunk_size)
                    return False

                for offset, digest in enumerate(iter_digest_blob(digest_blob)):
                    seed = start_seed + offset
                    prev = global_table.get(digest)

                    if prev is not None and prev != seed:
                        collision = {
                            "digest_hex": digest.hex(),
                            "seed_a": int(prev),
                            "seed_b": int(seed),
                            "scope": "global",
                        }
                        collisions.append(collision)

                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0.0

                        print("\n!!! GLOBAL COLLISION FOUND !!!")
                        print(json.dumps(collision, indent=2))
                        emit_metrics("collision", completed, seed_max, len(collisions), elapsed, rate, PROFILE, n, workers, chunk_size)
                        return False

                    global_table[digest] = seed

                completed += actual_count
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (seed_max - completed) / rate if rate > 0 else 0.0

                if chunks_done % PROGRESS_EVERY_CHUNKS == 0 or completed >= seed_max:
                    print(
                        f"  - Progress: {completed:,}/{seed_max:,} "
                        f"| chunks={chunks_done}/{len(chunks)} "
                        f"| rate={rate:,.1f}/s "
                        f"| elapsed={elapsed/60:.1f}m "
                        f"| eta={eta/60:.1f}m "
                        f"| last_chunk={result['runtime']:.2f}s",
                        flush=True,
                    )
                    write_progress(
                        completed,
                        seed_max,
                        elapsed,
                        rate,
                        chunks_done,
                        len(chunks),
                        PROFILE,
                        n,
                        workers,
                        chunk_size,
                    )

                if MAX_SECONDS is not None and elapsed >= MAX_SECONDS:
                    print("\n[WARNING] MAX_SECONDS reached. Partial run completed without collisions.")
                    emit_metrics("timeout_partial_pass", completed, seed_max, len(collisions), elapsed, rate, PROFILE, n, workers, chunk_size)
                    return True

    except KeyboardInterrupt:
        print("\n[ERROR] Interrupted by user.")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        print("[ERROR] Collision cluster crashed:", repr(e))
        traceback.print_exc()
        emit_metrics("runner_crash", completed, seed_max, len(collisions), elapsed, rate, PROFILE, n, workers, chunk_size)
        return False

    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0

    print("\n=== Collision Search Result ===")
    print(f"Checked {completed:,} seeds.")
    print("No collisions found in searched range.")
    print(f"Elapsed: {elapsed/60:.2f} minutes")
    print(f"Rate: {rate:,.1f} commitments/sec")

    emit_metrics("pass", completed, seed_max, len(collisions), elapsed, rate, PROFILE, n, workers, chunk_size)
    return True


if __name__ == "__main__":
    try:
        ok = run_sha_collision_cluster()
        sys.exit(0 if ok else 1)
    except KeyboardInterrupt:
        raise
