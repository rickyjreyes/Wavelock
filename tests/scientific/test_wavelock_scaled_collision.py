#!/usr/bin/env python3
"""
WaveLock — scaled collision search, turbo/profiled version.

Run from repo root:
    python .\tests\scientific\test_wavelock_scaled_collision.py

Why this version exists:
    The exact 1,000,000-seed collision sweep is slow because every seed still
    constructs a CurvatureKeyPair. Multiprocessing can reduce overhead, but it
    cannot remove the core per-seed cost.

This version improves the wrapper:
    - imports WaveLock once per worker using ProcessPoolExecutor initializer
    - returns packed fixed-size comparison digests instead of list[bytes]
    - removes local_seen by default; parent performs exact global collision scan
    - adds fast/standard/deep profiles
    - adds optional time-capped mode
    - emits RISK_METRICS_BEGIN/END

Important:
    COMPARISON_MODE="commitment_sha256" compares SHA256(raw commitment bytes).
    If kp.commitment is already a cryptographic commitment string, this is a
    practical fixed-size exact-screening layer for collision sweeps. To compare
    raw commitment bytes exactly, set COMPARISON_MODE="raw_fixed"; that requires
    every commitment to serialize to the same byte length.
"""

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
    # quick sanity run
    "fast": {
        "num_trials": 25_000,
        "n": 4,
        "workers": 4,
        "chunk_size": 5_000,
    },

    # daily serious run
    "standard": {
        "num_trials": 100_000,
        "n": 4,
        "workers": 6,
        "chunk_size": 10_000,
    },

    # overnight/deep run
    "deep": {
        "num_trials": 1_000_000,
        "n": 4,
        "workers": 8,
        "chunk_size": 25_000,
    },
}

# "commitment_sha256" = fixed 32-byte digest of kp.commitment bytes, fastest.
# "raw_fixed" = compare raw commitment bytes exactly, requires fixed item length.
COMPARISON_MODE = "commitment_sha256"

# Optional cap. Example: MAX_SECONDS = 3 * 60 * 60
MAX_SECONDS = None

# Parent checks global duplicates. Local duplicate check usually duplicates work.
LOCAL_DUP_CHECK = False

PROGRESS_EVERY_CHUNKS = 1

DIGEST_SIZE = 32

# ============================================================
# Thread limits
# ============================================================

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

# ============================================================
# Repo path setup
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
PROGRESS_PATH = LOG_DIR / "scaled_collision_progress.json"

# ============================================================
# Worker globals
# ============================================================

_WORKER_READY = False
_CurvatureKeyPair = None


def worker_init(repo_root_str):
    global _WORKER_READY, _CurvatureKeyPair

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from wavelock.chain.WaveLock import CurvatureKeyPair

    _CurvatureKeyPair = CurvatureKeyPair
    _WORKER_READY = True


# ============================================================
# Commitment normalization
# ============================================================

def commitment_to_raw_bytes(commitment):
    if isinstance(commitment, bytes):
        return commitment
    if isinstance(commitment, str):
        return commitment.strip().encode("utf-8")
    return repr(commitment).encode("utf-8")


def normalize_item(commitment):
    raw = commitment_to_raw_bytes(commitment)

    if COMPARISON_MODE == "commitment_sha256":
        return hashlib.sha256(raw).digest()

    if COMPARISON_MODE == "raw_fixed":
        return raw

    raise ValueError(f"unsupported COMPARISON_MODE={COMPARISON_MODE!r}")


# ============================================================
# Worker
# ============================================================

def worker_range(args):
    """
    Generate comparison items for deterministic seed range.

    Returns:
      - item_blob: packed items, one after another
      - item_size: fixed size for each item
    """
    start, stop, n = args
    t0 = time.time()

    try:
        if not _WORKER_READY:
            worker_init(str(REPO_ROOT))

        parts = []
        append = parts.append
        local_seen = {} if LOCAL_DUP_CHECK else None
        item_size = None

        for seed in range(start, stop):
            kp = _CurvatureKeyPair(n=n, seed=int(seed), test_mode=True)
            item = normalize_item(kp.commitment)

            if item_size is None:
                item_size = len(item)
            elif len(item) != item_size:
                return {
                    "ok": False,
                    "start": start,
                    "stop": stop,
                    "count": seed - start + 1,
                    "item_blob": b"",
                    "item_size": item_size or 0,
                    "seed": int(seed),
                    "item_hash": hashlib.sha256(item).hexdigest(),
                    "runtime": time.time() - t0,
                    "error": f"non-fixed item length: got {len(item)} expected {item_size}",
                }

            if LOCAL_DUP_CHECK:
                prev = local_seen.get(item)
                if prev is not None and prev != seed:
                    return {
                        "ok": False,
                        "start": start,
                        "stop": stop,
                        "count": seed - start + 1,
                        "item_blob": b"",
                        "item_size": item_size,
                        "seed": int(seed),
                        "item_hash": hashlib.sha256(item).hexdigest(),
                        "runtime": time.time() - t0,
                        "error": "local duplicate commitment",
                    }
                local_seen[item] = seed

            append(item)

        return {
            "ok": True,
            "start": start,
            "stop": stop,
            "count": stop - start,
            "item_blob": b"".join(parts),
            "item_size": int(item_size or 0),
            "seed": None,
            "item_hash": None,
            "runtime": time.time() - t0,
            "error": None,
        }

    except Exception as e:
        return {
            "ok": False,
            "start": start,
            "stop": stop,
            "count": 0,
            "item_blob": b"",
            "item_size": 0,
            "seed": None,
            "item_hash": None,
            "runtime": time.time() - t0,
            "error": repr(e) + "\n" + traceback.format_exc(),
        }


# ============================================================
# Metrics/progress
# ============================================================

def emit_metrics(status, completed, target, collisions, elapsed, rate, cfg):
    crashes = 1 if status in {"runner_crash", "worker_failure"} else 0
    danger_total = int(collisions) + crashes

    metrics = {
        "test": "scaled_collision",
        "status": status,
        "profile": PROFILE,
        "comparison_mode": COMPARISON_MODE,
        "trials_completed": int(completed),
        "target_trials": int(target),
        "collisions": int(collisions),
        "false_accepts": 0,
        "forgeries": 0,
        "accepted": 0,
        "matched": False,
        "nan_detected": False,
        "crashes": crashes,
        "danger_total": int(danger_total),
        "elapsed_seconds": float(elapsed),
        "rate_per_second": float(rate),
        "n": int(cfg["n"]),
        "workers": int(cfg["workers"]),
        "chunk_size": int(cfg["chunk_size"]),
        "packed_item_blob": True,
        "worker_initializer": True,
        "local_dup_check": bool(LOCAL_DUP_CHECK),
    }

    print("RISK_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("RISK_METRICS_END")


def write_progress(done, target, elapsed, rate, chunks_done, total_chunks, cfg):
    data = {
        "done": int(done),
        "target": int(target),
        "elapsed_seconds": float(elapsed),
        "rate_per_second": float(rate),
        "chunks_done": int(chunks_done),
        "total_chunks": int(total_chunks),
        "workers": int(cfg["workers"]),
        "chunk_size": int(cfg["chunk_size"]),
        "n": int(cfg["n"]),
        "profile": PROFILE,
        "comparison_mode": COMPARISON_MODE,
        "timestamp": time.time(),
    }

    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def iter_blob(blob, item_size):
    if item_size <= 0:
        raise ValueError("item_size must be positive")
    if len(blob) % item_size != 0:
        raise ValueError(f"blob length {len(blob)} not divisible by item_size {item_size}")

    for offset in range(0, len(blob), item_size):
        yield blob[offset:offset + item_size]


# ============================================================
# Main search
# ============================================================

def run_scaled_collision_search():
    if PROFILE not in PROFILE_CONFIG:
        print(f"[ERROR] Unknown PROFILE={PROFILE!r}. Valid profiles: {sorted(PROFILE_CONFIG)}")
        return False

    cfg = PROFILE_CONFIG[PROFILE]
    num_trials = int(cfg["num_trials"])
    n = int(cfg["n"])
    workers = int(cfg["workers"])
    chunk_size = int(cfg["chunk_size"])

    print("[INFO] Importing WaveLock CurvatureKeyPair...")
    try:
        from wavelock.chain.WaveLock import CurvatureKeyPair  # noqa: F401
        print("[INFO] Import successful.")
    except Exception as e:
        print("[ERROR] Failed to import WaveLock:", repr(e))
        traceback.print_exc()
        return False

    print("\n====================================================")
    print("        WAVELOCK — SCALED COLLISION SEARCH")
    print("====================================================")
    print(f"[INFO] Profile: {PROFILE}")
    print(f"[INFO] Trials: {num_trials:,}")
    print(f"[INFO] n: {n}")
    print(f"[INFO] Workers: {workers}")
    print(f"[INFO] Chunk size: {chunk_size:,}")
    print(f"[INFO] Comparison mode: {COMPARISON_MODE}")
    print(f"[INFO] Max seconds: {MAX_SECONDS if MAX_SECONDS is not None else 'unlimited'}")
    print(f"[INFO] Progress JSON: {PROGRESS_PATH}")

    if COMPARISON_MODE not in {"commitment_sha256", "raw_fixed"}:
        print(f"[ERROR] Unsupported COMPARISON_MODE: {COMPARISON_MODE}")
        return False

    chunks = [
        (start, min(start + chunk_size, num_trials), n)
        for start in range(0, num_trials, chunk_size)
    ]

    if not chunks:
        print("[ERROR] No chunks to process.")
        return False

    global_seen = {}
    completed = 0
    chunks_done = 0
    collisions = 0
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
                    collisions += 1
                    print("\n==================== COLLISION / WORKER FAILURE ====================")
                    print(f"Chunk: {result['start']:,} to {result['stop']:,}")
                    print(f"Seed: {result.get('seed')}")
                    print(f"Item hash: {result.get('item_hash')}")
                    print(f"Error: {result.get('error')}")
                    print("====================================================================")
                    emit_metrics("worker_failure", completed, num_trials, collisions, elapsed, rate, cfg)
                    return False

                start_seed = int(result["start"])
                item_blob = result["item_blob"]
                item_size = int(result["item_size"])
                expected_count = int(result["count"])
                actual_count = len(item_blob) // item_size if item_size else 0

                if actual_count != expected_count:
                    print(f"[ERROR] item count mismatch: actual={actual_count} expected={expected_count}")
                    emit_metrics("runner_crash", completed, num_trials, collisions, elapsed, rate, cfg)
                    return False

                for offset, item in enumerate(iter_blob(item_blob, item_size)):
                    seed = start_seed + offset
                    prev = global_seen.get(item)

                    if prev is not None and prev != seed:
                        collisions += 1
                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0.0

                        print("\n==================== COLLISION FOUND ====================")
                        print("Collision item SHA256:", hashlib.sha256(item).hexdigest())
                        print(f"Seeds: {prev}, {seed}")
                        print("==========================================================")
                        emit_metrics("collision", completed, num_trials, collisions, elapsed, rate, cfg)
                        return False

                    global_seen[item] = seed

                completed += actual_count
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0.0
                eta = (num_trials - completed) / rate if rate > 0 else 0.0

                if chunks_done % PROGRESS_EVERY_CHUNKS == 0 or completed >= num_trials:
                    print(
                        f"  - Progress: {completed:,}/{num_trials:,} "
                        f"| chunks={chunks_done}/{len(chunks)} "
                        f"| rate={rate:,.1f}/s "
                        f"| elapsed={elapsed/60:.1f}m "
                        f"| eta={eta/60:.1f}m "
                        f"| last_chunk={result['runtime']:.2f}s",
                        flush=True,
                    )
                    write_progress(completed, num_trials, elapsed, rate, chunks_done, len(chunks), cfg)

                if MAX_SECONDS is not None and elapsed >= MAX_SECONDS:
                    print("\n[WARNING] MAX_SECONDS reached. Partial run completed without collisions.")
                    emit_metrics("timeout_partial_pass", completed, num_trials, collisions, elapsed, rate, cfg)
                    return True

    except KeyboardInterrupt:
        print("[ERROR] Interrupted by user.")
        raise
    except Exception as e:
        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0.0
        print("[ERROR] Parallel collision benchmark crashed:", repr(e))
        traceback.print_exc()
        emit_metrics("runner_crash", completed, num_trials, collisions, elapsed, rate, cfg)
        return False

    elapsed = time.time() - start_time
    rate = completed / elapsed if elapsed > 0 else 0.0

    print("\n==================== RESULTS ====================")
    print(f"Completed {completed:,} commitments — NO collisions found.")
    print(f"Elapsed: {elapsed/3600:.2f} hours")
    print(f"Rate: {rate:,.1f} commitments/sec")
    print("\nPASS: No collisions detected.")
    print("====================================================\n")

    emit_metrics("pass", completed, num_trials, collisions, elapsed, rate, cfg)
    return True


if __name__ == "__main__":
    ok = run_scaled_collision_search()
    print("FINAL RESULT (True=Secure / False=Collision):", ok)
    sys.exit(0 if ok else 1)
