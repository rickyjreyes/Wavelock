"""Shared helpers for the curvature-capacity audit suite.

Provenance capture, artifact I/O, the lifting convention, deterministic message
generators, and a builder for the Design A eigenmode collision family (so the
audit can test whether those state-level collisions survive the path commitment).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time

import numpy as np

from wavelock.curvature_capacity import spec

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

P = spec.P
N = spec.N


# --- provenance ---------------------------------------------------------
def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def env_metadata() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "branch": git_branch(),
        "commit": git_commit(),
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def save_artifact(name: str, obj: dict) -> str:
    path = os.path.join(ARTIFACT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
    return path


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_messages(seed: int, count: int, min_len: int = 1, max_len: int = 400):
    g = rng(seed)
    return [g.integers(0, 256, size=int(g.integers(min_len, max_len + 1)),
                       dtype=np.uint8).tobytes() for _ in range(count)]


# --- lifting convention -------------------------------------------------
# Curvature/heat functionals are defined on *lifted integer representatives*.
# Convention CENTERED: map a residue r in [0, P) to the signed representative
# in (-P/2, P/2]:  lift(r) = r - P if r > P//2 else r. This is the unique
# minimal-magnitude representative and is the convention used everywhere unless
# a functional explicitly says otherwise. Sensitivity to this choice is tested
# in curvature_metrics.lifting_sensitivity().
def lift_centered(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.int64) % P
    return np.where(a > P // 2, a - P, a)


def lift_naive(arr: np.ndarray) -> np.ndarray:
    """Alternative lifting: representative in [0, P) (no centering)."""
    return np.asarray(arr, dtype=np.int64) % P


# --- Design A eigenmode collision family --------------------------------
def _amplitude_for_r(r: int):
    ainv = pow(spec.A, P - 2, P)
    s2 = int((spec.B - (2 * r * spec.D - 1) * ainv) % P)
    root = pow(s2, (P + 1) // 4, P)
    if (root * root) % P != s2:
        return None
    return s2, root, (P - root) % P


def eigenmode_states() -> dict:
    """Return {name: (N,N) int64 state} for the verified Design A zero-preimages.

    These all satisfy F(state) == 0 (the Design A one-round map sends them to the
    all-zero state); the zero state is itself a fixed point.
    """
    ii, jj = np.indices((N, N))
    sigs = {
        "checker_r4": ((-1) ** (ii + jj), 4),
        "rows_r2": ((-1) ** ii, 2),
        "cols_r2": ((-1) ** jj, 2),
        "p4cols_r1": (np.array([1, -1, -1, 1])[jj % 4], 1),
    }
    out = {}
    for name, (sig, r) in sigs.items():
        amp = _amplitude_for_r(r)
        if amp is None:
            continue
        _, sp, sm = amp
        out[name + "+"] = ((sp * sig) % P).astype(np.int64)
        out[name + "-"] = ((sm * sig) % P).astype(np.int64)
    out["zero"] = np.zeros((N, N), dtype=np.int64)
    return out


def hamming_bytes(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def now() -> float:
    return time.perf_counter()
