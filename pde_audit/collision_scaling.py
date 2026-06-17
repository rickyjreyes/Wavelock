"""Phase 8F: truncated-output collision scaling and multicollisions.

Full 256-bit collision search is infeasible, so we measure whether truncated
outputs collide at the birthday rate (structural-weakness detector). For each
truncation size we run multiple deterministic trials, record evaluations to the
first collision, and compare to the birthday expectation
E[evals] ~ 1.2533 * sqrt(2**n). Repeated for reduced round counts. Colliding
messages are preserved. Also searches for a 3-way multicollision and a
chosen-prefix collision on a reduced-round variant.

Operates on raw (optionally truncated) output only.

Run:
    python -m pde_audit.collision_scaling
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams

TRUNC_BITS = [8, 12, 16, 20, 24, 28, 32, 36, 40]
ROUND_VARIANTS = [4, 8, 32]   # 32 = normative


# pool sizes per T (the only hashing cost; larger T is slower)
POOL_SIZE = {4: 220_000, 8: 100_000, 32: 30_000}


def _msg(i: int) -> bytes:
    return i.to_bytes(8, "big")


def _build_pool(variant: PDEVariant, start: int, K: int) -> np.ndarray:
    """Top-40-bit digest of messages start..start+K-1 as a uint64 array."""
    pool = np.empty(K, dtype=np.uint64)
    for e in range(K):
        d = variant.hash(_msg(start + e))
        pool[e] = int.from_bytes(d[:5], "big")   # top 40 bits
    return pool


def _first_collision_in(slice40: np.ndarray, nbits: int):
    """First-collision index within a slice of 40-bit digests, truncated to
    nbits. Returns (evals, local_i, local_j) or None."""
    shift = 40 - nbits
    seen = {}
    for e in range(slice40.size):
        v = int(slice40[e]) >> shift
        if v in seen:
            return e + 1, seen[v], e
        seen[v] = e
    return None


def scaling_for_T(T: int, seed: int) -> dict:
    variant = PDEVariant(PDEParams(T=T))
    K = POOL_SIZE[T]
    start = (seed % 1000) * 10_000_000
    pool = _build_pool(variant, start, K)
    out = {}
    for nbits in TRUNC_BITS:
        expected = 1.2533 * math.sqrt(2 ** nbits)
        # split the pool into trial slices ~ 4x birthday (so a collision is
        # likely within each slice); at least 1 slice, at most 8.
        slice_len = int(min(max(4 * expected, 64), K))
        n_trials = max(1, min(8, K // slice_len))
        evals, example = [], None
        for t in range(n_trials):
            sl = pool[t * slice_len:(t + 1) * slice_len]
            res = _first_collision_in(sl, nbits)
            if res is None:
                continue
            e, li, lj = res
            evals.append(e)
            if example is None:
                example = {"msg_a": _msg(start + t * slice_len + li).hex(),
                           "msg_b": _msg(start + t * slice_len + lj).hex(),
                           "trunc_bits": nbits}
        if evals:
            arr = np.array(evals, dtype=float)
            out[str(nbits)] = {
                "trials": n_trials, "collisions_found": len(evals),
                "slice_len": slice_len,
                "evals_mean": float(arr.mean()), "evals_median": float(np.median(arr)),
                "evals_std": float(arr.std()),
                "birthday_expected": expected,
                "ratio_observed_over_expected": float(arr.mean() / expected),
                "example": example,
            }
        else:
            out[str(nbits)] = {
                "trials": n_trials, "collisions_found": 0, "slice_len": slice_len,
                "birthday_expected": expected,
                "note": "no collision within pool (birthday > available samples)",
            }
    out["_pool_size"] = K
    return out


def multicollision(seed: int, nbits: int = 16, want: int = 3) -> dict:
    """Find a truncated value with >= `want` distinct preimages."""
    variant = PDEVariant(PDEParams())     # normative T
    buckets = {}
    cursor = seed * 7_000_000
    budget = 40_000
    shift = 40 - nbits
    for e in range(budget):
        idx = cursor + e
        v = int.from_bytes(variant.hash(_msg(idx))[:5], "big") >> shift
        buckets.setdefault(v, []).append(idx)
        if len(buckets[v]) >= want:
            return {"trunc_bits": nbits, "multiplicity": want, "evals": e + 1,
                    "messages": [_msg(x).hex() for x in buckets[v]]}
    biggest = max(buckets.values(), key=len)
    return {"trunc_bits": nbits, "multiplicity_found": len(biggest),
            "evals": budget, "note": "target multiplicity not reached in budget"}


def chosen_prefix(seed: int, nbits: int = 24, T: int = 8) -> dict:
    """Two messages sharing a fixed prefix colliding on truncated output
    (reduced-round variant)."""
    variant = PDEVariant(PDEParams(T=T))
    prefix = b"CHOSEN-PREFIX::"
    seen = {}
    budget = 120_000
    cursor = seed * 3_000_000
    shift = 40 - nbits
    for e in range(budget):
        idx = cursor + e
        m = prefix + idx.to_bytes(8, "big")
        v = int.from_bytes(variant.hash(m)[:5], "big") >> shift
        if v in seen:
            return {"trunc_bits": nbits, "T": T, "evals": e + 1,
                    "prefix": prefix.hex(),
                    "msg_a": (prefix + seen[v].to_bytes(8, "big")).hex(),
                    "msg_b": m.hex()}
        seen[v] = idx
    return {"trunc_bits": nbits, "T": T, "evals": budget,
            "note": "no chosen-prefix collision within budget"}


def main(seed: int = 80060) -> dict:
    t0 = time.perf_counter()
    results = {
        "phase": "8F_collision_scaling",
        "metadata": H.env_metadata(),
        "seed": seed,
        "truncation_bits": TRUNC_BITS,
        "by_T": {},
    }
    for T in ROUND_VARIANTS:
        tt = time.perf_counter()
        sc = scaling_for_T(T, seed + T)
        results["by_T"][str(T)] = sc
        ratios = [v.get("ratio_observed_over_expected") for k, v in sc.items()
                  if isinstance(v, dict) and v.get("collisions_found")]
        ratios = [r for r in ratios if r is not None]
        print(f"  T={T:>2}: ratios(obs/birthday) for found collisions: "
              f"{[round(r,2) for r in ratios]}  ({round(time.perf_counter()-tt,1)}s)")
    results["multicollision"] = multicollision(seed + 1)
    print(f"  multicollision(16-bit, want 3): {results['multicollision'].get('evals')} evals "
          f"mult={results['multicollision'].get('multiplicity', results['multicollision'].get('multiplicity_found'))}")
    results["chosen_prefix"] = chosen_prefix(seed + 2)
    print(f"  chosen-prefix(24-bit,T=8): evals={results['chosen_prefix']['evals']}")
    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    path = H.save_artifact("phase8f_collision_scaling.json", results)
    print("  saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
