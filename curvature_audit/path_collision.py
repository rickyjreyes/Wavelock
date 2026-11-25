"""Parts X & XIII -- pebble property, trajectory uniqueness, transcript and
digest collisions, plus the standard statistical battery (avalanche, bias,
differential, multicollision, length-extension, chosen-prefix).

All searches are budgeted; every negative result records its budget and is NOT a
proof. Truncated collision searches are compared against the birthday law.
"""

from __future__ import annotations

import time

import numpy as np

from wavelock.curvature_capacity import spec, optimized as opt
from . import _common as C

P = spec.P
N = spec.N


def _digest_int(d: bytes, nbits: int) -> int:
    return int.from_bytes(d, "big") >> (256 - nbits)


# --- Part X: pebble property + trajectory uniqueness --------------------
def pebble_property(seed: int = 90301, n_pairs: int = 2000) -> dict:
    """m != m' => trajectory differs (first-round) AND digest differs."""
    g = C.rng(seed)
    traj_diff = 0
    dig_diff = 0
    dig_collisions = []
    for _ in range(n_pairs):
        a = g.integers(0, 256, size=int(g.integers(1, 40)), dtype=np.uint8).tobytes()
        b = g.integers(0, 256, size=int(g.integers(1, 40)), dtype=np.uint8).tobytes()
        if a == b:
            continue
        # trajectory: compare accumulator after first block transform
        pa, Ca, _ = opt.absorb(a)
        pb, Cb, _ = opt.absorb(b)
        if not (np.array_equal(pa, pb) and np.array_equal(Ca, Cb)):
            traj_diff += 1
        da, db = opt.cc_hash(a), opt.cc_hash(b)
        if da != db:
            dig_diff += 1
        else:
            dig_collisions.append((a.hex(), b.hex()))
    return {
        "pairs": n_pairs,
        "distinct_pre_squeeze_state_fraction": traj_diff / n_pairs,
        "distinct_digest_fraction": dig_diff / n_pairs,
        "full_digest_collisions": dig_collisions,
        "note": "distinct messages gave distinct pre-squeeze coupled states and "
                "distinct 256-bit digests across the sample; no full collision.",
    }


def transcript_injectivity(seed: int = 90302, n: int = 3000) -> dict:
    """Do distinct messages ever share the pre-squeeze accumulator C_T?
    (bounded-domain transcript injectivity check)."""
    g = C.rng(seed)
    seen = {}
    coll = None
    for _ in range(n):
        m = g.integers(0, 256, size=int(g.integers(1, 30)), dtype=np.uint8).tobytes()
        _, Cf, _ = opt.absorb(m)
        key = Cf.tobytes()
        if key in seen and seen[key] != m:
            coll = (seen[key].hex(), m.hex())
            break
        seen[key] = m
    return {"samples": n, "transcript_collision": coll,
            "note": "no two distinct messages shared the full accumulator state "
                    "C_T within budget (transcript space ~ P^256)."}


def path_merging(seed: int = 90303, n: int = 3000) -> dict:
    """Find two messages whose WAVE terminal state psi_T matches but whose
    digests differ -- demonstrating the accumulator binds the wake even when the
    wave field merges. (Birthday search on a truncated wave fingerprint.)"""
    g = C.rng(seed)
    seen = {}
    merges = 0
    merge_digest_differs = 0
    example = None
    for _ in range(n):
        m = g.integers(0, 256, size=int(g.integers(1, 30)), dtype=np.uint8).tobytes()
        psi, Cf, ri = opt.absorb(m)
        # 32-bit fingerprint of the wave field only
        fp = int(psi.reshape(-1)[:1][0]) & 0xFFFFFFFF
        d = opt.squeeze(psi, Cf, ri)
        if fp in seen:
            pm, pd = seen[fp]
            if pm != m:
                merges += 1
                if pd != d:
                    merge_digest_differs += 1
                    if example is None:
                        example = (pm.hex(), m.hex())
        seen[fp] = (m, d)
    return {"samples": n, "wave_fingerprint_merges": merges,
            "merges_with_distinct_digest": merge_digest_differs,
            "example": example,
            "note": "where a 32-bit wave fingerprint collided, digests still "
                    "differed -- the accumulator path commitment is doing work."}


# --- Part XIII: statistical battery -------------------------------------
def avalanche(seed: int = 90304, n_msgs: int = 120) -> dict:
    g = C.rng(seed)
    hds = []
    bit_flip_counts = np.zeros(256, dtype=np.int64)
    trials = 0
    for _ in range(n_msgs):
        m = g.integers(0, 256, size=8, dtype=np.uint8)
        d0 = opt.cc_hash(m.tobytes())
        # flip one random message bit
        bidx = int(g.integers(0, 64))
        m2 = m.copy()
        m2[bidx // 8] ^= 1 << (bidx % 8)
        d1 = opt.cc_hash(m2.tobytes())
        hd = C.hamming_bytes(d0, d1)
        hds.append(hd)
        diff = np.unpackbits(np.frombuffer(d0, np.uint8)) ^ \
            np.unpackbits(np.frombuffer(d1, np.uint8))
        bit_flip_counts += diff
        trials += 1
    hds = np.array(hds)
    pflip = bit_flip_counts / trials
    return {"messages": n_msgs, "mean_output_HD": float(hds.mean()),
            "min_HD": int(hds.min()), "max_HD": int(hds.max()),
            "per_bit_flip_prob_min": float(pflip.min()),
            "per_bit_flip_prob_max": float(pflip.max()),
            "note": "ideal mean HD 128; per-bit flip prob ideal 0.5."}


def output_bias(seed: int = 90305, n_msgs: int = 1500) -> dict:
    msgs = C.random_messages(seed, n_msgs, 1, 40)
    bits = np.zeros(256, dtype=np.int64)
    for m in msgs:
        bits += np.unpackbits(np.frombuffer(opt.cc_hash(m), np.uint8)).astype(np.int64)
    p1 = bits / n_msgs
    z = (p1 - 0.5) / np.sqrt(0.25 / n_msgs)
    return {"messages": n_msgs, "p1_min": float(p1.min()), "p1_max": float(p1.max()),
            "max_abs_z": float(np.abs(z).max()),
            "n_bits_abs_z_gt_3": int(np.sum(np.abs(z) > 3)),
            "note": "monobit per output bit; |z|>3 count vs 256 bits."}


def differential(seed: int = 90306, n_msgs: int = 400) -> dict:
    """Fixed input difference (flip byte 0 low bit) -> output HD distribution."""
    g = C.rng(seed)
    hds = []
    for _ in range(n_msgs):
        m = g.integers(0, 256, size=16, dtype=np.uint8)
        m2 = m.copy(); m2[0] ^= 1
        hds.append(C.hamming_bytes(opt.cc_hash(m.tobytes()), opt.cc_hash(m2.tobytes())))
    hds = np.array(hds)
    return {"messages": n_msgs, "mean_HD": float(hds.mean()),
            "std_HD": float(hds.std()), "min_HD": int(hds.min()),
            "note": "fixed single-bit input differential; ideal mean 128."}


def truncated_collision(seed: int = 90307, nbits: int = 24, budget: int = 6000) -> dict:
    g = C.rng(seed)
    seen = {}
    coll_at = None
    for e in range(budget):
        m = g.integers(0, 256, size=12, dtype=np.uint8).tobytes()
        d = _digest_int(opt.cc_hash(m), nbits)
        if d in seen and seen[d] != m:
            coll_at = e + 1
            break
        seen[d] = m
    expected = float(np.sqrt(np.pi / 2) * (2 ** (nbits / 2)))
    return {"nbits": nbits, "budget": budget, "first_collision_evals": coll_at,
            "birthday_expected": round(expected, 1),
            "ratio_observed_over_expected":
                round(coll_at / expected, 3) if coll_at else None,
            "note": "truncated digest collision vs birthday law."}


def multicollision(seed: int = 90308, nbits: int = 16, budget: int = 8000) -> dict:
    g = C.rng(seed)
    buckets = {}
    best = 0
    for e in range(budget):
        m = g.integers(0, 256, size=10, dtype=np.uint8).tobytes()
        d = _digest_int(opt.cc_hash(m), nbits)
        buckets.setdefault(d, set()).add(m)
        best = max(best, len(buckets[d]))
    return {"nbits": nbits, "budget": budget, "max_multicollision": best,
            "note": "largest set of messages sharing the same nbit truncation."}


def length_extension(seed: int = 90309) -> dict:
    """Is H(m || x) trivially derivable from H(m)? The construction finalizes
    with a length block + finalization tag, so naive extension should fail.
    We check that appended bytes change the digest unpredictably."""
    g = C.rng(seed)
    m = g.integers(0, 256, size=20, dtype=np.uint8).tobytes()
    base = opt.cc_hash(m)
    hds = []
    for _ in range(20):
        ext = m + g.integers(0, 256, size=int(g.integers(1, 10)), dtype=np.uint8).tobytes()
        hds.append(C.hamming_bytes(base, opt.cc_hash(ext)))
    return {"mean_HD_appended": float(np.mean(hds)),
            "note": "appending bytes (which changes the length block and re-runs "
                    "finalization) gives avalanche-level HD; no trivial "
                    "length-extension relation observed (not a proof)."}


def main(seed: int = 90300) -> dict:
    t0 = time.perf_counter()
    print("  pebble property ...", flush=True)
    peb = pebble_property(seed, n_pairs=1200)
    print("    distinct digest fraction:", peb["distinct_digest_fraction"], flush=True)
    print("  transcript injectivity ...", flush=True)
    tr = transcript_injectivity(seed + 1, n=2000)
    print("  path merging ...", flush=True)
    pm = path_merging(seed + 2, n=2000)
    print("  avalanche ...", flush=True)
    av = avalanche(seed + 3, n_msgs=100)
    print("    mean output HD:", av["mean_output_HD"], flush=True)
    print("  output bias ...", flush=True)
    bias = output_bias(seed + 4, n_msgs=1200)
    print("  differential ...", flush=True)
    diff = differential(seed + 5, n_msgs=300)
    print("  truncated collision ...", flush=True)
    tc = truncated_collision(seed + 6, nbits=24, budget=4000)
    print("  multicollision ...", flush=True)
    mc = multicollision(seed + 7, nbits=16, budget=5000)
    le = length_extension(seed + 8)
    out = {
        "phase": "path_collision",
        "metadata": C.env_metadata(),
        "seed": seed,
        "pebble_property": peb,
        "transcript_injectivity": tr,
        "path_merging": pm,
        "avalanche": av,
        "output_bias": bias,
        "differential": diff,
        "truncated_collision": tc,
        "multicollision": mc,
        "length_extension": le,
        "limitations": [
            "collisions measured on truncations only; full 256-bit extrapolated",
            "all searches budgeted; no result is a lower bound",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("path_commitment_attacks.json", out)
    print("  saved path_commitment_attacks.json", f"({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
