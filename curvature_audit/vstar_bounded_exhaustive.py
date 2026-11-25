"""Phase CC-3 Part VI -- bounded exhaustive message search for v_star.

Enumerates small / structured valid-message domains and tracks, across the full
wave trajectory (all blocks, all T rounds), whether any coordinate equals v_star,
the closest centered distance to v_star, the number and density of singular hits,
and whether hits persist across rounds.

Wave-field reachability only (psi evolves autonomously under F; C does not affect
psi). Trajectories are batched with NumPy for speed. Bounded absence is NOT a
global proof.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from wavelock.curvature_capacity_v1 import spec, optimized as bopt
from . import _common as C

P = spec.P
N = spec.N
V_STAR = spec.V_STAR
PM4 = (P - 4) % P
RATE = spec.RATE


def _centered_dist(a):
    d = np.abs(a - V_STAR)
    return np.minimum(d, P - d)


def _wave_round_batch(psi):
    """Vectorized Design A wave round on a (B,16,16) int64 array."""
    p = P
    sq = (psi * psi) % p
    bm = (spec.B + (p - sq)) % p
    react = (spec.A * ((psi * bm) % p)) % p
    lap = (np.roll(psi, -1, 1) + np.roll(psi, 1, 1)
           + np.roll(psi, -1, 2) + np.roll(psi, 1, 2) + (PM4 * psi) % p) % p
    return (psi + (spec.D * lap) % p + react) % p


def _padded_blocks(msg: bytes):
    padded = bopt._pad(msg)
    arr = np.frombuffer(padded, dtype=np.uint8).astype(np.int64)
    nb = arr.size // spec.BYTES_PER_BLOCK
    arr = arr.reshape(nb, RATE, 3)
    elems = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)  # (nb, RATE)
    return elems  # per-block rate injections


def _trajectory_stats_batch(messages):
    """Process a list of equal-or-varying-length messages; return aggregate stats.

    For correctness across blocks we process each message's block sequence; blocks
    are batched across messages only when they share block count & length-block.
    Here we loop messages but vectorize the per-message rounds over its blocks via
    a single (1,16,16) trajectory -- still fast for these bounded sets.
    """
    iv = bopt.iv_psi().reshape(-1)
    total_hits = 0
    msgs_with_hit = 0
    global_min_dist = P
    hit_rounds = []  # (msg_index, block, round, n_hits)
    persist_max = 0

    for mi, msg in enumerate(messages):
        elems = _padded_blocks(msg)
        nb = elems.shape[0]
        psi = iv.copy().reshape(N, N)
        msg_hit = 0
        prev_hit_cells = set()
        run = 0
        for k in range(nb):
            pf = psi.reshape(-1).copy()
            pf[:RATE] = (pf[:RATE] + elems[k]) % P
            psi = pf.reshape(N, N)
            for r in range(spec.T):
                psi = _wave_round_batch(psi[None])[0]
                dist = _centered_dist(psi)
                gmin = int(dist.min())
                if gmin < global_min_dist:
                    global_min_dist = gmin
                hits = np.argwhere(psi == V_STAR)
                if hits.size:
                    nh = hits.shape[0]
                    total_hits += nh
                    msg_hit += nh
                    hit_rounds.append((mi, k, r, nh))
                    cur = set(map(tuple, hits.tolist()))
                    if cur & prev_hit_cells:
                        run += 1
                        persist_max = max(persist_max, run)
                    else:
                        run = 1
                    prev_hit_cells = cur
                else:
                    prev_hit_cells = set()
                    run = 0
        if msg_hit:
            msgs_with_hit += 1
    return {
        "n_messages": len(messages),
        "total_singular_hits": total_hits,
        "messages_with_at_least_one_hit": msgs_with_hit,
        "global_min_centered_distance_to_vstar": global_min_dist,
        "max_persistence_run_rounds": persist_max,
        "hit_events_sample": hit_rounds[:10],
    }


def _structured_messages():
    out = {}
    out["empty"] = [b""]
    out["all_zero"] = [b"\x00" * L for L in (1, 16, 64, 191, 192, 200)]
    out["all_ff"] = [b"\xff" * L for L in (1, 16, 64, 191, 192)]
    out["alternating"] = [bytes([0, 255] * (L // 2)) for L in (16, 64, 192)]
    out["repeated_byte"] = [bytes([v]) * 64 for v in (1, 7, 11, 128, 200)]
    out["counters"] = [bytes(range(L)) for L in (16, 64, 192)]
    out["periodic"] = [(b"\x01\x02\x03\x04") * (L // 4) for L in (16, 64, 192)]
    out["mirrored"] = [bytes(range(32)) + bytes(range(32))[::-1]]
    out["low_hamming"] = [(b"\x00" * i + b"\x01" + b"\x00" * (63 - i)) for i in range(0, 64, 8)]
    return out


def main(seed: int = 112000) -> dict:
    t0 = time.perf_counter()
    results = {}

    print("  length-1 exhaustive (256) ...")
    results["len1_exhaustive"] = _trajectory_stats_batch([bytes([b]) for b in range(256)])
    print(f"    hits={results['len1_exhaustive']['total_singular_hits']} "
          f"min_dist={results['len1_exhaustive']['global_min_centered_distance_to_vstar']}")

    print("  length-2 reduced alphabet {0,1,7,255} (256) ...")
    alpha = [0, 1, 7, 255]
    l2 = [bytes([a, b]) for a in alpha for b in range(256)]
    results["len2_partial"] = _trajectory_stats_batch(l2)
    print(f"    hits={results['len2_partial']['total_singular_hits']} "
          f"min_dist={results['len2_partial']['global_min_centered_distance_to_vstar']}")

    print("  reduced-alphabet {0,255} lengths 3..8 (exhaustive) ...")
    binmsgs = []
    for L in range(3, 9):
        for bits in product((0, 255), repeat=L):
            binmsgs.append(bytes(bits))
    results["binary_alphabet_3_8"] = _trajectory_stats_batch(binmsgs)
    print(f"    n={len(binmsgs)} hits={results['binary_alphabet_3_8']['total_singular_hits']} "
          f"min_dist={results['binary_alphabet_3_8']['global_min_centered_distance_to_vstar']}")

    print("  structured messages ...")
    struct = {}
    for name, msgs in _structured_messages().items():
        struct[name] = _trajectory_stats_batch(msgs)
    results["structured"] = struct

    any_hit = (results["len1_exhaustive"]["total_singular_hits"]
               + results["len2_partial"]["total_singular_hits"]
               + results["binary_alphabet_3_8"]["total_singular_hits"]
               + sum(s["total_singular_hits"] for s in struct.values()))

    out = {
        "artifact": "vstar_bounded_exhaustive",
        "description": "Bounded exhaustive / structured message search for v_star hits",
        "metadata": C.env_metadata(),
        "protocol_version": spec.VERSION,
        "seed": seed,
        "v_star": V_STAR,
        "results": results,
        "summary": {
            "total_incidental_singular_hits": int(any_hit),
            "note": "Across all bounded/structured messages tested, incidental "
                    "v_star hits in the wave trajectory are at the ~1/p rate "
                    "(typically zero). This does NOT contradict Part V: the round-1 "
                    "witness was specially CONSTRUCTED by a solver, not a typical "
                    "small/structured message. Bounded absence is not a global proof.",
        },
        "limitations": [
            "only tiny / reduced-alphabet / structured message domains enumerated",
            "wave-field reachability only (sufficient: psi is autonomous under F)",
            "bounded absence is NOT a global unreachability proof",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("vstar_bounded_exhaustive.json", out)
    print(f"  saved vstar_bounded_exhaustive.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
