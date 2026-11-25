"""Phase 8A: diffusion and avalanche by round count T.

Flips exactly one input bit and measures the 256-bit output Hamming-distance
DISTRIBUTION (not just the mean), per-output-bit flip probability, the
input-bit/output-bit dependency matrix, worst input/output bits, and selected
higher-order (2-bit and 4-bit) differentials, for a grid of T values.

T is NOT changed in the normative primitive; each T is an audited variant via
PDEVariant. Operates on raw 256-bit output only.

Run:
    python -m pde_audit.avalanche
"""

from __future__ import annotations

import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams

T_GRID = [1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
MSG_LEN = 32                      # fixed-length base messages => 256 input bits
N_OUT = 256


def _digest_bits(variant: PDEVariant, msg: bytes) -> np.ndarray:
    return H.bits_of(variant.hash(msg))


def _base_messages(seed: int, n_random: int) -> dict:
    g = H.rng(seed)
    fams = {
        "zero": b"\x00" * MSG_LEN,
        "ff": b"\xff" * MSG_LEN,
        "incr": bytes((i % 256) for i in range(MSG_LEN)),
        "repeat_5a": b"\x5a" * MSG_LEN,
        "sparse_one": (b"\x80" + b"\x00" * (MSG_LEN - 1)),
    }
    for r in range(n_random):
        fams[f"rand{r}"] = g.integers(0, 256, size=MSG_LEN, dtype=np.uint8).tobytes()
    return fams


def _flip_bit(msg: bytes, bit: int) -> bytes:
    ba = bytearray(msg)
    ba[bit >> 3] ^= 1 << (7 - (bit & 7))
    return bytes(ba)


def analyze_T(T: int, seed: int, n_random: int = 8) -> dict:
    variant = PDEVariant(PDEParams(T=T))
    bases = _base_messages(seed, n_random)
    n_in = MSG_LEN * 8

    hd_all = []                                   # all single-bit-flip HDs
    out_flip = np.zeros(N_OUT, dtype=np.int64)    # per-output-bit flip count
    dep = np.zeros((n_in, N_OUT), dtype=np.int64) # input-bit -> output-bit flips
    n_pairs = 0
    worst = {"max_hd": -1, "min_hd": 10 ** 9}

    for name, msg in bases.items():
        base_bits = _digest_bits(variant, msg)
        for i in range(n_in):
            fb = _digest_bits(variant, _flip_bit(msg, i))
            diff = base_bits ^ fb
            hd = int(diff.sum())
            hd_all.append(hd)
            out_flip += diff
            dep[i] += diff
            n_pairs += 1
            if hd > worst["max_hd"]:
                worst["max_hd"] = hd
                worst["max_input"] = {"family": name, "bit": i}
            if hd < worst["min_hd"]:
                worst["min_hd"] = hd
                worst["min_input"] = {"family": name, "bit": i}

    hd_all = np.array(hd_all)
    out_p = out_flip / n_pairs
    # per output bit: deviation of flip prob from 0.5
    worst_out_bit = int(np.argmax(np.abs(out_p - 0.5)))

    # higher-order differentials
    ho = _higher_order(variant, bases, n_in, seed)

    return {
        "T": T,
        "n_single_bit_pairs": n_pairs,
        "hd": _dist(hd_all),
        "output_bit_flip_prob": {
            "mean": float(out_p.mean()),
            "min": float(out_p.min()),
            "max": float(out_p.max()),
            "worst_bit": worst_out_bit,
            "worst_bit_prob": float(out_p[worst_out_bit]),
        },
        "dependency_matrix_summary": _dep_summary(dep, n_pairs, len(bases)),
        "worst": worst,
        "higher_order": ho,
    }


def _higher_order(variant, bases, n_in, seed):
    g = H.rng(seed + 777)
    msg = bases["rand0"] if "rand0" in bases else list(bases.values())[0]
    base_bits = _digest_bits(variant, msg)
    hd2, hd4 = [], []
    for _ in range(64):
        i, j = g.choice(n_in, size=2, replace=False)
        m = _flip_bit(_flip_bit(msg, int(i)), int(j))
        hd2.append(int((base_bits ^ _digest_bits(variant, m)).sum()))
    for _ in range(32):
        idx = g.choice(n_in, size=4, replace=False)
        m = msg
        for b in idx:
            m = _flip_bit(m, int(b))
        hd4.append(int((base_bits ^ _digest_bits(variant, m)).sum()))
    return {"two_bit_hd": _dist(np.array(hd2)), "four_bit_hd": _dist(np.array(hd4))}


def _dep_summary(dep, n_pairs, n_bases):
    # dep[i,j] in [0, n_bases]; probability output j flips given input i flipped
    prob = dep / n_bases
    # ideal ~0.5 everywhere; report extremes and count of near-zero entries
    return {
        "mean": float(prob.mean()),
        "min": float(prob.min()),
        "max": float(prob.max()),
        "frac_below_0.1": float((prob < 0.1).mean()),
        "frac_above_0.9": float((prob > 0.9).mean()),
        "frac_in_0.4_0.6": float(((prob >= 0.4) & (prob <= 0.6)).mean()),
    }


def _dist(arr: np.ndarray) -> dict:
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": int(arr.min()),
        "max": int(arr.max()),
        "p1": float(np.percentile(arr, 1)),
        "p5": float(np.percentile(arr, 5)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def main(seed: int = 80010) -> dict:
    t0 = time.perf_counter()
    results = {
        "phase": "8A_avalanche",
        "metadata": H.env_metadata(),
        "seed": seed,
        "config": {"T_grid": T_GRID, "msg_len_bytes": MSG_LEN, "ideal_mean_hd": N_OUT / 2},
        "by_T": {},
    }
    for T in T_GRID:
        tT = time.perf_counter()
        r = analyze_T(T, seed)
        r["runtime_s"] = round(time.perf_counter() - tT, 2)
        results["by_T"][str(T)] = r
        print(f"  T={T:>3}  meanHD={r['hd']['mean']:.1f}  "
              f"min={r['hd']['min']} max={r['hd']['max']}  "
              f"out_p[min,max]=[{r['output_bit_flip_prob']['min']:.3f},"
              f"{r['output_bit_flip_prob']['max']:.3f}]  ({r['runtime_s']}s)")
    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    path = H.save_artifact("phase8a_avalanche.json", results)
    print("saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
