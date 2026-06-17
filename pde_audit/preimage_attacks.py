"""Phase 8G: preimage / second-preimage experiments (truncated, reduced-round).

Attempts brute force, hill climbing, simulated annealing, and a genetic search
against truncated-output targets, plus a second-preimage search and a
state-recovery underdetermination check. Measures attack cost as a function of
output bits, round count, and message degrees of freedom, and compares to the
generic 2**n preimage expectation.

The point of the local/metaheuristic searches is to test for EXPLOITABLE
STRUCTURE: if simulated annealing / GA / hill climbing cannot beat blind
sampling, the truncated objective has no usable gradient (evidence, not proof,
of one-wayness on the tested scale). Operates on raw / truncated output only.

Run:
    python -m pde_audit.preimage_attacks
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams

MSG_LEN = 32  # message degrees of freedom for local-search experiments


def _trunc(variant: PDEVariant, msg: bytes, nbits: int) -> int:
    return int.from_bytes(variant.hash(msg)[:5], "big") >> (40 - nbits)


def _trunc_bits(variant, msg, nbits):
    v = _trunc(variant, msg, nbits)
    return np.array([(v >> (nbits - 1 - i)) & 1 for i in range(nbits)], dtype=np.uint8)


# ---------------------------------------------------------------------------
# brute force
# ---------------------------------------------------------------------------
def brute_force(variant, target: int, nbits: int, budget: int, start: int):
    for e in range(budget):
        m = (start + e).to_bytes(8, "big")
        if _trunc(variant, m, nbits) == target:
            return {"found": True, "evals": e + 1, "msg": m.hex()}
    return {"found": False, "evals": budget}


# ---------------------------------------------------------------------------
# hill climbing / simulated annealing on n-bit Hamming objective
# ---------------------------------------------------------------------------
def _hamming_to_target(variant, msg, target_bits, nbits):
    return int((target_bits ^ _trunc_bits(variant, msg, nbits)).sum())


def hill_climb(variant, target_bits, nbits, seed, budget):
    g = H.rng(seed)
    cur = bytearray(g.integers(0, 256, size=MSG_LEN, dtype=np.uint8).tobytes())
    best = _hamming_to_target(variant, bytes(cur), target_bits, nbits)
    evals = 1
    trace_best = best
    while evals < budget and best > 0:
        cand = bytearray(cur)
        pos = int(g.integers(0, MSG_LEN))
        cand[pos] ^= 1 << int(g.integers(0, 8))
        d = _hamming_to_target(variant, bytes(cand), target_bits, nbits)
        evals += 1
        if d <= best:
            cur, best = cand, d
        trace_best = min(trace_best, best)
    return {"best_distance": int(best), "evals": evals, "solved": best == 0}


def simulated_annealing(variant, target_bits, nbits, seed, budget):
    g = H.rng(seed)
    cur = bytearray(g.integers(0, 256, size=MSG_LEN, dtype=np.uint8).tobytes())
    cur_d = _hamming_to_target(variant, bytes(cur), target_bits, nbits)
    best = cur_d
    evals = 1
    T0, T1 = 4.0, 0.05
    while evals < budget and best > 0:
        frac = evals / budget
        temp = T0 * (T1 / T0) ** frac
        cand = bytearray(cur)
        pos = int(g.integers(0, MSG_LEN))
        cand[pos] ^= 1 << int(g.integers(0, 8))
        d = _hamming_to_target(variant, bytes(cand), target_bits, nbits)
        evals += 1
        if d <= cur_d or g.random() < math.exp(-(d - cur_d) / temp):
            cur, cur_d = cand, d
            best = min(best, d)
    return {"best_distance": int(best), "evals": evals, "solved": best == 0}


def genetic(variant, target_bits, nbits, seed, budget, pop=40):
    g = H.rng(seed)
    population = [bytearray(g.integers(0, 256, size=MSG_LEN, dtype=np.uint8).tobytes())
                 for _ in range(pop)]
    evals = 0
    best = nbits

    def fit(ind):
        nonlocal evals
        evals += 1
        return _hamming_to_target(variant, bytes(ind), target_bits, nbits)

    scores = [fit(ind) for ind in population]
    while evals < budget and best > 0:
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = [scores[i] for i in order]
        best = min(best, scores[0])
        survivors = population[: pop // 2]
        children = []
        for _ in range(pop - len(survivors)):
            a, b = (survivors[int(g.integers(0, len(survivors)))] for _ in range(2))
            cut = int(g.integers(1, MSG_LEN))
            child = bytearray(a[:cut] + b[cut:])
            if g.random() < 0.5:
                child[int(g.integers(0, MSG_LEN))] ^= 1 << int(g.integers(0, 8))
            children.append(child)
        population = survivors + children
        scores = scores[: len(survivors)] + [fit(c) for c in children]
    return {"best_distance": int(best), "evals": evals, "solved": best == 0}


def random_baseline(variant, target_bits, nbits, seed, budget):
    """Blind sampling: best n-bit Hamming distance achieved in `budget` draws."""
    g = H.rng(seed)
    best = nbits
    for _ in range(budget):
        m = g.integers(0, 256, size=MSG_LEN, dtype=np.uint8).tobytes()
        best = min(best, _hamming_to_target(variant, m, target_bits, nbits))
        if best == 0:
            break
    return {"best_distance": int(best)}


# ---------------------------------------------------------------------------
# second preimage
# ---------------------------------------------------------------------------
def second_preimage(variant, nbits, seed, budget):
    target_msg = b"second-preimage-target"
    target = _trunc(variant, target_msg, nbits)
    start = seed * 5_000_000
    for e in range(budget):
        m = (start + e).to_bytes(8, "big")
        if m == target_msg:
            continue
        if _trunc(variant, m, nbits) == target:
            return {"found": True, "evals": e + 1, "nbits": nbits,
                    "target_msg": target_msg.hex(), "second": m.hex()}
    return {"found": False, "evals": budget, "nbits": nbits}


# ---------------------------------------------------------------------------
# state-recovery underdetermination
# ---------------------------------------------------------------------------
def state_recovery_underdetermination(seed, trials=4000):
    """The first squeeze round exposes only 64 comparison bits of a 256-cell
    state (~256*31 hidden bits). Count how many random states match a fixed
    pattern of those 64 bits -> demonstrates massive non-uniqueness."""
    v = PDEVariant(PDEParams())
    g = H.rng(seed)
    pa, pb = v._pairs_a, v._pairs_b
    ref = g.integers(0, v.P.p, size=256, dtype=np.int64)
    ref_bits = (ref[pa] > ref[pb]).astype(np.uint8)
    matches = 0
    for _ in range(trials):
        s = g.integers(0, v.P.p, size=256, dtype=np.int64)
        if np.array_equal((s[pa] > s[pb]).astype(np.uint8), ref_bits):
            matches += 1
    return {"trials": trials, "states_matching_64_squeeze_bits": matches,
            "expected_if_uniform": trials / (2 ** 64),
            "note": "64 exposed bits vs ~7936-bit state => preimage of the "
                    "squeeze is hugely underdetermined"}


def main(seed: int = 80070) -> dict:
    t0 = time.perf_counter()
    results = {"phase": "8G_preimage", "metadata": H.env_metadata(), "seed": seed}

    # brute force vs round count and output bits
    bf = {}
    for T in [4, 8, 32]:
        v = PDEVariant(PDEParams(T=T))
        for nbits in ([8, 12, 16] if T == 32 else [8, 12, 16, 20]):
            target = _trunc(v, b"preimage-target", nbits)
            budget = int(min(8 * 2 ** nbits, 300_000))
            r = brute_force(v, target, nbits, budget, start=seed * 11_000_000)
            r["birthday_free_expected"] = 2 ** nbits
            bf[f"T{T}_n{nbits}"] = r
    results["brute_force"] = bf
    print("  brute force samples:",
          {k: (v["evals"] if v["found"] else "miss") for k, v in list(bf.items())[:6]})

    # local search vs blind sampling (n=32 truncation, reduced T=8)
    v8 = PDEVariant(PDEParams(T=8))
    g = H.rng(seed + 5)
    tb = _trunc_bits(v8, g.integers(0, 256, size=MSG_LEN, dtype=np.uint8).tobytes(), 32)
    budget = 20_000
    results["local_search_n32_T8"] = {
        "objective_bits": 32, "budget": budget,
        "hill_climb": hill_climb(v8, tb, 32, seed + 6, budget),
        "simulated_annealing": simulated_annealing(v8, tb, 32, seed + 7, budget),
        "genetic": genetic(v8, tb, 32, seed + 8, budget),
        "random_baseline": random_baseline(v8, tb, 32, seed + 9, budget),
        "interpretation": "if metaheuristics do not beat random_baseline, the "
                          "truncated objective has no exploitable gradient",
    }
    ls = results["local_search_n32_T8"]
    print(f"  local search n=32 T=8 best-dist: HC={ls['hill_climb']['best_distance']} "
          f"SA={ls['simulated_annealing']['best_distance']} "
          f"GA={ls['genetic']['best_distance']} rand={ls['random_baseline']['best_distance']}")

    # second preimage
    results["second_preimage"] = {
        str(n): second_preimage(PDEVariant(PDEParams(T=8)), n, seed + 20 + n, 200_000)
        for n in [16, 20]
    }
    # state recovery underdetermination
    results["state_recovery"] = state_recovery_underdetermination(seed + 30)
    print(f"  second-preimage 16-bit: {results['second_preimage']['16'].get('evals')} evals "
          f"found={results['second_preimage']['16']['found']}")
    print(f"  state-recovery: {results['state_recovery']['states_matching_64_squeeze_bits']} "
          f"/{results['state_recovery']['trials']} states match 64 squeeze bits")

    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    path = H.save_artifact("phase8g_preimage.json", results)
    print("  saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
