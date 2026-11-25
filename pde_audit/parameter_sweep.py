"""Phase 8I: parameter study (predeclared grid, audited not optimized).

The grid below is DECLARED BEFORE execution. For each regime a compact metric
set is measured: avalanche mean HD, per-bit bias, squeeze tie rate, one-round
Jacobian full-rank fraction, a short-cycle/collapse indicator, reduced-output
(16-bit) collision ratio vs birthday, and runtime. Every regime is recorded,
including poor ones.

v0 constants are NOT changed here. Any future recommendation is deferred to the
results document; a changed candidate would get a new version and new vectors.

Run:
    python -m pde_audit.parameter_sweep
"""

from __future__ import annotations

import math
import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams
from .state_map import one_round_jacobian, brent_cycle

P = (1 << 31) - 1

# --- predeclared grid: vary one axis at a time around v0 -------------------
V0 = dict(N=16, p=P, D=5, a=3, b=1431655765, G=7, T=32)
GRID = [("v0_baseline", dict(V0))]
for T in [4, 8, 16, 48, 64]:
    GRID.append((f"T={T}", {**V0, "T": T}))
for D in [1, 2, 3, 7, 11]:
    GRID.append((f"D={D}", {**V0, "D": D}))
for a in [1, 2, 5, 7]:
    GRID.append((f"a={a}", {**V0, "a": a}))
for b in [P // 4, P // 3, P // 2, (2 * P) // 3, 1]:
    GRID.append((f"b={b}", {**V0, "b": b}))


def _metrics(params: PDEParams, seed: int) -> dict:
    v = PDEVariant(params)
    g = H.rng(seed)

    # avalanche (small)
    hds = []
    base_msgs = [g.integers(0, 256, size=32, dtype=np.uint8).tobytes() for _ in range(24)]
    for m in base_msgs:
        bb = H.bits_of(v.hash(m))
        for _ in range(6):
            bit = int(g.integers(0, 256))
            ba = bytearray(m); ba[bit >> 3] ^= 1 << (7 - (bit & 7))
            hds.append(int((bb ^ H.bits_of(v.hash(bytes(ba)))).sum()))
    hds = np.array(hds)

    # bit bias + tie rate over states
    acc = np.zeros(256)
    ties = 0
    n_bias = 800
    pa, pb = v._pairs_a, v._pairs_b
    for _ in range(n_bias):
        m = g.integers(0, 256, size=int(g.integers(1, 200)), dtype=np.uint8).tobytes()
        psi = v.absorb(m)
        acc += H.bits_of(v.squeeze(psi))
        flat = psi.reshape(-1)
        ties += int((flat[pa] == flat[pb]).sum())
    bias = float(np.max(np.abs(acc / n_bias - 0.5)))

    # jacobian full-rank fraction
    n = params.N * params.N
    full = 0
    for _ in range(12):
        s = g.integers(0, params.p, size=(params.N, params.N), dtype=np.int64)
        if H.mod_rank_np(one_round_jacobian(s, params), params.p) == n:
            full += 1
    jac_frac = full / 12

    # short-cycle indicator
    start = g.integers(0, params.p, size=(params.N, params.N), dtype=np.int64)
    cyc = brent_cycle(params, start, budget=4000)

    # reduced 16-bit collision ratio vs birthday
    seen = {}
    coll_evals = None
    for e in range(40000):
        d = int.from_bytes(v.hash(e.to_bytes(8, "big"))[:4], "big") >> 16
        if d in seen:
            coll_evals = e + 1
            break
        seen[d] = e
    birthday16 = 1.2533 * math.sqrt(2 ** 16)
    ratio = (coll_evals / birthday16) if coll_evals else None

    return {
        "avalanche_mean_hd": round(float(hds.mean()), 2),
        "avalanche_min_hd": int(hds.min()),
        "max_bit_bias": round(bias, 4),
        "tie_rate_per_pair": ties / (n_bias * 64),
        "jacobian_full_rank_frac": jac_frac,
        "short_cycle_found": bool(cyc["found"]),
        "coll16_evals": coll_evals,
        "coll16_ratio_vs_birthday": round(ratio, 2) if ratio else None,
    }


def main(seed: int = 80100) -> dict:
    t0 = time.perf_counter()
    rows = {}
    for name, pd in GRID:
        tt = time.perf_counter()
        params = PDEParams(**pd)
        m = _metrics(params, seed)
        m["runtime_s"] = round(time.perf_counter() - tt, 2)
        rows[name] = {"params": pd, "metrics": m}
        print(f"  {name:14s} HD={m['avalanche_mean_hd']:6.1f} bias={m['max_bit_bias']:.3f} "
              f"tie={m['tie_rate_per_pair']:.1e} jacFR={m['jacobian_full_rank_frac']:.2f} "
              f"cyc={m['short_cycle_found']} coll16r={m['coll16_ratio_vs_birthday']} "
              f"({m['runtime_s']}s)")

    results = {
        "phase": "8I_parameter_sweep",
        "metadata": H.env_metadata(),
        "seed": seed,
        "grid_predeclared": [name for name, _ in GRID],
        "v0": V0,
        "rows": rows,
        "note": "v0 constants unchanged; recommendations deferred to results doc.",
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    path = H.save_artifact("phase8i_parameter_sweep.json", results)
    print("  saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
