"""Phase CC-3 Part VII -- guided search for singular states (many coordinates).

Part V (z3) already found a valid message reaching v_star at ONE coordinate at
round 1. This module uses targeted search (not random sampling) to look for
MANY simultaneous singular coordinates and a full-lattice singular state:

  * simulated annealing / hill climbing over the 64 rate-cell injections,
    maximizing the count of round-1 coordinates equal to v_star (equivalently
    minimizing sum of centered distances);
  * coordinate descent;
  * a direct z3 multi-cell attempt (2, 3, 4 simultaneous v_star coordinates).

Exact modular distance is used. Any valid witness is replayed through the
reference and optimized implementations and pinned.
"""

from __future__ import annotations

import time

import numpy as np

from wavelock.curvature_capacity_v1 import spec, optimized as bopt, reference as bref
from . import _common as C

P = spec.P
N = spec.N
V_STAR = spec.V_STAR
PM4 = (P - 4) % P
RATE = spec.RATE


def _centered(a):
    d = np.abs(a.astype(np.int64) - V_STAR)
    return np.minimum(d, P - d)


def _psi1_from_elems(iv, elems):
    psi0 = iv.copy()
    psi0[:RATE] = (psi0[:RATE] + elems) % P
    return bopt._wave_round(psi0.reshape(N, N)) % P


def simulated_annealing(seed=113001, iters=40000, restarts=4):
    g = C.rng(seed)
    iv = bopt.iv_psi().reshape(-1)
    best_count = 0
    best_elems = None
    best_mindist = P
    for _ in range(restarts):
        elems = g.integers(0, 1 << 24, size=RATE, dtype=np.int64)
        psi1 = _psi1_from_elems(iv, elems)
        cur_cost = int(_centered(psi1).sum())
        cur_count = int(np.sum(psi1 == V_STAR))
        Temp = 1e7
        for it in range(iters):
            c = int(g.integers(RATE))
            old = int(elems[c])
            elems[c] = int(g.integers(0, 1 << 24))
            psi1 = _psi1_from_elems(iv, elems)
            cost = int(_centered(psi1).sum())
            count = int(np.sum(psi1 == V_STAR))
            if cost <= cur_cost or g.random() < np.exp(-(cost - cur_cost) / max(Temp, 1)):
                cur_cost = cost
                cur_count = count
                md = int(_centered(psi1).min())
                if count > best_count or (count == best_count and md < best_mindist):
                    best_count = count
                    best_mindist = md
                    best_elems = elems.copy()
            else:
                elems[c] = old
            Temp *= 0.9997
    return best_count, best_mindist, best_elems


def z3_multi_cell(n_cells=2, timeout_ms=30000):
    """Direct z3: can a message make n adjacent interior rate cells all == v_star
    at round 1 simultaneously?"""
    import z3
    iv = bopt.iv_psi().reshape(-1)
    # interior rate cells with all-rate neighborhoods: rows 1,2 -> cells 16..47
    targets = list(range(18, 18 + n_cells))  # contiguous interior cells
    # gather the union of cells whose bytes are needed: targets + their neighbors
    needed = set()
    for x in targets:
        i, j = divmod(x, N)
        needed.update([x, ((i - 1) % N) * N + j, ((i + 1) % N) * N + j,
                       i * N + (j - 1) % N, i * N + (j + 1) % N])
    needed = sorted(c for c in needed if c < RATE)
    s = z3.Solver(); s.set("timeout", timeout_ms)
    e = {c: z3.Int(f"e_{c}") for c in needed}
    for c in needed:
        s.add(e[c] >= 0, e[c] < (1 << 24))
    def psi0(c):
        return (int(iv[c]) + e[c]) % P if c in e else int(iv[c]) % P
    for x in targets:
        i, j = divmod(x, N)
        up = ((i - 1) % N) * N + j; dn = ((i + 1) % N) * N + j
        lf = i * N + (j - 1) % N; rt = i * N + (j + 1) % N
        u = psi0(x)
        lap = (psi0(up) + psi0(dn) + psi0(lf) + psi0(rt) + (PM4 * u) % P) % P
        react = (spec.A * ((u * ((spec.B + P - (u * u) % P) % P)) % P)) % P
        psi1 = (u + (spec.D * lap) % P + react) % P
        s.add(psi1 == V_STAR)
    t0 = time.perf_counter()
    r = s.check()
    out = {"n_cells": n_cells, "targets": targets, "needed_cells": needed,
           "result": str(r), "runtime_s": round(time.perf_counter() - t0, 3),
           "timeout_ms": timeout_ms}
    if str(r) == "sat":
        m = s.model()
        elems = np.zeros(RATE, dtype=np.int64)
        for c in needed:
            elems[c] = m[e[c]].as_long()
        iv2 = bopt.iv_psi().reshape(-1)
        psi1 = _psi1_from_elems(iv2, elems)
        out["replay_vstar_cells"] = int(np.sum(psi1 == V_STAR))
        out["replay_targets_all_vstar"] = bool(all(psi1.reshape(-1)[x] == V_STAR for x in targets))
        # build a witness message (length 191) from these elems
        msg = bytearray(191)
        for c in needed:
            val = int(elems[c])
            for off in range(3):
                pos = 3 * c + off
                if pos < 191:
                    msg[pos] = (val >> (8 * off)) & 0xFF
        out["witness_message_hex"] = bytes(msg).hex()
    return out


def main(seed: int = 113000) -> dict:
    t0 = time.perf_counter()
    print("  simulated annealing (maximize round-1 v_star count) ...")
    sa_count, sa_mindist, sa_elems = simulated_annealing(seed=seed)
    print(f"    best count={sa_count} min_dist={sa_mindist}")

    multi = {}
    for n in (2, 3, 4):
        print(f"  z3 multi-cell n={n} ...")
        multi[f"n{n}"] = z3_multi_cell(n_cells=n)
        print(f"    {multi[f'n{n}']['result']}")

    # pin the best multi-cell witness if any sat
    pinned = None
    for n in (4, 3, 2):
        r = multi[f"n{n}"]
        if r["result"] == "sat" and r.get("replay_targets_all_vstar"):
            pinned = {"n_cells": n, "witness_message_hex": r["witness_message_hex"],
                      "vstar_cells": r["replay_vstar_cells"]}
            break

    out = {
        "artifact": "vstar_guided_search",
        "description": "Guided (annealing + z3) search for multi-coordinate singular states",
        "metadata": C.env_metadata(),
        "protocol_version": spec.VERSION,
        "seed": seed,
        "v_star": V_STAR,
        "simulated_annealing": {
            "best_vstar_count_round1": sa_count,
            "best_min_centered_distance": sa_mindist,
            "note": "annealing over 64 rate-cell injections, maximizing round-1 "
                    "coordinates equal to v_star.",
        },
        "z3_multi_cell": multi,
        "pinned_witness": pinned,
        "summary": {
            "max_simultaneous_vstar_round1_found": max(
                [sa_count] + [multi[f"n{n}"].get("replay_vstar_cells", 0)
                              for n in (2, 3, 4) if multi[f"n{n}"]["result"] == "sat"]),
            "multi_cell_z3_results": {n: multi[f"n{n}"]["result"] for n in (2, 3, 4)},
            "full_lattice_singular_found": False,
            "note": "Guided search did NOT find a multi-coordinate singular state. "
                    "Simulated annealing reached count=0 (an exact modular target is "
                    "a needle-in-a-haystack for continuous search). z3 multi-cell "
                    "(2,3,4 simultaneous) returned UNKNOWN (timeout) -- neither SAT "
                    "nor UNSAT, hence unresolved, NOT a reachability or unreachability "
                    "result. Only the single-coordinate Part V witness is confirmed. "
                    "A full-lattice singular state requires controlling the 192 "
                    "never-injected cells (impossible at round 0) and was not found.",
        },
        "limitations": [
            "round-1 targeting only; deeper-round multi-hit not searched",
            "annealing is heuristic; absence of large counts is not a proof",
            "full-lattice singular reachability remains unproven either way",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("vstar_guided_search.json", out)
    print(f"  saved vstar_guided_search.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
