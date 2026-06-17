"""Phase 8H: algebraic analysis of the F_p polynomial dynamical system.

Each round is a degree-3 polynomial map over F_p, so the formal degree of the
T-round map grows like 3**T. This module measures whether that formal degree
implies hardness:

  * functional degree (toy): interpolate the N=1 univariate map f^T over a small
    field and report its REDUCED functional degree (<= p-1 by Fermat), versus
    the formal 3**T. Formal degree != hardness.
  * affine-relation test (full N=16): sample (input, one_round(input)) pairs and
    test over F_p whether any nonzero affine relation a.in + b.out = c holds
    (it must not, if the map is genuinely nonlinear). Uses modular rank.
  * SMT reduced-round inversion (z3): encode evolve_T for small toy systems
    (N=2, small p, increasing T) and ask z3 for a preimage of a target output;
    record solver time / success / timeout to expose the scaling of an algebraic
    attack. Full N=16, p=2**31-1 inversion is far beyond this and is documented
    as infeasible for the available solver, not claimed "passed".

Operates on raw state only.

Run:
    python -m pde_audit.algebraic_analysis
"""

from __future__ import annotations

import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams
from . import toy as toy_mod

try:
    import z3
    _Z3 = True
except Exception:  # pragma: no cover
    _Z3 = False


# ---------------------------------------------------------------------------
# functional degree of the N=1 toy map (univariate over F_p)
# ---------------------------------------------------------------------------
def _toy_n1_map(x, p, D, a, b, T):
    # N=1: all four neighbors are the same cell, lap = (4 + (p-4))*x = p*x = 0.
    for _ in range(T):
        sq = (x * x) % p
        react = (a * (x * ((b + (p - sq)) % p) % p)) % p
        x = (x + react) % p   # diffusion vanishes for N=1
    return x


def _poly_degree_over_fp(values, p):
    """Degree of the unique polynomial interpolating f on all of F_p, via
    finite differences of the coefficient vector (Gaussian elimination on the
    Vandermonde system over F_p)."""
    n = p
    # Build Vandermonde V[i][j] = i^j, solve V c = values  (mod p)
    V = [[pow(i, j, p) for j in range(n)] for i in range(n)]
    # augmented Gaussian elimination
    M = [row[:] + [values[i] % p] for i, row in enumerate(V)]
    col = 0
    for r in range(n):
        piv = None
        for rr in range(r, n):
            if M[rr][col] % p != 0:
                piv = rr; break
        if piv is None:
            col += 1
            if col >= n: break
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][col], p - 2, p)
        M[r] = [(x * inv) % p for x in M[r]]
        for rr in range(n):
            if rr != r and M[rr][col] % p != 0:
                f = M[rr][col]
                M[rr] = [(M[rr][k] - f * M[r][k]) % p for k in range(n + 1)]
        col += 1
        if col >= n: break
    coeffs = [M[i][n] for i in range(n)]
    deg = 0
    for j in range(n - 1, -1, -1):
        if coeffs[j] % p != 0:
            deg = j; break
    return deg


def functional_degree_study() -> list:
    out = []
    for p in [5, 7, 11, 13]:
        for T in [1, 2, 3, 4, 8]:
            vals = [_toy_n1_map(x, p, 5, 3, (p // 2), T) for x in range(p)]
            deg = _poly_degree_over_fp(vals, p)
            out.append({"N": 1, "p": p, "T": T, "formal_degree": 3 ** T,
                        "functional_degree": deg, "max_possible": p - 1})
    return out


# ---------------------------------------------------------------------------
# affine-relation test on the full one-round map
# ---------------------------------------------------------------------------
def affine_relation_test(seed: int, n_samples: int = 600) -> dict:
    v = PDEVariant(PDEParams())
    g = H.rng(seed)
    p = v.P.p
    n = v.P.N * v.P.N
    rows = []
    for _ in range(n_samples):
        s = g.integers(0, p, size=(v.P.N, v.P.N), dtype=np.int64)
        out = v.one_round(s).reshape(-1)
        rows.append(np.concatenate([s.reshape(-1), out, [1]]))
    M = np.array(rows, dtype=np.int64) % p
    # a nonzero affine relation exists iff the columns are linearly dependent,
    # i.e. column rank < 2n+1.
    rank = H.mod_rank_np(M, p)
    width = 2 * n + 1
    return {"n_samples": n_samples, "matrix_cols": width, "column_rank": rank,
            "full_column_rank": min(width, n_samples),
            "affine_relation_exists": rank < min(width, n_samples),
            "note": "rank == min(cols, rows) => no nonzero affine relation a.in+b.out=c"}


# ---------------------------------------------------------------------------
# SMT reduced-round inversion (z3) on toy systems
# ---------------------------------------------------------------------------
def _z3_one_round(cells, P: toy_mod.ToyParams):
    N, p = P.N, P.p
    pm4 = (p - 4) % p
    out = []
    for i in range(N):
        for j in range(N):
            psi = cells[i * N + j]
            sq = (psi * psi) % p
            react = (P.a * (psi * ((P.b + (p - sq)) % p) % p)) % p
            lap = (cells[((i + 1) % N) * N + j] + cells[((i - 1) % N) * N + j]
                   + cells[i * N + (j + 1) % N] + cells[i * N + (j - 1) % N]
                   + (pm4 * psi) % p) % p
            out.append((psi + (P.D * lap) % p + react) % p)
    return out


def z3_invert(P: toy_mod.ToyParams, seed: int, timeout_ms: int = 20000) -> dict:
    if not _Z3:
        return {"available": False}
    g = H.rng(seed)
    ncells = P.N * P.N
    # pick a random input, compute its output, then ask z3 to find ANY preimage
    secret = g.integers(0, P.p, size=ncells, dtype=np.int64)
    cur = [int(x) for x in secret]
    for _ in range(P.T):
        cur = [c % P.p for c in _np_one_round_list(cur, P)]
    target = cur

    xs = [z3.Int(f"x{i}") for i in range(ncells)]
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    for x in xs:
        s.add(x >= 0, x < P.p)
    cells = xs
    for _ in range(P.T):
        cells = _z3_one_round(cells, P)
    for c, t in zip(cells, target):
        s.add(c % P.p == t)
    t0 = time.perf_counter()
    res = s.check()
    dt = time.perf_counter() - t0
    return {"available": True, "N": P.N, "p": P.p, "T": P.T,
            "result": str(res), "solved": str(res) == "sat",
            "solve_time_s": round(dt, 3), "timeout_ms": timeout_ms}


def _np_one_round_list(cells, P):
    N, p = P.N, P.p
    pm4 = (p - 4) % p
    out = []
    for i in range(N):
        for j in range(N):
            psi = cells[i * N + j]
            sq = (psi * psi) % p
            react = (P.a * (psi * ((P.b + (p - sq)) % p) % p)) % p
            lap = (cells[((i + 1) % N) * N + j] + cells[((i - 1) % N) * N + j]
                   + cells[i * N + (j + 1) % N] + cells[i * N + (j - 1) % N]
                   + (pm4 * psi) % p) % p
            out.append((psi + (P.D * lap) % p + react) % p)
    return out


def smt_study(seed: int) -> dict:
    if not _Z3:
        return {"available": False,
                "note": "z3 not installed; SMT inversion not attempted"}
    results = []
    grid = [
        toy_mod.ToyParams(N=2, p=5, D=5, a=3, b=2, T=1),
        toy_mod.ToyParams(N=2, p=5, D=5, a=3, b=2, T=2),
        toy_mod.ToyParams(N=2, p=7, D=5, a=3, b=3, T=1),
        toy_mod.ToyParams(N=2, p=7, D=5, a=3, b=3, T=2),
        toy_mod.ToyParams(N=2, p=11, D=5, a=3, b=5, T=2),
        toy_mod.ToyParams(N=2, p=13, D=5, a=3, b=6, T=3),
    ]
    for P in grid:
        results.append(z3_invert(P, seed))
    return {"available": True, "inversions": results}


def main(seed: int = 80080) -> dict:
    t0 = time.perf_counter()
    fdeg = functional_degree_study()
    print("  functional degree (N=1 toy): formal 3^T vs functional <= p-1")
    for r in fdeg[:5]:
        print(f"    p={r['p']} T={r['T']}: formal={r['formal_degree']} "
              f"functional={r['functional_degree']} (max {r['max_possible']})")
    aff = affine_relation_test(seed + 1)
    print(f"  affine-relation test (full N=16): column_rank={aff['column_rank']}/"
          f"{aff['full_column_rank']} affine_relation={aff['affine_relation_exists']}")
    smt = smt_study(seed + 2)
    if smt.get("available"):
        for r in smt["inversions"]:
            print(f"  z3 invert N={r['N']} p={r['p']} T={r['T']}: {r['result']} "
                  f"in {r['solve_time_s']}s")

    results = {
        "phase": "8H_algebraic",
        "metadata": H.env_metadata(),
        "seed": seed,
        "z3_available": _Z3,
        "functional_degree": fdeg,
        "affine_relation": aff,
        "smt_inversion": smt,
        "degree_note": "formal degree 3^T does NOT imply hardness: as a function "
                       "over F_p the degree is bounded by p-1 (Fermat); full-"
                       "system hardness is empirical, not implied by 3^T.",
        "full_system_solver_note": "z3 SMT inversion of the full N=16, p=2**31-1 "
                                    "evolve_T is far beyond the installed solver's "
                                    "reach and was not attempted; this is a "
                                    "limitation, not a pass.",
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    path = H.save_artifact("phase8h_algebraic.json", results)
    print("  saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
