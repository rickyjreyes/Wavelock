"""Phase 8B: state-map injectivity, rank, collapse, fixed points, and cycles.

Because the T-round transformation is NOT known to be bijective, this is a
primary analysis. Two tracks:

  * full system (N=16, p=2**31-1): modular Jacobian rank distribution of the
    one-round map at sampled states, constant-state fixed-point analysis, an
    explicit all-zero fixed-point check, Brent cycle detection within a step
    budget, and a duplicate-next-state search over a sampled set.
  * exhaustive toy systems (pde_audit/toy.py): full enumeration for small (N,p).

Operates on raw state only.

Run:
    python -m pde_audit.state_map
"""

from __future__ import annotations

import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams
from . import toy as toy_mod


# ---------------------------------------------------------------------------
# full-system Jacobian of the one-round map
# ---------------------------------------------------------------------------
def one_round_jacobian(psi: np.ndarray, params: PDEParams) -> np.ndarray:
    """256x256 modular Jacobian of one_round at state psi.

    psi'[c] = psi[c] + D*lap[c] + a*psi[c]*(b - psi[c]^2)
      d/d psi[c]      = 1 + D*(p-4) + a*b - 3*a*psi[c]^2     (mod p)
      d/d psi[nbr]    = D                                     (4 toroidal nbrs)
    """
    N, p = params.N, params.p
    n = N * N
    J = np.zeros((n, n), dtype=np.int64)
    flat = psi.reshape(-1) % p
    diag = (1 + params.D * (p - 4) + params.a * params.b
            - 3 * params.a * (flat * flat % p)) % p
    for i in range(N):
        for j in range(N):
            c = i * N + j
            J[c, c] = diag[c]
            for (di, dj) in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = ((i + di) % N) * N + ((j + dj) % N)
                J[c, nb] = (J[c, nb] + params.D) % p
    return J


def jacobian_rank_study(params: PDEParams, seed: int, n_samples: int) -> dict:
    g = H.rng(seed)
    v = PDEVariant(params)
    n = params.N * params.N
    ranks = []
    structured = [
        np.zeros((params.N, params.N), dtype=np.int64),                 # all-zero
        np.full((params.N, params.N), params.p - 1, dtype=np.int64),    # all max
        v.iv(),                                                          # IV
    ]
    samples = structured + [
        g.integers(0, params.p, size=(params.N, params.N), dtype=np.int64)
        for _ in range(n_samples)
    ]
    for s in samples:
        J = one_round_jacobian(s, params)
        ranks.append(H.mod_rank_np(J, params.p))
    ranks = np.array(ranks)
    return {
        "n": int(ranks.size),
        "full_rank_value": n,
        "rank_mean": float(ranks.mean()),
        "rank_min": int(ranks.min()),
        "rank_max": int(ranks.max()),
        "frac_full_rank": float((ranks == n).mean()),
        "frac_rank_deficient": float((ranks < n).mean()),
        "rank_deficit_max": int(n - ranks.min()),
    }


# ---------------------------------------------------------------------------
# fixed points (constant states) and all-zero
# ---------------------------------------------------------------------------
def fixed_point_study(params: PDEParams) -> dict:
    """Constant state v*1: lap=0, psi' = v + a*v*(b - v^2).

    Fixed iff a*v*(b-v^2) == 0  <=>  v == 0  or  v^2 == b (mod p).
    """
    N, p = params.N, params.p
    v = PDEVariant(params)
    # all-zero is a fixed point of one_round and hence evolve_T
    zero = np.zeros((N, N), dtype=np.int64)
    zero_fixed_1 = bool(np.array_equal(v.one_round(zero), zero))
    zero_fixed_T = bool(np.array_equal(v.evolve_T(zero), zero))

    # constant fixed points from v^2 == b: count square roots of b
    # (b is a QR mod p iff b^((p-1)/2) == 1)
    b = params.b % p
    is_qr = (b == 0) or (pow(b, (p - 1) // 2, p) == 1)
    n_const_fixed = 1 + (2 if (is_qr and b != 0) else (1 if b == 0 else 0))
    return {
        "all_zero_is_fixed_one_round": zero_fixed_1,
        "all_zero_is_fixed_T_round": zero_fixed_T,
        "b_is_quadratic_residue": bool(is_qr),
        "num_constant_fixed_points": int(n_const_fixed),
        "note": "constant states v with v==0 or v^2==b are fixed; lap vanishes on constants",
    }


# ---------------------------------------------------------------------------
# Brent cycle detection on the one-round map (state space is astronomically
# large; report what is found within a step budget)
# ---------------------------------------------------------------------------
def brent_cycle(params: PDEParams, start: np.ndarray, budget: int) -> dict:
    v = PDEVariant(params)

    def f(s):
        return v.one_round(s)

    def key(s):
        return s.tobytes()

    # Brent's algorithm with a step budget
    power = lam = 1
    tortoise = start.copy()
    hare = f(start)
    steps = 0
    while key(tortoise) != key(hare):
        if lam == power:
            tortoise = hare.copy()
            power *= 2
            lam = 0
        hare = f(hare)
        lam += 1
        steps += 1
        if steps > budget:
            return {"found": False, "steps": steps, "lambda": None, "mu": None}
    # found cycle length lam; find mu (start index of cycle)
    tortoise = start.copy()
    hare = start.copy()
    for _ in range(lam):
        hare = f(hare)
    mu = 0
    while key(tortoise) != key(hare):
        tortoise = f(tortoise)
        hare = f(hare)
        mu += 1
        steps += 1
        if steps > budget:
            return {"found": False, "steps": steps, "lambda": lam, "mu": None}
    return {"found": True, "lambda": int(lam), "mu": int(mu), "steps": steps}


def cycle_study(params: PDEParams, seed: int, n_starts: int, budget: int) -> dict:
    g = H.rng(seed)
    results = []
    for _ in range(n_starts):
        start = g.integers(0, params.p, size=(params.N, params.N), dtype=np.int64)
        results.append(brent_cycle(params, start, budget))
    found = [r for r in results if r["found"]]
    return {
        "n_starts": n_starts,
        "budget": budget,
        "num_short_cycles_found": len(found),
        "examples": found[:5],
    }


# ---------------------------------------------------------------------------
# duplicate next-state search (full system; structural collisions only)
# ---------------------------------------------------------------------------
def duplicate_search(params: PDEParams, seed: int, n_states: int) -> dict:
    g = H.rng(seed)
    v = PDEVariant(params)
    seen = {}
    dups = 0
    for _ in range(n_states):
        s = g.integers(0, params.p, size=(params.N, params.N), dtype=np.int64)
        out = v.one_round(s).tobytes()
        if out in seen:
            dups += 1
        else:
            seen[out] = True
    return {"n_states": n_states, "duplicate_next_states": dups,
            "note": "random full states; duplicates would indicate gross many-to-one collapse"}


# ---------------------------------------------------------------------------
# toy enumeration
# ---------------------------------------------------------------------------
def toy_study() -> list:
    out = []
    for P in toy_mod.DEFAULT_TOY_GRID:
        out.append(toy_mod.enumerate_toy(P))
    return out


def main(seed: int = 80020) -> dict:
    t0 = time.perf_counter()
    p0 = PDEParams()
    print("  jacobian rank study (full N=16) ...")
    jac = jacobian_rank_study(p0, seed, n_samples=60)
    print(f"    frac_full_rank={jac['frac_full_rank']:.3f} "
          f"rank[min,max]=[{jac['rank_min']},{jac['rank_max']}]/{jac['full_rank_value']}")
    print("  fixed-point study ...")
    fp = fixed_point_study(p0)
    print(f"    all_zero_fixed={fp['all_zero_is_fixed_T_round']} "
          f"const_fixed={fp['num_constant_fixed_points']}")
    print("  cycle study (Brent, budget 20000) ...")
    cyc = cycle_study(p0, seed + 1, n_starts=8, budget=20000)
    print(f"    short_cycles_found={cyc['num_short_cycles_found']}/{cyc['n_starts']}")
    print("  duplicate-next-state search ...")
    dup = duplicate_search(p0, seed + 2, n_states=20000)
    print(f"    duplicate_next_states={dup['duplicate_next_states']}/{dup['n_states']}")
    print("  exhaustive toy enumeration ...")
    toy = toy_study()
    for t in toy:
        pr = t["params"]
        print(f"    N={pr['N']} p={pr['p']} T={pr['T']}: img_frac={t['image_fraction']:.3f} "
              f"maxpre={t['max_preimage_multiplicity']} fixed={t['num_fixed_points']} "
              f"cycles={t['num_cycles']} -> {t['one_round_is']}")

    results = {
        "phase": "8B_state_map",
        "metadata": H.env_metadata(),
        "seed": seed,
        "params_v0": p0.__dict__,
        "jacobian_rank": jac,
        "fixed_points": fp,
        "cycles": cyc,
        "duplicate_next_state": dup,
        "toy_exhaustive": toy,
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    path = H.save_artifact("phase8b_state_map.json", results)
    print("saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
