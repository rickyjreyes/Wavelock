"""Phase CC-2 Part VIII -- exact algebraic solver attacks on Candidate B.

Uses SymPy (polynomial algebra over GF(p)) and z3 (SMT) to attack reduced-round
and restricted Candidate B systems. A timeout is NOT a security result.

Solver tasks:
  1. SymPy: symbolic structure/degree of the 1-round and 2-round scalar accumulator.
  2. z3: injection collision variety j_B(u,v)=j_B(u',v') exists (SAT, example).
  3. z3 (computer-assisted result): for a fixed concrete v != v_star,
     j_B(u,v)=j_B(u',v) AND u!=u' is UNSAT  ->  injective in u off the hyperplane.
  4. z3: collapse at v_star -- j_B(u,v_star)=0 for all u (SAT family).
  5. z3: one-round coupled 2x2 collision over the REAL prime (bounded, with timeout).
  6. z3: one-round 2x2 preimage of a chosen target (bounded, with timeout).

Records solver, equation/variable counts, degree, runtime, timeout, result,
limitations for each task.
"""

from __future__ import annotations

import time

from wavelock.curvature_capacity_v1 import spec as bspec
from wavelock.curvature_capacity import spec as aspec
from . import _common as C

P = bspec.P
GAMMA = bspec.GAMMA
V_STAR = bspec.V_STAR


def _solver_versions() -> dict:
    v = {}
    try:
        import sympy
        v["sympy"] = sympy.__version__
    except Exception as e:
        v["sympy"] = f"unavailable: {e}"
    try:
        import z3
        v["z3"] = z3.get_version_string()
    except Exception as e:
        v["z3"] = f"unavailable: {e}"
    return v


# ---------------------------------------------------------------------------
# 1. SymPy symbolic structure
# ---------------------------------------------------------------------------
def sympy_structure() -> dict:
    import sympy
    from sympy import symbols, Poly, expand, factor
    u, v, c = symbols("u v c")

    # injection j_B = u + gamma*u*v
    jB = u + GAMMA * u * v
    jB_factored = factor(jB)

    # scalar 1-round accumulator (ignore Laplacian coupling): treat Cd ~ c
    # C' = MU*c + A_C*c^2 + W*jB + rho  (W, rho concrete-ish; keep symbolic in u,v,c)
    MU, A_C = aspec.MU, aspec.A_C
    Cp = MU * c + A_C * c ** 2 + jB  # drop W, rho (degree-irrelevant scalars)
    Cp_poly_u = Poly(Cp, u)
    Cp_poly_c = Poly(Cp, c)

    # scalar 2-round: feed C' back as c, and a new injection (u2, v2)
    u2, v2 = symbols("u2 v2")
    jB2 = u2 + GAMMA * u2 * v2
    Cpp = MU * Cp + A_C * Cp ** 2 + jB2
    Cpp_deg_c = Poly(expand(Cpp), c).degree()
    Cpp_deg_u = Poly(expand(Cpp), u).degree()

    return {
        "solver": "sympy",
        "injection": "j_B = u + GAMMA*u*v",
        "injection_factored": str(jB_factored),
        "injection_degree_in_u": 1,
        "injection_degree_in_v": 1,
        "scalar_1round_degree_in_u": Cp_poly_u.degree(),
        "scalar_1round_degree_in_c": Cp_poly_c.degree(),
        "scalar_2round_degree_in_c": int(Cpp_deg_c),
        "scalar_2round_degree_in_u": int(Cpp_deg_u),
        "note": "j_B factors as u*(1+GAMMA*v) (degree 1 in u, vs Candidate A's "
                "degree 2). The scalar accumulator self-square A_C*c^2 doubles the "
                "degree in c each round (2 -> 4 -> ...), same as Candidate A; the "
                "injection contributes degree 1 in u (vs A's 2). Lower injection "
                "degree is the structural change.",
    }


# ---------------------------------------------------------------------------
# z3 helpers
# ---------------------------------------------------------------------------
def _z3_injection_tasks() -> dict:
    import z3

    res = {}

    # 2. collision variety exists (SAT)
    s = z3.Solver()
    s.set("timeout", 10000)
    u, v, u2, v2 = z3.Ints("u v u2 v2")
    for var in (u, v, u2, v2):
        s.add(var >= 0, var < P)
    jB = (u + GAMMA * u * v) % P
    jB2 = (u2 + GAMMA * u2 * v2) % P
    s.add(jB == jB2)
    s.add(z3.Or(u != u2, v != v2))
    t0 = time.perf_counter()
    r = s.check()
    res["collision_variety_exists"] = {
        "result": str(r),
        "runtime_s": round(time.perf_counter() - t0, 3),
        "interpretation": "SAT expected: j_B(u,v)=j_B(u',v') is a codimension-1 "
                          "variety; collisions trivially exist in the free (u,v) plane.",
    }

    # 3. computer-assisted: fixed concrete v != v_star, prove injective in u
    s2 = z3.Solver()
    s2.set("timeout", 20000)
    v_fixed = 12345678  # != v_star
    assert v_fixed != V_STAR
    a, b = z3.Ints("a b")
    s2.add(a >= 0, a < P, b >= 0, b < P, a != b)
    s2.add((a + GAMMA * a * v_fixed) % P == (b + GAMMA * b * v_fixed) % P)
    t0 = time.perf_counter()
    r2 = s2.check()
    res["injective_in_u_off_hyperplane"] = {
        "v_fixed": v_fixed,
        "result": str(r2),
        "unsat_means_injective": True,
        "runtime_s": round(time.perf_counter() - t0, 3),
        "interpretation": "UNSAT proves: for this fixed v != v_star, j_B(.,v) is "
                          "injective in u (computer-assisted, single concrete v). "
                          "Generalized symbolically in CC_CORE_V1_RESTRICTED_BINDING.md.",
    }

    # 4. collapse at v_star (SAT: any two distinct u give equal injection 0)
    s3 = z3.Solver()
    s3.set("timeout", 10000)
    a3, b3 = z3.Ints("a3 b3")
    s3.add(a3 >= 0, a3 < P, b3 >= 0, b3 < P, a3 != b3)
    s3.add((a3 + GAMMA * a3 * V_STAR) % P == (b3 + GAMMA * b3 * V_STAR) % P)
    t0 = time.perf_counter()
    r3 = s3.check()
    res["collapse_at_v_star"] = {
        "v_star": V_STAR,
        "result": str(r3),
        "runtime_s": round(time.perf_counter() - t0, 3),
        "interpretation": "SAT: at v=v_star every u gives j_B=0, so any two distinct "
                          "u collide. Confirms the singular collapse.",
    }
    return res


def _z3_coupled_2x2(timeout_ms: int = 30000) -> dict:
    """Attempt a one-round coupled 2x2 collision over the REAL prime with z3."""
    import z3

    p = P
    pm4 = (p - 4) % p

    def lap(cells):
        # 2x2 toroidal laplacian: neighbors wrap; up==down, left==right (N=2 degeneracy)
        out = []
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                up = ((i - 1) % 2) * 2 + j
                dn = ((i + 1) % 2) * 2 + j
                lf = i * 2 + (j - 1) % 2
                rt = i * 2 + (j + 1) % 2
                out.append((cells[up] + cells[dn] + cells[lf] + cells[rt] + pm4 * cells[idx]) % p)
        return out

    def wave(psi):
        out = []
        L = lap(psi)
        for idx in range(4):
            ps = psi[idx]
            sq = (ps * ps) % p
            bm = (aspec.B + p - sq) % p
            out.append((ps + (aspec.D * L[idx]) % p + (aspec.A * ((ps * bm) % p)) % p) % p)
        return out

    def acc(Cf, psi, psin):
        out = []
        Lc = lap(Cf)
        for idx in range(4):
            u = psi[idx]; v = psin[idx]
            j = (u + GAMMA * ((u * v) % p)) % p
            cd = (Cf[idx] + (aspec.D_C * Lc[idx]) % p) % p
            w = (1 + 1 * aspec.WA + (idx + 1) * aspec.WB + 1 * (idx + 1) * aspec.WC) % p
            rho = aspec.RHO0 % p
            out.append((aspec.MU * cd + (aspec.A_C * ((cd * cd) % p)) % p + (w * j) % p + rho) % p)
        return out

    s = z3.Solver()
    s.set("timeout", timeout_ms)
    psi = z3.IntVector("psi", 4)
    Cf = z3.IntVector("C", 4)
    psi2 = z3.IntVector("psiB", 4)
    Cf2 = z3.IntVector("CB", 4)
    for arr in (psi, Cf, psi2, Cf2):
        for x in arr:
            s.add(x >= 0, x < p)
    psin = wave(psi); Cn = acc(Cf, psi, psin)
    psin2 = wave(psi2); Cn2 = acc(Cf2, psi2, psin2)
    for idx in range(4):
        s.add(psin[idx] == psin2[idx])
        s.add(Cn[idx] == Cn2[idx])
    s.add(z3.Or(*[psi[i] != psi2[i] for i in range(4)] + [Cf[i] != Cf2[i] for i in range(4)]))
    t0 = time.perf_counter()
    r = s.check()
    dt = round(time.perf_counter() - t0, 3)
    out = {"result": str(r), "runtime_s": dt, "timeout_ms": timeout_ms,
           "variables": 16, "equations": 8,
           "degree": "wave round degree 3, accumulator degree 2 (in C)"}
    if str(r) == "sat":
        m = s.model()
        out["example_psi"] = [m[psi[i]].as_long() for i in range(4)]
        out["example_C"] = [m[Cf[i]].as_long() for i in range(4)]
        out["example_psiB"] = [m[psi2[i]].as_long() for i in range(4)]
        out["example_CB"] = [m[Cf2[i]].as_long() for i in range(4)]
    out["interpretation"] = (
        "Searches for a one-round coupled collision on a 2x2 toy over the REAL "
        "prime p=2^31-1. 'unknown' = solver timeout (NOT a security result). "
        "N=2 is degenerate; any collision here is not extrapolated to N=16."
    )
    return out


def _z3_preimage_2x2(timeout_ms: int = 30000) -> dict:
    """Attempt to find a one-round 2x2 (psi,C) mapping to a chosen target state."""
    import z3
    p = P
    pm4 = (p - 4) % p
    # target: choose a concrete reachable image by running the map forward, then
    # ask z3 to find ANY preimage (should be SAT -- at least the original).
    import numpy as np
    g = C.rng(99001)
    psi0 = [int(x) for x in g.integers(0, p, size=4)]
    C0 = [int(x) for x in g.integers(0, p, size=4)]

    def lap_np(cells):
        a = np.array(cells, dtype=np.int64).reshape(2, 2)
        return ((np.roll(a, -1, 0) + np.roll(a, 1, 0) + np.roll(a, -1, 1)
                 + np.roll(a, 1, 1) + pm4 * a) % p).reshape(-1)

    def wave_np(psi):
        a = np.array(psi, dtype=np.int64)
        sq = (a * a) % p
        bm = (aspec.B + p - sq) % p
        L = lap_np(psi)
        return ((a + (aspec.D * L) % p + (aspec.A * ((a * bm) % p)) % p) % p).tolist()

    def acc_np(Cf, psi, psin):
        Lc = lap_np(Cf)
        out = []
        for idx in range(4):
            u = psi[idx]; v = psin[idx]
            j = (u + GAMMA * ((u * v) % p)) % p
            cd = (Cf[idx] + (aspec.D_C * Lc[idx]) % p) % p
            w = (1 + aspec.WA + (idx + 1) * aspec.WB + (idx + 1) * aspec.WC) % p
            out.append((aspec.MU * cd + (aspec.A_C * ((cd * cd) % p)) % p
                        + (w * int(j)) % p + aspec.RHO0 % p) % p)
        return out

    psin0 = wave_np(psi0)
    Cn0 = acc_np(C0, psi0, psin0)

    s = z3.Solver()
    s.set("timeout", timeout_ms)
    psi = z3.IntVector("psi", 4)
    Cf = z3.IntVector("C", 4)
    for arr in (psi, Cf):
        for x in arr:
            s.add(x >= 0, x < p)

    def lap_z3(cells):
        out = []
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                up = ((i - 1) % 2) * 2 + j
                dn = ((i + 1) % 2) * 2 + j
                lf = i * 2 + (j - 1) % 2
                rt = i * 2 + (j + 1) % 2
                out.append((cells[up] + cells[dn] + cells[lf] + cells[rt] + pm4 * cells[idx]) % p)
        return out

    Lpsi = lap_z3(psi)
    psin = []
    for idx in range(4):
        ps = psi[idx]
        sq = (ps * ps) % p
        bm = (aspec.B + p - sq) % p
        psin.append((ps + (aspec.D * Lpsi[idx]) % p + (aspec.A * ((ps * bm) % p)) % p) % p)
    Lc = lap_z3(Cf)
    for idx in range(4):
        u = psi[idx]; v = psin[idx]
        j = (u + GAMMA * ((u * v) % p)) % p
        cd = (Cf[idx] + (aspec.D_C * Lc[idx]) % p) % p
        w = (1 + aspec.WA + (idx + 1) * aspec.WB + (idx + 1) * aspec.WC) % p
        Cn = (aspec.MU * cd + (aspec.A_C * ((cd * cd) % p)) % p + (w * j) % p + aspec.RHO0 % p) % p
        s.add(psin[idx] == psin0[idx])
        s.add(Cn == Cn0[idx])
    t0 = time.perf_counter()
    r = s.check()
    dt = round(time.perf_counter() - t0, 3)
    out = {"result": str(r), "runtime_s": dt, "timeout_ms": timeout_ms,
           "target_psi": psin0, "target_C": Cn0}
    if str(r) == "sat":
        m = s.model()
        rec_psi = [m[psi[i]].as_long() for i in range(4)]
        out["recovered_psi"] = rec_psi
        out["recovered_is_original"] = rec_psi == psi0
    out["interpretation"] = (
        "Preimage of a known-reachable target on the 2x2 toy over the real prime. "
        "SAT is expected (the original is a preimage); the interest is whether z3 "
        "finds it quickly and whether multiple preimages exist. 'unknown' = timeout."
    )
    return out


def main(seed: int = 99000) -> dict:
    t0 = time.perf_counter()
    versions = _solver_versions()
    print("  solver versions:", versions)

    result = {
        "artifact": "candidate_b_algebraic_solver",
        "description": "Exact algebraic solver (SymPy + z3) attacks on reduced Candidate B",
        "metadata": C.env_metadata(),
        "seed": seed,
        "solver_versions": versions,
    }

    if "unavailable" not in versions.get("sympy", ""):
        print("  sympy structure ...")
        result["sympy_structure"] = sympy_structure()

    if "unavailable" not in versions.get("z3", ""):
        print("  z3 injection tasks ...")
        result["z3_injection"] = _z3_injection_tasks()
        print("    ", {k: v["result"] for k, v in result["z3_injection"].items()})
        print("  z3 coupled 2x2 collision (real prime, timeout) ...")
        result["z3_coupled_2x2_collision"] = _z3_coupled_2x2()
        print("    result:", result["z3_coupled_2x2_collision"]["result"])
        print("  z3 preimage 2x2 (real prime, timeout) ...")
        result["z3_preimage_2x2"] = _z3_preimage_2x2()
        print("    result:", result["z3_preimage_2x2"]["result"])

    result["summary"] = {
        "injective_in_u_off_hyperplane_z3":
            result.get("z3_injection", {}).get("injective_in_u_off_hyperplane", {}).get("result"),
        "coupled_2x2_collision":
            result.get("z3_coupled_2x2_collision", {}).get("result"),
        "note": "A timeout ('unknown') is NOT a security result. Full N=16, T=32 "
                "systems are far beyond solver reach; only reduced/restricted "
                "systems were attempted.",
    }
    result["limitations"] = [
        "Only 2x2 toy and single-coordinate injection systems were tractable",
        "Full N=16 / T=32 system is not encodable for any available solver",
        "z3 'unknown' results are timeouts, not proofs",
        "N=2 collisions (if any) are degenerate and not extrapolated",
    ]
    result["runtime_s"] = round(time.perf_counter() - t0, 2)
    C.save_artifact("candidate_b_algebraic_solver.json", result)
    print(f"  saved candidate_b_algebraic_solver.json ({result['runtime_s']}s)")
    return result


if __name__ == "__main__":
    main()
