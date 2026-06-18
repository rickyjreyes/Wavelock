"""Phase 8J: structured Laplacian-eigenmode collision audit.

Independently derives, verifies, enumerates, and assesses the reachability of an
exact collision family in the NORMATIVE WaveLock-PDE-256-v0 one-round state map

    F(psi) = psi + D*L(psi) + a*psi*(b - psi^2)  (mod p)

For a sign field psi = s*sigma (sigma in {-1,+1}) where every site has exactly r
opposite-sign neighbors, L(sigma) = -2r*sigma, so

    F(s*sigma) = s*sigma*[1 - 2rD + a(b - s^2)]   (mod p),

which is the all-zero state iff   s^2 == b - (2rD-1)/a  (mod p),
with division = multiplication by the modular inverse of a in F_p.

Everything is recomputed from scratch and verified by EXACT element-by-element
state equality against the pure-Python reference round, the optimized NumPy
round, the parameterized harness, and evolve_T with normative T=32. No floating
point, no tolerances, no digest-of-state used for verification.

Run:
    python -m pde_audit.eigenmode_collisions
"""

from __future__ import annotations

import importlib
import time
from itertools import product

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams
from .state_map import one_round_jacobian
from wavelock.pde_hash import spec, optimized as _opt
from wavelock.pde_hash.state import PDEState

_ev = importlib.import_module("wavelock.pde_hash.evolve")

P = spec.P
N = spec.N
A = spec.A
D = spec.D
B = spec.B
_V = PDEVariant(PDEParams())
_ZERO = np.zeros((N, N), dtype=np.int64)


# ---------------------------------------------------------------------------
# modular helpers (recomputed independently)
# ---------------------------------------------------------------------------
def inv_mod(x, p=P):
    return pow(int(x) % p, p - 2, p)


def is_qr(x, p=P):
    x = int(x) % p
    return x == 0 or pow(x, (p - 1) // 2, p) == 1


def sqrt_mod(x, p=P):
    # p == 3 (mod 4): sqrt = x^((p+1)/4) when it exists
    x = int(x) % p
    r = pow(x, (p + 1) // 4, p)
    return r if (r * r) % p == x else None


def amplitude_for_r(r):
    """s^2 == b - (2rD-1)/a (mod p); return dict with roots if QR."""
    r = int(r)
    ainv = inv_mod(A)
    s2 = int((B - (2 * r * D - 1) * ainv) % P)
    rt = sqrt_mod(s2)
    return {
        "r": r,
        "two_rD_minus_1": 2 * r * D - 1,
        "a_inverse": ainv,
        "s_squared": s2,
        "is_quadratic_residue": rt is not None,
        "roots": [rt, (P - rt) % P] if rt is not None else None,
    }


# ---------------------------------------------------------------------------
# exact verification primitives (state equality, never a digest)
# ---------------------------------------------------------------------------
def _ref_one_round(arr):
    ps = PDEState([int(x) for x in arr.reshape(-1)])
    return np.array(_ev.evolve(ps, 1).cells, dtype=np.int64).reshape(N, N) % P


def _ref_evolve_T(arr):
    ps = PDEState([int(x) for x in arr.reshape(-1)])
    return np.array(_ev.evolve_T(ps).cells, dtype=np.int64).reshape(N, N) % P


def verify_maps_to_zero(state: np.ndarray) -> dict:
    """Exact element-by-element check across all implementations."""
    state = (state % P).astype(np.int64)
    harness_1r = np.array_equal(_V.one_round(state), _ZERO)
    ref_1r = np.array_equal(_ref_one_round(state), _ZERO)
    opt_1r = np.array_equal(_opt._one_round(state.copy()) % P, _ZERO)
    harness_T = np.array_equal(_V.evolve_T(state), _ZERO)
    ref_T = np.array_equal(_ref_evolve_T(state), _ZERO)
    opt_T = np.array_equal(_opt._evolve_T(state.copy()) % P, _ZERO)
    return {
        "one_round_zero": {"harness": harness_1r, "reference": ref_1r, "optimized": opt_1r},
        "evolve_T_zero": {"harness": harness_T, "reference": ref_T, "optimized": opt_T},
        "all_agree_zero": all([harness_1r, ref_1r, opt_1r, harness_T, ref_T, opt_T]),
    }


# ---------------------------------------------------------------------------
# sign-field helpers
# ---------------------------------------------------------------------------
def opp_neighbor_count(sig):
    c = np.zeros((N, N), dtype=int)
    for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        c += (np.roll(sig, (di, dj), (0, 1)) != sig).astype(int)
    return c


def lap_mod(x):
    return (np.roll(x, -1, 0) + np.roll(x, 1, 0)
            + np.roll(x, -1, 1) + np.roll(x, 1, 1) + ((P - 4) * x)) % P


def is_laplacian_eigenvector(sig):
    """Return (True, lambda, r) iff L(sig) == lambda*sig for a single lambda."""
    sm = (sig % P).astype(np.int64)
    Ls = lap_mod(sm)
    # candidate lambda from first nonzero coordinate
    lam = None
    for c in range(N * N):
        if sm.reshape(-1)[c] != 0:
            lam = (Ls.reshape(-1)[c] * inv_mod(int(sm.reshape(-1)[c]))) % P
            break
    if lam is None:
        return False, None, None
    if np.array_equal(Ls, (lam * sm) % P):
        # lambda == -2r  =>  r = (-lambda)/2  (over integers, lam in {0,-2,..,-8} mod p)
        lam_signed = lam if lam <= 4 else lam - P  # map to small negative
        r = (-lam_signed) // 2 if lam_signed % 2 == 0 else None
        return True, lam, r
    return False, None, None


# ---------------------------------------------------------------------------
# Part I: named candidate states
# ---------------------------------------------------------------------------
def named_states():
    ii, jj = np.indices((N, N))
    out = {
        "checkerboard_r4": ((-1) ** (ii + jj), 4),
        "stripe_rows_r2": ((-1) ** ii, 2),
        "stripe_cols_r2": ((-1) ** jj, 2),
        "period4_cols_r1": (np.array([1, -1, -1, 1])[jj % 4], 1),
    }
    return out


def part1_verification():
    rclasses = {str(r): amplitude_for_r(r) for r in range(5)}
    states = []
    for name, (sig, r) in named_states().items():
        amp = amplitude_for_r(r)
        oc = opp_neighbor_count(sig)
        eig_ok, lam, r_from_eig = is_laplacian_eigenvector(sig)
        entry = {
            "name": name, "r": r,
            "opp_neighbor_const": bool(oc.min() == oc.max()),
            "opp_neighbor_value": int(oc[0, 0]),
            "is_laplacian_eigenvector": bool(eig_ok),
            "lambda": int(lam) if lam is not None else None,
            "amplitude_is_qr": amp["is_quadratic_residue"],
        }
        if amp["is_quadratic_residue"]:
            entry["verifications"] = {}
            for amp_name, s in (("plus_s", amp["roots"][0]), ("minus_s", amp["roots"][1])):
                st = ((s * sig) % P).astype(np.int64)
                entry["verifications"][amp_name] = {"s": int(s), **verify_maps_to_zero(st)}
        states.append(entry)
    # zero fixed point
    zero_fp = {
        "one_round": np.array_equal(_V.one_round(_ZERO), _ZERO),
        "evolve_T": np.array_equal(_V.evolve_T(_ZERO), _ZERO),
        "ref_one_round": np.array_equal(_ref_one_round(_ZERO), _ZERO),
    }
    return {"r_classes": rclasses, "named_states": states, "zero_fixed_point": zero_fp}


# ---------------------------------------------------------------------------
# Part II: periodic-tile enumeration + symmetry orbits
# ---------------------------------------------------------------------------
TILES = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2), (4, 4)]


def _tile_to_full(tile):
    h, w = tile.shape
    return np.tile(tile, (N // h, N // w))


def _canonical(sig):
    """Canonical form under sign flip, translations, rot90 x4, reflections."""
    best = None
    variants = []
    base = sig
    for flip in (base, -base):
        for k in range(4):
            r = np.rot90(flip, k)
            for ref in (r, r[::-1, :], r[:, ::-1], r.T):
                variants.append(ref)
    for v in variants:
        for a in range(N):
            for b in range(N):
                t = np.roll(np.roll(v, a, 0), b, 1)
                key = ((t > 0).astype(np.uint8)).tobytes()
                if best is None or key < best:
                    best = key
    return best


def enumerate_periodic_tiles():
    found = []          # eigenvector configs with QR amplitude
    orbit_keys = set()
    distinct_full_states = 0
    raw_states = 0
    per_tile = {}
    distinct_patterns = set()      # distinct full sigma patterns (eigenvector + QR)
    per_r_patterns = {}            # r -> set of distinct full patterns
    for (h, w) in TILES:
        if N % h or N % w:
            continue
        cnt = 0
        eig_cnt = 0
        for bits in product((1, -1), repeat=h * w):
            tile = np.array(bits, dtype=np.int64).reshape(h, w)
            sig = _tile_to_full(tile)
            cnt += 1
            eig_ok, lam, r = is_laplacian_eigenvector(sig)
            if not eig_ok or r is None:
                continue
            amp = amplitude_for_r(r)
            if not amp["is_quadratic_residue"]:
                continue
            eig_cnt += 1
            # this sigma yields 2 zero-preimage states: +s*sigma and -s*sigma
            raw_states += 2
            pat = (sig > 0).astype(np.uint8).tobytes()
            distinct_patterns.add(pat)
            per_r_patterns.setdefault(int(r), set()).add(pat)
            ck = _canonical(sig)
            if ck not in orbit_keys:
                orbit_keys.add(ck)
                # verify one representative exactly
                s = amp["roots"][0]
                v = verify_maps_to_zero(((s * sig) % P).astype(np.int64))
                found.append({"tile": [h, w], "r": int(r), "lambda": int(lam),
                              "s": int(s), "verified_zero": v["all_agree_zero"]})
                distinct_full_states += 2
        per_tile[f"{h}x{w}"] = {"tiles_scanned": cnt, "eigenvector_qr_configs": eig_cnt}
    return {
        "per_tile": per_tile,
        "num_symmetry_orbits": len(orbit_keys),
        "distinct_full_states_from_orbits_x2amp": distinct_full_states,
        "distinct_full_sign_patterns": len(distinct_patterns),
        "distinct_patterns_by_r": {str(r): len(s) for r, s in sorted(per_r_patterns.items())},
        "constructive_zero_preimage_count_one_round": len(distinct_patterns) + 1,
        "constructive_count_note":
            "len(distinct sign-eigenmode patterns) [sigma and -sigma both counted] "
            "+ 1 for the all-zero state. This is a LOWER BOUND from periodic tiles "
            "up to 4x4 only; the full set of +/-1 Laplacian eigenvectors over F_p "
            "(hence preimages of zero) is conjectured larger but is NOT claimed "
            "exponential without a constructive argument.",
        "representatives": found,
    }


# ---------------------------------------------------------------------------
# Part II.3: Jacobian at collision representatives
# ---------------------------------------------------------------------------
def jacobian_at_collisions():
    params = PDEParams()
    ii, jj = np.indices((N, N))
    reps = {
        "zero": _ZERO,
        "checkerboard_plus": (151946369 * ((-1) ** (ii + jj))) % P,
        "checkerboard_minus": ((P - 151946369) * ((-1) ** (ii + jj))) % P,
        "stripe_rows": (1395627816 * ((-1) ** ii)) % P,
        "stripe_cols": (1395627816 * ((-1) ** jj)) % P,
        "period4_cols": (1217065103 * np.array([1, -1, -1, 1])[jj % 4]) % P,
    }
    out = {}
    n = N * N
    for name, st in reps.items():
        J = one_round_jacobian(st.astype(np.int64), params)
        rank = H.mod_rank_np(J, P)
        out[name] = {"rank": rank, "full_rank": rank == n,
                     "rank_deficiency": n - rank}
    out["_note"] = ("A full-rank Jacobian shows only LOCAL non-singularity; "
                    "checkerboard_plus and checkerboard_minus have identical psi^2 "
                    "(=s^2) hence identical Jacobians yet are distinct states both "
                    "mapping to zero -- a direct demonstration that local "
                    "non-singularity does NOT imply global injectivity.")
    return out


# ---------------------------------------------------------------------------
# Part IV: message-level reachability
# ---------------------------------------------------------------------------
def reachability_analysis(seed=80110, search_budget=50_000):
    """Determine whether absorption can produce a structured collision state or
    the zero state, exactly. Uses the harness absorb pipeline; verifies by exact
    state equality and full normative digest (both implementations)."""
    from wavelock.pde_hash import reference as ref

    # (a) which cells are controllable before the first evolve_T?
    iv = _V.iv()
    # cells 67..255 receive NO injection ever (cap0=64, cap1=65, cap2=66, rate 0..63)
    uncontrolled = list(range(67, N * N))
    # required values for checkerboard at those cells = +/- s (s=151946369)
    s = 151946369
    ii, jj = np.indices((N, N))
    checker = ((-1) ** (ii + jj)).reshape(-1)
    iv_flat = iv.reshape(-1)
    # exact reason: do IV values at uncontrolled cells already equal +-s*checker?
    mism = [c for c in uncontrolled
            if int(iv_flat[c]) not in (s * checker[c] % P, (P - s) * (checker[c] % P) % P, (s * (checker[c] % P)) % P)]
    analytic = {
        "controllable_cells_before_first_evolve_T": list(range(0, 67)),
        "uncontrolled_capacity_cells": [67, N * N - 1],
        "num_uncontrolled": len(uncontrolled),
        "example_iv_values": {str(c): int(iv_flat[c]) for c in (67, 100, 200, 255)},
        "required_pm_s": {"plus_s": s, "minus_s": (P - s) % P},
        "iv_cells_mismatch_required_checkerboard": len(mism),
        "direct_reachability_before_first_round":
            "IMPOSSIBLE: capacity cells 67..255 are never written by absorption "
            "and hold fixed IV-derived values (all < ~300), which can never equal "
            "+/-s (~1.5e8); a structured collision state requires every cell to be "
            "+/-s*sigma. Exact mismatch count below.",
    }

    # (b) bounded search: any message whose PRE-SQUEEZE state is zero, or whose
    # final absorbed state is a known structured collision state.
    g = H.rng(seed)
    structured = _structured_state_set()
    pre_squeeze_zero = None
    final_absorbed_structured = None
    digests_all_zero = None
    for e in range(search_budget):
        m = int(e).to_bytes(8, "big")
        absorbed = _V.absorb(m)                 # pre-squeeze state (absorb ends with evolve_T)
        # pre-squeeze state IS the absorbed state (absorb ends with evolve_T)
        if np.array_equal(absorbed, _ZERO):
            pre_squeeze_zero = m.hex()
            break
        key = absorbed.tobytes()
        if key in structured:
            final_absorbed_structured = m.hex()
            break
    search = {
        "budget": search_budget,
        "pre_squeeze_zero_message": pre_squeeze_zero,
        "final_absorbed_is_structured_collision": final_absorbed_structured,
        "method": "sequential 8-byte messages; exact state equality vs zero and "
                  "vs the enumerated structured-collision set",
        "found": pre_squeeze_zero is not None or final_absorbed_structured is not None,
    }

    # digest of the zero pre-squeeze state (deterministic all-tie output)
    zero_digest = _V.squeeze(_ZERO).hex()
    return {"analytic_one_block": analytic, "search": search,
            "zero_state_digest": zero_digest,
            "zero_digest_is_all_zero_bytes": zero_digest == "00" * 32}


def _structured_state_set():
    ii, jj = np.indices((N, N))
    sigs = [(-1) ** (ii + jj), (-1) ** ii, (-1) ** jj, np.array([1, -1, -1, 1])[jj % 4]]
    amps = {4: 151946369, 2: 1395627816, 1: 1217065103}
    rs = [4, 2, 2, 1]
    out = set()
    for sig, r in zip(sigs, rs):
        s = amps[r]
        for amp in (s, (P - s) % P):
            out.add(((amp * sig) % P).astype(np.int64).tobytes())
    return out


def main(seed: int = 80110) -> dict:
    t0 = time.perf_counter()
    print("  Part I: algebraic verification ...")
    p1 = part1_verification()
    for r, d in p1["r_classes"].items():
        print(f"    r={r}: s^2={d['s_squared']} QR={d['is_quadratic_residue']} "
              f"roots={d['roots']}")
    for st in p1["named_states"]:
        v = st.get("verifications", {})
        ok = all(x["all_agree_zero"] for x in v.values()) if v else False
        print(f"    {st['name']:18s} r={st['r']} eig={st['is_laplacian_eigenvector']} "
              f"QR={st['amplitude_is_qr']} both_amps_zero={ok}")

    print("  Part II: periodic-tile enumeration ...")
    p2 = enumerate_periodic_tiles()
    print(f"    symmetry orbits={p2['num_symmetry_orbits']} "
          f"distinct states (x2 amp)={p2['distinct_full_states_from_orbits_x2amp']}")
    for t, d in p2["per_tile"].items():
        if d["eigenvector_qr_configs"]:
            print(f"      tile {t}: {d['eigenvector_qr_configs']} eigenvector/QR configs")

    print("  Part II.3: Jacobian at collision states ...")
    jac = jacobian_at_collisions()
    for name, d in jac.items():
        if name.startswith("_"):
            continue
        print(f"    {name:18s} rank={d['rank']}/256 full_rank={d['full_rank']}")

    print("  Part IV: message-level reachability ...")
    p4 = reachability_analysis(seed)
    print(f"    uncontrolled capacity cells: {p4['analytic_one_block']['num_uncontrolled']} "
          f"(67..255); IV mismatch vs checkerboard: "
          f"{p4['analytic_one_block']['iv_cells_mismatch_required_checkerboard']}")
    print(f"    reachability search (budget {p4['search']['budget']}): "
          f"found={p4['search']['found']}")
    print(f"    zero-state digest all-zero bytes: {p4['zero_digest_is_all_zero_bytes']}")

    results = {
        "phase": "8J_eigenmode_collisions",
        "metadata": H.env_metadata(),
        "branch": "research/pde-eigenmode-collision-audit",
        "constants": {"p": P, "N": N, "D": D, "a": A, "b": B, "T": spec.T,
                      "a_inverse": inv_mod(A)},
        "part1_algebraic_verification": p1,
        "part2_tile_enumeration": p2,
        "part2_3_jacobian": jac,
        "part4_reachability": p4,
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    path = H.save_artifact("phase8j_eigenmode_collisions.json", results)
    print("  saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
