"""Phase 8K: multi-block message-reachability and collision-lifting audit.

Builds an exact, instrumented multi-block trace of the WaveLock-PDE-256-v0
absorption pipeline and uses it to study whether a valid message can steer the
absorbed state onto a structured collision state, the all-zero state, or any
pair of internally colliding states. Two control models are separated:

  Model A (relaxed): each of the 64 rate inputs is an arbitrary F_p element
                     (a structural upper bound, NOT a valid message);
  Model B (bytes):   real packing e = b0 + 256 b1 + 65536 b2 in [0, 2**24),
                     with padding / length / counter / finalization.

Everything operates on raw state / raw 256-bit output. The normative primitive
is NOT modified; this module reproduces it exactly (parity-tested).

Run:
    python -m pde_audit.multiblock_reachability
"""

from __future__ import annotations

import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams, matmul_mod, mod_rank_np, mod_solve
from .state_map import one_round_jacobian
from . import eigenmode_collisions as ec
from wavelock.pde_hash import spec, optimized as _opt, reference as _ref

P, N, G, T = spec.P, spec.N, spec.G, spec.T
NC = N * N
RATE = spec.RATE
CAP0, CAP1, CAP2 = spec.CAP0, spec.CAP1, spec.CAP2
_V = PDEVariant(PDEParams())
_ZERO = np.zeros((N, N), dtype=np.int64)


# ---------------------------------------------------------------------------
# Part II: exact multi-block trace
# ---------------------------------------------------------------------------
def pad_to_blocks(message: bytes):
    """Return list of (64,) int64 packed element arrays for the padded message."""
    m = bytes(message)
    out = bytearray(m)
    out.append(1)
    out.extend(b"\x00" * ((-len(out)) % spec.BYTES_PER_BLOCK))
    lb = bytearray(spec.BYTES_PER_BLOCK)
    lb[0:8] = (len(m) * 8).to_bytes(8, "big")
    out.extend(lb)
    arr = np.frombuffer(bytes(out), dtype=np.uint8).astype(np.int64)
    nb = arr.size // spec.BYTES_PER_BLOCK
    arr = arr.reshape(nb, RATE, 3)
    return [arr[k, :, 0] + (arr[k, :, 1] << 8) + (arr[k, :, 2] << 16) for k in range(nb)]


def inject(state_flat, elems, k, nb):
    """U = S + I(B_k, k, final)  (exact, returns new flat array)."""
    u = state_flat.copy()
    u[:RATE] = (u[:RATE] + elems) % P
    q0, q1 = spec.encode_block_counter(k)
    u[CAP0] = (u[CAP0] + q0 * G) % P
    u[CAP2] = (u[CAP2] + q1 * G) % P
    if k == nb - 1:
        u[CAP1] = (u[CAP1] + spec.D_TAG) % P
    return u


def block_trace(message: bytes, capture_rounds=False) -> dict:
    """Exact per-block (and optional per-round) trace of absorption."""
    blocks = pad_to_blocks(message)
    nb = len(blocks)
    S = _V.iv().reshape(-1).astype(np.int64)
    trace = {"n_blocks": nb, "blocks": []}
    for k in range(nb):
        entry = {"S_before": S.copy()}
        U = inject(S, blocks[k], k, nb)
        entry["U_after_injection"] = U.copy()
        psi = U.reshape(N, N)
        if capture_rounds:
            rounds = []
            for _ in range(T):
                psi = _V.one_round(psi)
                rounds.append(psi.reshape(-1).copy())
            entry["rounds"] = rounds
            S = psi.reshape(-1)
        else:
            S = _V.evolve_T(psi).reshape(-1)
        entry["S_after"] = S.copy()
        trace["blocks"].append(entry)
    trace["pre_squeeze"] = S.copy()
    return trace


# ---------------------------------------------------------------------------
# Part IV: first-block impossibility, strengthened over all eigenmodes
# ---------------------------------------------------------------------------
def first_block_impossibility() -> dict:
    """For every enumerated eigenmode (and its symmetry variants), the pre-first-
    round state U_0 has cells 67..255 fixed to IV values; count mismatches."""
    iv = _V.iv().reshape(-1)
    uncontrolled = np.arange(67, NC)
    # gather all distinct eigenmode sign patterns from tile enumeration
    patterns = _all_eigenmode_patterns()
    results = []
    min_mismatch = NC
    for pat_bits, r in patterns:
        sig = (np.frombuffer(pat_bits, dtype=np.uint8).astype(np.int64).reshape(N, N) * 2 - 1)
        amp = ec.amplitude_for_r(r)
        s = amp["roots"][0]
        # try all symmetry variants + both amplitudes to MINIMIZE mismatch
        best = NC
        for sg in _symmetry_variants(sig):
            for ss in (s, (P - s) % P):
                req = ((ss * sg) % P).reshape(-1)
                mm = int(np.count_nonzero(iv[uncontrolled] != req[uncontrolled]))
                best = min(best, mm)
        results.append({"r": r, "min_uncontrolled_mismatch": best})
        min_mismatch = min(min_mismatch, best)
    return {
        "num_eigenmode_patterns_checked": len(patterns),
        "uncontrolled_cells": [67, NC - 1],
        "min_uncontrolled_mismatch_over_all_variants": min_mismatch,
        "iv_values_at_uncontrolled_max": int(iv[uncontrolled].max()),
        "eigenmode_amplitudes": {"r1": ec.amplitude_for_r(1)["roots"],
                                  "r2": ec.amplitude_for_r(2)["roots"],
                                  "r4": ec.amplitude_for_r(4)["roots"]},
        "proof": ("Every uncontrolled cell 67..255 holds an IV value <= "
                  f"{int(iv[uncontrolled].max())}, while every eigenmode requires "
                  "+/- s (s in {151946369,1217065103,1395627816,...} ~1e8-1e9). "
                  "Since min mismatch over ALL symmetry/sign variants is "
                  f"{min_mismatch} > 0, no eigenmode state can equal U_0 for any "
                  "one-block message: direct first-block reachability is IMPOSSIBLE."),
        "by_pattern_sample": results[:6],
    }


def _all_eigenmode_patterns():
    from itertools import product
    out = {}
    for (h, w) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2), (4, 4)]:
        if N % h or N % w:
            continue
        for bits in product((1, -1), repeat=h * w):
            sig = np.tile(np.array(bits, dtype=np.int64).reshape(h, w), (N // h, N // w))
            ok, lam, r = ec.is_laplacian_eigenvector(sig)
            if ok and r is not None and ec.amplitude_for_r(r)["is_quadratic_residue"]:
                out[((sig > 0).astype(np.uint8)).tobytes()] = int(r)
    return list(out.items())


def _symmetry_variants(sig):
    seen = []
    for flip in (sig, -sig):
        for k in range(4):
            r = np.rot90(flip, k)
            for v in (r, r[::-1, :], r[:, ::-1]):
                seen.append(v.copy())
    return seen


# ---------------------------------------------------------------------------
# Part VI: differential controllability of capacity coordinates
# ---------------------------------------------------------------------------
def _jac_columns(start_flat, M0):
    """Return J_{evolve_T}(start) @ M0  (256 x cols), exact mod p, via chain rule
    applied column-block-wise; also return the evolved state."""
    params = PDEParams()
    psi = start_flat.reshape(N, N).astype(np.int64)
    M = np.asarray(M0, dtype=np.int64) % P
    for _ in range(T):
        J = one_round_jacobian(psi, params)
        M = matmul_mod(J, M, P)
        psi = _V.one_round(psi)
    return M, psi.reshape(-1)


def differential_controllability(seed=80120, n_blocks_max=3) -> dict:
    """Rank over F_p of the map (block rate inputs) -> S_k[67:255]."""
    g = H.rng(seed)
    iv = _V.iv().reshape(-1)
    E = np.zeros((NC, RATE), dtype=np.int64)
    for c in range(RATE):
        E[c, c] = 1                      # selection: rate cells 0..63
    cap_rows = np.arange(67, NC)         # 189 capacity coordinates
    out = {"capacity_target_dim": len(cap_rows), "by_blocks": {}}

    # choose a concrete random message trajectory of n_blocks blocks
    # (relaxed model: random F_p rate vectors; Jacobian is state-dependent)
    states_U = []
    S = iv.copy()
    nb = n_blocks_max
    rate_inputs = [g.integers(0, P, size=RATE, dtype=np.int64) for _ in range(nb)]
    for k in range(nb):
        U = S.copy()
        U[:RATE] = (U[:RATE] + rate_inputs[k]) % P
        q0, q1 = spec.encode_block_counter(k)
        U[CAP0] = (U[CAP0] + q0 * G) % P
        U[CAP2] = (U[CAP2] + q1 * G) % P
        states_U.append(U.copy())
        S = _V.evolve_T(U.reshape(N, N)).reshape(-1)   # advance trajectory only

    # Build cumulative Jacobian of S_k w.r.t. all earlier block inputs.
    # dS_1/db0 = K0.  For dS_2: db0 -> J(U_1)@(dU_1/db0)=J(U_1)@(dS_1/db0); but we
    # already folded J(U_1) into K1 only for db1. Recompute multi-block columns:
    for nblk in range(1, n_blocks_max + 1):
        # columns for S_nblk wrt blocks 0..nblk-1, each 256x64
        cols = []
        # propagate each block's selection through subsequent evolve_T's
        for src in range(nblk):
            M = E.copy()
            # block src enters U_src rate; then evolve_T through blocks src..nblk-1
            for kk in range(src, nblk):
                M, _ = _jac_columns(states_U[kk], M)
            cols.append(M)
        full = np.concatenate(cols, axis=1)          # 256 x (64*nblk)
        sub = full[cap_rows, :]                       # 189 x (64*nblk)
        rank = mod_rank_np(sub, P)
        out["by_blocks"][str(nblk)] = {
            "input_vars": RATE * nblk,
            "capacity_rank": int(rank),
            "rank_saturated_to_189": rank == len(cap_rows),
            "deficiency": int(len(cap_rows) - rank),
        }
    out["note"] = ("Differential (local, linearized) controllability only. Full "
                   "differential rank != global reachability; rank below 189 means "
                   "the capacity target is locally unreachable with that many "
                   "blocks.")
    return out


# ---------------------------------------------------------------------------
# Part VI.2: linearized (modular-Newton) steering of capacity coords (Model A)
# ---------------------------------------------------------------------------
def linearized_steering(seed=80120, n_prior=3, iters=6) -> dict:
    """Use the rank-189 differential map to try to drive S_{n_prior}[67:255] onto
    the checkerboard capacity pattern via modular linear solves on the rate
    inputs of the prior blocks (Model A, relaxed F_p). Each step is re-evaluated
    through the EXACT nonlinear pipeline. Reports residual vs a random baseline.

    Over F_p there is no valuation/Hensel descent, so a linearized step need not
    reduce the nonlinear residual; this measures whether it does in practice.
    """
    g = H.rng(seed)
    iv = _V.iv().reshape(-1)
    target = _checker_target()
    cap = np.arange(67, NC)
    E = np.zeros((NC, RATE), dtype=np.int64)
    for c in range(RATE):
        E[c, c] = 1

    # initial random Model-A rate inputs for the prior blocks
    rate = [g.integers(0, P, size=RATE, dtype=np.int64) for _ in range(n_prior)]

    def evolve_traj(rate_list):
        S = iv.copy()
        Us = []
        for k in range(n_prior):
            U = S.copy()
            U[:RATE] = (U[:RATE] + rate_list[k]) % P
            q0, q1 = spec.encode_block_counter(k)
            U[CAP0] = (U[CAP0] + q0 * G) % P
            U[CAP2] = (U[CAP2] + q1 * G) % P
            Us.append(U.copy())
            S = _V.evolve_T(U.reshape(N, N)).reshape(-1)
        return S, Us

    def residual(S):
        return int(np.count_nonzero(S[cap] != target[cap]))

    S, Us = evolve_traj(rate)
    traj = [residual(S)]
    reached = False
    for _ in range(iters):
        # capacity Jacobian of S_{n_prior} wrt all prior rate vars (189 x 64*n_prior)
        cols = []
        for src in range(n_prior):
            M = E.copy()
            for kk in range(src, n_prior):
                M, _ = _jac_columns(Us[kk], M)
            cols.append(M)
        Jcap = np.concatenate(cols, axis=1)[cap, :]
        rhs = (target[cap] - S[cap]) % P
        delta = mod_solve(Jcap, rhs, P)
        if delta is None:
            traj.append(traj[-1])
            break
        for k in range(n_prior):
            rate[k] = (rate[k] + delta[k * RATE:(k + 1) * RATE]) % P
        S, Us = evolve_traj(rate)
        traj.append(residual(S))
        if traj[-1] == 0:
            reached = True
            break

    # random baseline: best residual from equal number of random trajectories
    base = NC
    for _ in range(iters + 1):
        rr = [g.integers(0, P, size=RATE, dtype=np.int64) for _ in range(n_prior)]
        Sb, _ = evolve_traj(rr)
        base = min(base, residual(Sb))

    return {"model": "A_relaxed", "n_prior_blocks": n_prior, "iters": iters,
            "capacity_target_dim": int(len(cap)),
            "residual_trajectory": traj, "reached_capacity_target": reached,
            "random_baseline_best_residual": base,
            "note": ("modular-Newton on the rank-189 differential map; F_p has no "
                     "valuation so a linearized step need not decrease the "
                     "nonlinear residual -- convergence is the empirical question.")}


# ---------------------------------------------------------------------------
# Part V/IX: bounded heuristic searches (Model B bytes + Model A relaxed)
# ---------------------------------------------------------------------------
def _checker_target():
    ii, jj = np.indices((N, N))
    s = 151946369
    return ((s * ((-1) ** (ii + jj))) % P).reshape(-1)


def search_targets(seed=80121, budget=20000) -> dict:
    """Model B byte-constrained search over messages, measuring exact-match
    objectives. Success = exact equality only."""
    g = H.rng(seed)
    target = _checker_target()
    structured = ec._structured_state_set() if hasattr(ec, "_structured_state_set") else set()
    best_cap_match = 0
    best_zero_dist = NC
    best_digest_hd = 256
    found = {"pre_squeeze_zero": None, "structured_state": None,
             "digest_all_zero": None, "exact_eigenmode_U": None}
    for e in range(budget):
        # byte-constrained message of 1..3 blocks
        nblk = 1 + (e % 3)
        msg = g.integers(0, 256, size=nblk * spec.BYTES_PER_BLOCK - 1, dtype=np.uint8).tobytes()
        tr = block_trace(msg)
        pre = tr["pre_squeeze"]
        # objective 5: pre-squeeze distance to zero (nonzero-coordinate count)
        zd = int(np.count_nonzero(pre))
        best_zero_dist = min(best_zero_dist, zd)
        if zd == 0:
            found["pre_squeeze_zero"] = msg.hex()
            break
        # objective 6: digest HD to 00..00
        dig = _V.squeeze(pre.reshape(N, N))
        hd = sum(bin(x).count("1") for x in dig)
        best_digest_hd = min(best_digest_hd, hd)
        if hd == 0:
            found["digest_all_zero"] = msg.hex()
            break
        # objective 1/3: capacity-match of last U to checkerboard target
        U_last = tr["blocks"][-1]["U_after_injection"]
        cm = int(np.count_nonzero(U_last[67:] == target[67:]))
        best_cap_match = max(best_cap_match, cm)
        if U_last.tobytes() in structured:
            found["structured_state"] = msg.hex()
            break
    return {
        "model": "B_byte_constrained", "budget": budget,
        "best_capacity_coord_matches_to_checker(of 189)": best_cap_match,
        "best_pre_squeeze_nonzero_coords(min, of 256)": best_zero_dist,
        "best_digest_hamming_to_zero(min, of 256)": best_digest_hd,
        "found": found,
        "any_exact_lift": any(v is not None for v in found.values()),
    }


def relaxed_one_block_zero(seed=80122, budget=20000) -> dict:
    """Model A relaxed: arbitrary F_p rate vector in ONE block; can evolve_T(U_0)
    hit zero? (capacity cells 67..255 stay at IV; only 64 rate + cap0/cap2 move)."""
    g = H.rng(seed)
    iv = _V.iv().reshape(-1)
    best = NC
    hit = None
    for e in range(budget):
        U = iv.copy()
        U[:RATE] = g.integers(0, P, size=RATE, dtype=np.int64)
        # cap0/cap2 also fixed by counter for a real block; leave as IV+counter(0)
        q0, q1 = spec.encode_block_counter(0)
        U[CAP0] = (U[CAP0] + q0 * G) % P
        U[CAP2] = (U[CAP2] + q1 * G) % P
        S = _V.evolve_T(U.reshape(N, N)).reshape(-1)
        nz = int(np.count_nonzero(S))
        if nz < best:
            best = nz
        if nz == 0:
            hit = [int(x) for x in U[:RATE]]
            break
    return {"model": "A_relaxed_one_block", "budget": budget,
            "best_pre_squeeze_nonzero_coords": best, "reached_zero": hit is not None,
            "note": "capacity cells 67..255 fixed at IV; relaxed control of 64 rate "
                    "cells is a 64-dim affine slice that generically misses the finite "
                    "preimage set F^{-T}(0)."}


# ---------------------------------------------------------------------------
# Part X: weaker structured-leakage distinguisher
# ---------------------------------------------------------------------------
def eigenmode_projection_distinguisher(seed=80123, n=2000) -> dict:
    """Do messages with high checkerboard-mode projection of their pre-squeeze
    state yield distinguishable digests? (distinguisher, not a collision claim)"""
    g = H.rng(seed)
    ii, jj = np.indices((N, N))
    mode = ((-1) ** (ii + jj)).reshape(-1).astype(np.float64)
    mode /= np.linalg.norm(mode)
    projs, digbits = [], []
    for _ in range(n):
        msg = g.integers(0, 256, size=int(g.integers(1, 400)), dtype=np.uint8).tobytes()
        pre = block_trace(msg)["pre_squeeze"].astype(np.float64)
        # center around p/2 so sign structure shows
        v = pre - P / 2
        projs.append(float(abs(v @ mode)))
        digbits.append(H.bits_of(_V.squeeze((pre % P).astype(np.int64).reshape(N, N))))
    projs = np.array(projs)
    digbits = np.array(digbits)
    # correlate projection magnitude with each output bit; report max |corr|
    cors = []
    for b in range(256):
        col = digbits[:, b].astype(np.float64)
        if col.std() > 0:
            cors.append(abs(np.corrcoef(projs, col)[0, 1]))
    return {"n": n, "max_abs_corr_proj_vs_bit": float(max(cors)) if cors else None,
            "mean_abs_corr": float(np.mean(cors)) if cors else None,
            "note": "checkerboard-mode projection of pre-squeeze state vs output "
                    "bits; high correlation would be a (weak) structural leak."}


def main(seed: int = 80120) -> dict:
    t0 = time.perf_counter()
    print("  Part IV: first-block impossibility ...")
    fb = first_block_impossibility()
    print(f"    eigenmodes checked={fb['num_eigenmode_patterns_checked']} "
          f"min uncontrolled mismatch (all variants)={fb['min_uncontrolled_mismatch_over_all_variants']}")

    print("  Part VI: differential controllability of 189 capacity coords ...")
    dc = differential_controllability(seed)
    for nb, d in dc["by_blocks"].items():
        print(f"    {nb} block(s): {d['input_vars']} vars -> capacity rank "
              f"{d['capacity_rank']}/189 (deficiency {d['deficiency']})")

    print("  Part VI.2: linearized (modular-Newton) capacity steering (Model A) ...")
    ls = linearized_steering(seed, n_prior=3, iters=6)
    print(f"    residual trajectory (capacity mismatch /189): {ls['residual_trajectory']} "
          f"reached={ls['reached_capacity_target']} random_best={ls['random_baseline_best_residual']}")

    print("  Part V/IX: byte-constrained search ...")
    sb = search_targets(seed + 1, budget=6000)
    print(f"    best capacity matches={sb['best_capacity_coord_matches_to_checker(of 189)']}/189 "
          f"min pre-squeeze nonzero={sb['best_pre_squeeze_nonzero_coords(min, of 256)']}/256 "
          f"min digest HD={sb['best_digest_hamming_to_zero(min, of 256)']} "
          f"exact_lift={sb['any_exact_lift']}")

    print("  Part III/A: relaxed one-block -> zero ...")
    ra = relaxed_one_block_zero(seed + 2, budget=6000)
    print(f"    reached zero={ra['reached_zero']} best nonzero coords={ra['best_pre_squeeze_nonzero_coords']}")

    print("  Part X: eigenmode-projection distinguisher ...")
    dx = eigenmode_projection_distinguisher(seed + 3, n=1500)
    print(f"    max|corr proj vs bit|={dx['max_abs_corr_proj_vs_bit']:.3f}")

    results = {
        "phase": "8K_multiblock_reachability",
        "metadata": H.env_metadata(),
        "branch": "research/pde-eigenmode-collision-audit",
        "constants": {"p": P, "N": N, "T": T, "rate": RATE,
                      "cap0": CAP0, "cap1": CAP1, "cap2": CAP2},
        "part4_first_block_impossibility": fb,
        "part6_differential_controllability": dc,
        "part6_2_linearized_steering": ls,
        "part5_9_byte_search": sb,
        "part3_relaxed_one_block": ra,
        "part10_distinguisher": dx,
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    path = H.save_artifact("phase8k_multiblock_reachability.json", results)
    print("  saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
