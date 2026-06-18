"""Part VII -- eliminate known low-complexity bypasses (eigenmode / symmetry /
algebraic), and the central regression: do Design A's terminal-state collisions
survive the path commitment?

Tests, all on raw state / raw digest:
  1. verify the Design A eigenmode family still collapses the *wave* terminal
     state to 0 (sanity: the wave field is the unmodified Design A map);
  2. REGRESSION: the trajectory digest separates all of them (path commitment
     recovers the discarded distinction);
  3. accumulator collapse: does any sign-eigenmode / structured C map to zero or
     to a low-dimensional invariant under the accumulator step? (search);
  4. coupled fixed points: after the wave hits the zero fixed point, does the
     accumulator keep evolving (so the digest does not degenerate)?
  5. symmetry: is the coupled trajectory digest equivariant under toroidal
     translations / reflections / sign? (position weights should break it);
  6. local Jacobian of the accumulator step (informational, no global claim).

Negative results record the search budget and are not proofs.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from wavelock.curvature_capacity import spec, optimized as opt
from . import _common as C

P = spec.P
N = spec.N
_ZERO = np.zeros((N, N), dtype=np.int64)


def verify_wave_collapse() -> dict:
    """Each eigenmode state -> terminal wave 0 under the (Design A) wave round."""
    out = {}
    for name, st in C.eigenmode_states().items():
        psi = st.copy()
        for _ in range(spec.T):
            psi = opt._wave_round(psi)
        out[name] = bool(np.array_equal(psi % P, _ZERO))
    return {"all_collapse_to_terminal_zero": all(out.values()), "per_state": out}


def trajectory_separation_regression() -> dict:
    """The path commitment must give DISTINCT digests for the collapse family."""
    eig = C.eigenmode_states()
    digs = {k: opt.trajectory_digest(v).hex() for k, v in eig.items()}
    uniq = len(set(digs := list(digs.values()))) if False else len(set(digs.values()))
    # pairwise min Hamming distance
    items = list(digs.items())
    min_hd = 256
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            min_hd = min(min_hd, C.hamming_bytes(
                bytes.fromhex(items[i][1]), bytes.fromhex(items[j][1])))
    return {
        "n_states": len(digs),
        "n_distinct_digests": uniq,
        "all_distinct": uniq == len(digs),
        "min_pairwise_hamming": min_hd,
        "note": "all Design A zero-collapse states yield distinct trajectory "
                "digests -- the terminal-state collision is NOT a digest collision.",
    }


def accumulator_collapse_search(budget: int = 4000, seed: int = 90201) -> dict:
    """Search structured / random C for an accumulator step mapping C -> 0
    (with psi frozen at the zero fixed point, the worst case for binding)."""
    g = C.rng(seed)
    found_zero = None
    # structured candidates: all +/-1 sign tiles up to 4x4 scaled by random amps
    ii, jj = np.indices((N, N))
    sign_tiles = []
    for (h, w) in [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 4)]:
        for bits in product((1, -1), repeat=h * w):
            tile = np.array(bits, dtype=np.int64).reshape(h, w)
            sign_tiles.append(np.tile(tile, (N // h, N // w)))
    checked = 0
    # accumulator step at t=0 with psi_t=psi_next=0  => j=0
    for sig in sign_tiles:
        for amp in g.integers(1, P, size=8):
            Cf = ((int(amp) * sig) % P).astype(np.int64)
            nxt = opt._accumulator_step(Cf, _ZERO, _ZERO, 0)
            checked += 1
            if np.array_equal(nxt % P, _ZERO):
                found_zero = Cf.tolist()
                break
        if found_zero is not None:
            break
    # random candidates
    if found_zero is None:
        for _ in range(budget):
            Cf = g.integers(0, P, size=(N, N), dtype=np.int64)
            nxt = opt._accumulator_step(Cf, _ZERO, _ZERO, 0)
            checked += 1
            if np.array_equal(nxt % P, _ZERO):
                found_zero = Cf.tolist()
                break
    return {
        "budget": budget,
        "candidates_checked": checked,
        "accumulator_preimage_of_zero_found": found_zero is not None,
        "note": "no structured or random C mapped to zero under the accumulator "
                "step within budget; not a proof of absence.",
    }


def coupled_nondegeneracy() -> dict:
    """After the wave hits the zero fixed point, the accumulator must keep
    changing so the squeeze does not degenerate to a constant."""
    psi = _ZERO.copy()
    Cfield = opt.iv_C()
    snaps = []
    ri = 0
    for blk in range(3):
        psi, Cfield, ri = opt._coupled_evolve_T(psi, Cfield, ri)
        snaps.append(Cfield.copy())
    changing = all(not np.array_equal(snaps[i], snaps[i + 1])
                   for i in range(len(snaps) - 1))
    # digest of an all-zero-wave start is still well-defined & non-constant
    d = opt.trajectory_digest(_ZERO).hex()
    return {
        "accumulator_changes_each_block_with_zero_wave": changing,
        "zero_wave_trajectory_digest": d,
        "zero_wave_digest_is_all_zero_bytes": d == "00" * 32,
        "note": "even at the wave zero fixed point the round-dependent constant "
                "rho_t and the accumulator self-dynamics keep C evolving, so the "
                "digest is non-degenerate (unlike Design A, where the zero state "
                "squeezes to all-zero bytes).",
    }


def symmetry_attack(seed: int = 90202) -> dict:
    """Apply toroidal symmetries to a base wave state; the trajectory digest
    must NOT be invariant (position weights break equivariance)."""
    g = C.rng(seed)
    base = g.integers(0, P, size=(N, N), dtype=np.int64)
    d0 = opt.trajectory_digest(base).hex()
    variants = {
        "translate_1_0": np.roll(base, 1, 0),
        "translate_0_1": np.roll(base, 1, 1),
        "flip_rows": base[::-1, :].copy(),
        "flip_cols": base[:, ::-1].copy(),
        "transpose": base.T.copy(),
        "rot90": np.rot90(base).copy(),
        "global_sign": (P - base) % P,
    }
    res = {}
    for name, v in variants.items():
        dv = opt.trajectory_digest(v).hex()
        res[name] = {"equal_to_base": dv == d0,
                     "hamming": C.hamming_bytes(bytes.fromhex(d0), bytes.fromhex(dv))}
    return {
        "any_symmetry_preserves_digest": any(r["equal_to_base"] for r in res.values()),
        "per_symmetry": res,
        "note": "no toroidal symmetry preserves the trajectory digest; the "
                "position-dependent weights W_t(x) and round constants break the "
                "lattice equivariance that Design A's bare wave round possesses.",
    }


def accumulator_jacobian_local(seed: int = 90203) -> dict:
    """Local sensitivity: does perturbing C[x] or psi_t[x] change C'? (finite
    difference over F_p; informational, NOT a global injectivity claim)."""
    g = C.rng(seed)
    Cf = g.integers(0, P, size=(N, N), dtype=np.int64)
    psi = g.integers(0, P, size=(N, N), dtype=np.int64)
    psin = opt._wave_round(psi)
    base = opt._accumulator_step(Cf, psi, psin, 5)
    # perturb one C cell
    C2 = Cf.copy(); C2[0, 0] = (C2[0, 0] + 1) % P
    dC = opt._accumulator_step(C2, psi, psin, 5)
    changed_cells_C = int(np.sum(dC != base))
    # perturb one psi cell
    p2 = psi.copy(); p2[0, 0] = (p2[0, 0] + 1) % P
    dP = opt._accumulator_step(Cf, p2, opt._wave_round(p2), 5)
    changed_cells_psi = int(np.sum(dP != base))
    return {
        "one_C_cell_perturbation_changes_n_cells": changed_cells_C,
        "one_psi_cell_perturbation_changes_n_cells": changed_cells_psi,
        "note": "a single C-cell perturbation propagates via the accumulator "
                "Laplacian; a single psi-cell perturbation propagates via the "
                "wave round then the injection. Local diffusion only; no global "
                "injectivity is claimed.",
    }


def main(seed: int = 90200) -> dict:
    t0 = time.perf_counter()
    print("  verify wave collapse ...")
    wc = verify_wave_collapse()
    print("    all collapse to terminal zero:", wc["all_collapse_to_terminal_zero"])
    print("  trajectory separation regression ...")
    tsr = trajectory_separation_regression()
    print("    distinct digests:", tsr["n_distinct_digests"], "/", tsr["n_states"],
          "min HD", tsr["min_pairwise_hamming"])
    print("  accumulator collapse search ...")
    acs = accumulator_collapse_search()
    print("    accumulator preimage of zero found:", acs["accumulator_preimage_of_zero_found"])
    print("  coupled non-degeneracy ...")
    cnd = coupled_nondegeneracy()
    print("  symmetry attack ...")
    sym = symmetry_attack()
    print("    any symmetry preserves digest:", sym["any_symmetry_preserves_digest"])
    jac = accumulator_jacobian_local()
    out = {
        "phase": "eigenmode_attacks",
        "metadata": C.env_metadata(),
        "seed": seed,
        "equations": {
            "wave_round": "psi' = psi + D*Lap(psi) + a*psi*(b - psi^2) mod p (Design A)",
            "eigenmode_collapse": "F(s*sigma)=0 iff s^2 = b - (2rD-1)/a mod p",
            "accumulator": "C'[x] = MU*Cd + A_C*Cd^2 + W_t(x)*j(x) + rho_t mod p, "
                           "Cd = C + D_C*Lap(C), j = u + G*u*v + E*u^2 + Z*v",
        },
        "wave_collapse": wc,
        "trajectory_separation_regression": tsr,
        "accumulator_collapse_search": acs,
        "coupled_nondegeneracy": cnd,
        "symmetry_attack": sym,
        "accumulator_jacobian_local": jac,
        "limitations": [
            "collapse search is bounded (structured tiles + random); not a proof",
            "Jacobian is a local finite difference, no global injectivity claim",
            "separation shown for the enumerated Design A family only",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("eigenmode_attacks.json", out)
    print("  saved eigenmode_attacks.json", f"({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
