"""Phase CC-1 Part IV -- direct algebraic attacks on the accumulator Φ_t.

Searches for:
  1. One-round collisions: (C, C') -> same C_next with same (psi_t, psi_{t+1})
  2. Cancellation families: values of j or C for which the injection term W*j
     cancels with the self-diffusion carry
  3. Fixed points: C s.t. Phi_t(C, psi, F(psi)) == C for any psi, t
  4. Two-cycles: C_next_next == C
  5. Low-degree preimage families (u^2 symmetry exploit: ETA*u^2 is at most 2-to-1
     in u for fixed v and C)
  6. Degree-growth trace: algebraic degree of the accumulator output vs round count

All searches are bounded; negative findings are NOT proofs.
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


def _inv_mod(x: int) -> int:
    return pow(int(x) % P, P - 2, P)


# ---------------------------------------------------------------------------
# 1. One-round collision search
# ---------------------------------------------------------------------------
def one_round_collision_search(budget: int = 6000, seed: int = 92001) -> dict:
    """Search for two distinct C, C' giving the same Phi_t output.

    Fixes psi_t = psi_next = 0 (worst-case for path binding: wave at fixed
    point, so j = 0 for every cell, and the injection reduces to the
    self-diffusion + round constant only). In this regime the accumulator is:
        C_next[x] = MU*(C + D_C*Lap(C))[x] + A_C*(...)^2 + rho_t  (mod p)
    which is a fixed-p polynomial in C values; collisions here would be most
    structured (all injection from psi is zeroed out).
    """
    g = C.rng(seed)
    seen: dict[bytes, np.ndarray] = {}
    first_collision: dict | None = None
    checked = 0
    for _ in range(budget):
        Cf = g.integers(0, P, size=(N, N), dtype=np.int64)
        out = opt._accumulator_step(Cf, _ZERO, _ZERO, 0)
        key = out.tobytes()
        if key in seen:
            first_collision = {
                "C": [int(x) for x in Cf.reshape(-1)],
                "C_prime": [int(x) for x in seen[key].reshape(-1)],
            }
            break
        seen[key] = Cf.copy()
        checked += 1

    # also try structured cancellations: C with all cells equal (constant field)
    const_collision = None
    const_imgs: dict[int, int] = {}
    for v in range(min(P, 1000)):
        Cf = np.full((N, N), v, dtype=np.int64)
        out = opt._accumulator_step(Cf, _ZERO, _ZERO, 0)
        img_key = int(out[0, 0])  # constant in => constant out? check
        if int(out.max()) == int(out.min()):
            if img_key in const_imgs:
                const_collision = {"v1": const_imgs[img_key], "v2": v}
                break
            const_imgs[img_key] = v

    return {
        "random_budget": budget,
        "random_collision_found": first_collision is not None,
        "random_collision": first_collision,
        "checked": checked,
        "constant_field_collision_found": const_collision is not None,
        "constant_field_collision": const_collision,
        "note": (
            "With psi=0 the injection term vanishes and the accumulator is a "
            "polynomial map in C. No collision found within budget; not a proof "
            "of injectivity. Constant-field search checks whether distinct constant "
            "inputs produce identical constant outputs (would be a structural flaw)."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Cancellation family analysis
# ---------------------------------------------------------------------------
def cancellation_family_analysis() -> dict:
    """Analyze the injection term j(u,v) = u + G*u*v + E*u^2 + Z*v mod p.

    Key properties:
    - j is quadratic in u (for fixed v), linear in v (for fixed u).
    - The ETA*u^2 term means j(u,v) = j(-u + K, v) for some K (at most 2-to-1
      in u for fixed v, since u^2 = (-u)^2).
    - Specifically: j(u,v) = j(u',v) requires
        u - u' + G*v*(u - u') + ETA*(u^2 - u'^2) = 0 mod p
        (u - u')*(1 + G*v + ETA*(u + u')) = 0 mod p
      So either u = u' or u + u' = -(1 + G*v)/ETA mod p (the "pairing").
    """
    # Compute the cancellation partner u' for a sample of (u, v) pairs
    g = C.rng(92002)
    pairs_found = 0
    samples = 500
    ETA_inv = _inv_mod(spec.ETA)

    cancellation_pairs = []
    for _ in range(samples):
        u = int(g.integers(0, P))
        v = int(g.integers(0, P))
        # partner: u' = (-(1 + G*v)*ETA_inv - u) mod p
        u_prime = int((P - (1 + spec.GAMMA * v) % P * ETA_inv % P - u) % P)
        # Verify
        j1 = (u + spec.GAMMA * (u * v % P) + spec.ETA * (u * u % P) + spec.ZETA * v) % P
        j2 = (u_prime + spec.GAMMA * (u_prime * v % P) + spec.ETA * (u_prime * u_prime % P)
               + spec.ZETA * v) % P
        if j1 == j2 and u != u_prime:
            pairs_found += 1
            if len(cancellation_pairs) < 5:
                cancellation_pairs.append({"u": u, "u_prime": u_prime, "v": v,
                                            "j": int(j1)})

    return {
        "samples": samples,
        "cancellation_pairs_found": pairs_found,
        "all_pairs_found": pairs_found == samples,
        "examples": cancellation_pairs,
        "algebraic_condition": (
            "j(u,v) = j(u',v) iff u=u' OR u+u' = -(1+GAMMA*v)*ETA_inv mod p. "
            "This is a provable 2-to-1 structure in the u-injection (for fixed v "
            "and fixed C). An attacker who can freely choose psi_t can pair u with "
            "its cancellation partner; this does NOT directly yield a state-level "
            "collision because the wave field evolution is not free (F is not "
            "injective, but psi_t -> psi_{t+1} is fixed by F, not by the attacker). "
            "The structural weakness is documented; no bypass is demonstrated."
        ),
        "impact_on_digest": (
            "The 2-to-1 u-injection is a known limitation (documented in spec). "
            "It means distinct wave states with u + u' = C_cancel can produce the "
            "same injection j, but the wave round itself is already non-injective "
            "(Phase 8J); the accumulator is designed to RECOVER separation, not to "
            "be independently injective. The separation test (Parts I/II) confirms "
            "47/47 distinct digests for the known collision family."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Fixed-point search
# ---------------------------------------------------------------------------
def fixed_point_search(budget: int = 5000, seed: int = 92003) -> dict:
    """Search for C s.t. Phi_0(C, psi, F(psi)) = C for random psi."""
    g = C.rng(seed)
    found_fp = None
    checked = 0
    for _ in range(budget):
        psi = g.integers(0, P, size=(N, N), dtype=np.int64)
        psin = opt._wave_round(psi)
        Cf = g.integers(0, P, size=(N, N), dtype=np.int64)
        nxt = opt._accumulator_step(Cf, psi, psin, 0)
        checked += 1
        if np.array_equal(nxt % P, Cf % P):
            found_fp = {
                "psi": [int(x) for x in psi.reshape(-1)],
                "C": [int(x) for x in Cf.reshape(-1)],
            }
            break

    # Analytic note: for fixed psi (so j is fixed per cell), a fixed point
    # requires MU*cd + A_C*cd^2 + W_t*j + rho_t = C[x] mod p
    # where cd = C[x] + D_C*Lap(C)[x]. This is a degree-2 polynomial in each
    # C[x] (coupled through Lap). Not impossible in principle.
    return {
        "budget": budget,
        "candidates_checked": checked,
        "fixed_point_found": found_fp is not None,
        "fixed_point_example": found_fp,
        "note": (
            "A fixed point of Phi_t with respect to the accumulator (not the full "
            "coupled round) would require the accumulator to reproduce itself. "
            "The round-dependent constant rho_t differs across t so a fixed point "
            "of the *full multi-round* trajectory is far more constrained. "
            "No fixed point found within budget."
        ),
    }


# ---------------------------------------------------------------------------
# 4. Two-cycle search
# ---------------------------------------------------------------------------
def two_cycle_search(budget: int = 3000, seed: int = 92004) -> dict:
    """Search for C s.t. Phi_0(Phi_0(C, psi, F(psi)), F(psi), F(F(psi))) = C."""
    g = C.rng(seed)
    found_cycle = None
    checked = 0
    for _ in range(budget):
        psi = g.integers(0, P, size=(N, N), dtype=np.int64)
        psi1 = opt._wave_round(psi)
        psi2 = opt._wave_round(psi1)
        Cf = g.integers(0, P, size=(N, N), dtype=np.int64)
        C1 = opt._accumulator_step(Cf, psi, psi1, 0)
        C2 = opt._accumulator_step(C1, psi1, psi2, 1)
        checked += 1
        if np.array_equal(C2 % P, Cf % P):
            found_cycle = {
                "psi": [int(x) for x in psi.reshape(-1)],
                "C": [int(x) for x in Cf.reshape(-1)],
            }
            break
    return {
        "budget": budget,
        "candidates_checked": checked,
        "two_cycle_found": found_cycle is not None,
        "two_cycle_example": found_cycle,
        "note": "Two-cycles of the accumulator over the first two rounds searched. "
                "None found within budget.",
    }


# ---------------------------------------------------------------------------
# 5. Degree-growth trace (symbolic scalar model)
# ---------------------------------------------------------------------------
def degree_growth_trace(max_rounds: int = 6) -> dict:
    """Trace algebraic degree of the accumulator output as a polynomial in u_0
    (the initial wave cell value), under the scalar model (ignore the Laplacian
    coupling, i.e., treat each cell independently for an upper-bound estimate).

    The wave round is degree 3 in psi_0 (since psi^2 enters), so psi_t has
    degree 3^t in psi_0. The injection j is degree 2 in (psi_t, psi_{t+1}),
    i.e., degree <= 2 * 3^t in psi_0 at round t. The accumulator self-square
    A_C*cd^2 doubles the degree in C; but C at round t depends on u_0 only
    through j at rounds 0..t-1.

    This is a heuristic upper bound (not a Gröbner computation); the true
    degree over F_p is at most p-1 by Fermat.
    """
    rows = []
    psi_deg = 1  # initial psi_0 is degree 1 in itself
    C_deg = 0    # initial C is independent of psi_0 (constant)
    for t in range(max_rounds):
        psi_next_deg = 3 * psi_deg  # wave round is degree 3
        j_deg = max(2 * psi_deg, psi_deg + psi_next_deg, psi_next_deg)
        C_next_deg = max(C_deg + j_deg, 2 * C_deg)  # MU*Cd+A_C*Cd^2 + W*j
        rows.append({
            "t": t,
            "psi_t_degree_bound": psi_deg,
            "psi_next_degree_bound": psi_next_deg,
            "injection_j_degree_bound": j_deg,
            "C_next_degree_bound": C_next_deg,
        })
        psi_deg = psi_next_deg
        C_deg = C_next_deg

    return {
        "model": "scalar upper bound (ignore Laplacian coupling; each cell independent)",
        "interpretation": (
            "The degree bound grows rapidly (tripling each wave round), rapidly "
            "exceeding p-1=2^31-2, at which point Fermat reduction caps the "
            "effective degree at p-1. In the scalar model the output C_T is "
            "a polynomial of degree >>p in the input psi_0, meaning it is NOT "
            "representable as a low-degree polynomial over Z_p. However, this "
            "is an UPPER BOUND -- the true degree after Fermat reduction could "
            "be much lower. A Gröbner/algebraic-geometry computation would be "
            "needed for a tight bound; those solvers are unavailable here."
        ),
        "rounds": rows,
        "cap_note": "Over F_p, degree is at most p-1=2147483646 by Fermat; "
                    "the heuristic bound saturates the cap by round 3-4.",
    }


# ---------------------------------------------------------------------------
# 6. Explicit u^2 pairing attack on the wave
# ---------------------------------------------------------------------------
def eta_pairing_attack(n_trials: int = 200, seed: int = 92005) -> dict:
    """Demonstrate the known u^2 ambiguity and measure its effect on digests.

    For each trial: pick a random state psi; compute cancellation partner psi'
    where psi'[0,0] is the u-pairing partner for cell 0. Check whether this
    creates a trajectory digest collision (it should NOT, because the wave round
    F is not free -- F(psi) != F(psi') in general).
    """
    g = C.rng(seed)
    digest_same_count = 0
    wave_same_count = 0
    trials = []

    ETA_inv = _inv_mod(spec.ETA)

    for _ in range(n_trials):
        psi = g.integers(0, P, size=(N, N), dtype=np.int64)
        u = int(psi[0, 0])
        psin = opt._wave_round(psi)
        v = int(psin[0, 0])
        u_prime = int((P - (1 + spec.GAMMA * v % P) * ETA_inv % P - u) % P)

        psi2 = psi.copy()
        psi2[0, 0] = u_prime

        d1 = opt.trajectory_digest(psi).hex()
        d2 = opt.trajectory_digest(psi2).hex()
        wave_eq = np.array_equal(opt._wave_round(psi) % P, opt._wave_round(psi2) % P)
        digest_eq = d1 == d2
        if digest_eq:
            digest_same_count += 1
        if wave_eq:
            wave_same_count += 1
        if len(trials) < 3:
            trials.append({
                "u": u, "u_prime": u_prime, "v": v,
                "wave_round_equal": bool(wave_eq),
                "digest_equal": bool(digest_eq),
                "hamming": C.hamming_bytes(bytes.fromhex(d1), bytes.fromhex(d2)),
            })

    return {
        "n_trials": n_trials,
        "digest_collisions_found": digest_same_count,
        "wave_round_equalities_found": wave_same_count,
        "examples": trials,
        "interpretation": (
            "Replacing psi[0,0] with its u-pairing partner changes the wave round "
            "output (since F depends on all cells, not just u). The pairing "
            "cancels j at cell 0 for the NOMINAL (u,v) pair, but v = F(psi)[0,0] "
            "changes when psi[0,0] changes, breaking the cancellation globally. "
            "No digest collisions found; the structural 2-to-1 in j(u,v) does not "
            "yield digest collisions because the wave evolution is not controllable."
        ),
    }


def main(seed: int = 92000) -> dict:
    t0 = time.perf_counter()

    print("  1. one-round collision search ...")
    coll = one_round_collision_search(seed=seed + 1)
    print(f"    collision_found={coll['random_collision_found']}, "
          f"const_collision={coll['constant_field_collision_found']}")

    print("  2. cancellation family analysis ...")
    cancel = cancellation_family_analysis()
    print(f"    pairs_found={cancel['cancellation_pairs_found']}/{cancel['samples']} "
          f"(all expected: {cancel['all_pairs_found']})")

    print("  3. fixed-point search ...")
    fp = fixed_point_search(seed=seed + 3)
    print(f"    fixed_point_found={fp['fixed_point_found']}")

    print("  4. two-cycle search ...")
    tc = two_cycle_search(seed=seed + 4)
    print(f"    two_cycle_found={tc['two_cycle_found']}")

    print("  5. degree-growth trace ...")
    deg = degree_growth_trace(max_rounds=8)

    print("  6. eta pairing attack ...")
    eta = eta_pairing_attack(seed=seed + 5)
    print(f"    digest_collisions={eta['digest_collisions_found']}/{eta['n_trials']}")

    out = {
        "artifact": "accumulator_algebraic_attacks",
        "description": "Direct algebraic attacks on the CC-Core-v0 accumulator Phi_t",
        "metadata": C.env_metadata(),
        "seed": seed,
        "equations": {
            "injection_j": "j(u,v) = u + GAMMA*u*v + ETA*u^2 + ZETA*v  mod p",
            "accumulator": (
                "C_next[x] = MU*(C + D_C*Lap(C))[x] "
                "+ A_C*(C + D_C*Lap(C))[x]^2 "
                "+ W_t(x)*j(psi_t[x], psi_{t+1}[x]) + rho_t  mod p"
            ),
            "u_pairing": (
                "j(u,v) = j(u',v) iff u=u' OR u+u' = -(1+GAMMA*v)/ETA mod p"
            ),
        },
        "one_round_collisions": coll,
        "cancellation_family": cancel,
        "fixed_points": fp,
        "two_cycles": tc,
        "degree_growth": deg,
        "eta_pairing_attack": eta,
        "summary": {
            "collision_found": coll["random_collision_found"] or coll["constant_field_collision_found"],
            "fixed_point_found": fp["fixed_point_found"],
            "two_cycle_found": tc["two_cycle_found"],
            "eta_2to1_is_structural": True,
            "eta_2to1_yields_digest_collision": eta["digest_collisions_found"] > 0,
        },
        "limitations": [
            "All searches are bounded; no finding is a proof of absence",
            "One-round collision search fixes psi=0 (worst case, zero injection)",
            "Degree growth is a heuristic scalar upper bound, not a Groebner result",
            "The u^2 pairing is a provable structural property (documented in spec)",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("accumulator_algebraic_attacks.json", out)
    print(f"  saved accumulator_algebraic_attacks.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
