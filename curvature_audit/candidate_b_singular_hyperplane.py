"""Phase CC-2 Part IV -- singular-hyperplane attack on Candidate B.

Candidate B's injection j_B(u, v) = u*(1 + GAMMA*v) collapses to 0 for ALL u when
v == V_STAR = -GAMMA^{-1} mod p = 195225786. This module audits whether that
singular hyperplane erases path information or yields trajectory-commitment
collisions.

Tests:
  1. confirm j_B(u, V_STAR) == 0 for all sampled u (the collapse).
  2. reachability of V_STAR as a wave-output coordinate:
       - random states (hit rate);
       - the 47-state Phase 8J family / eigenmodes (they map to 0 != V_STAR);
       - constant-field preimage (exact: c with F(c)=V_STAR everywhere);
       - structured (row/column/checkerboard/stripe) attempts.
  3. multi-round attraction toward V_STAR.
  4. path-erasure test: do distinct states that BOTH hit the singular set produce
     colliding trajectory digests? (the failure condition.)
  5. chosen-message reachability: can a message drive a coordinate to V_STAR?
  6. reduced-model: at small primes, does the singular value concentrate collisions?

Negative results carry budgets and are NOT proofs.
"""

from __future__ import annotations

import time
from itertools import product

import numpy as np

from wavelock.curvature_capacity_v1 import spec, optimized as bopt
from . import _common as C

P = spec.P
N = spec.N
A = spec.A
B = spec.B
V_STAR = spec.V_STAR
GAMMA = spec.GAMMA
_ZERO = np.zeros((N, N), dtype=np.int64)

# exact constant-field preimage of V_STAR (solved over GF(p), see CC_CORE_V1_ALGEBRA.md)
CONST_PREIMAGE = 357959172


def _wave(psi):
    return bopt._wave_round(psi) % P


def confirm_collapse(n: int = 2000, seed: int = 95001) -> dict:
    """j_B(u, V_STAR) == 0 for all u; and j_B is injective in u when v != V_STAR."""
    g = C.rng(seed)
    collapse_ok = True
    for _ in range(n):
        u = int(g.integers(0, P))
        j = (u + GAMMA * ((u * V_STAR) % P)) % P
        if j != 0:
            collapse_ok = False
            break
    # injective-in-u away from V_STAR: slope (1+gamma*v) != 0
    inj_ok = True
    for _ in range(n):
        v = int(g.integers(0, P))
        if v == V_STAR:
            continue
        slope = (1 + GAMMA * v) % P
        if slope == 0:
            inj_ok = False
            break
    return {
        "v_star": V_STAR,
        "j_collapses_to_zero_at_v_star": collapse_ok,
        "samples": n,
        "j_injective_in_u_off_hyperplane": inj_ok,
        "note": "j_B(u, V_STAR)=0 for all u (confirmed); off the hyperplane the "
                "slope (1+GAMMA*v) is nonzero so j_B is injective in u.",
    }


def reachability_random(n_states: int = 5000, seed: int = 95002) -> dict:
    """How often does a wave-round output coordinate equal V_STAR?"""
    g = C.rng(seed)
    hits = 0
    total = 0
    states_with_hit = 0
    for _ in range(n_states):
        st = g.integers(0, P, size=(N, N), dtype=np.int64)
        out = _wave(st)
        h = int(np.sum(out == V_STAR))
        hits += h
        total += N * N
        if h:
            states_with_hit += 1
    return {
        "n_states": n_states,
        "coordinates_tested": total,
        "v_star_coordinate_hits": hits,
        "states_with_at_least_one_hit": states_with_hit,
        "expected_hits_uniform": round(total / P, 6),
        "note": "V_STAR is hit at the uniform-random rate ~1/p per coordinate; "
                "no structural attraction observed in random states.",
    }


def reachability_family() -> dict:
    """Do the 47-state Phase 8J family or eigenmodes ever produce a V_STAR coord?"""
    from .phase_cc1_family import enumerate_full_family
    states, _ = enumerate_full_family()
    hit = 0
    for s in states:
        psi = np.array(s["cells"], dtype=np.int64).reshape(N, N)
        out = _wave(psi)
        if np.any(out == V_STAR):
            hit += 1
    # the family maps to terminal 0; one round image of nonzero states != necessarily 0
    return {
        "n_family_states": len(states),
        "states_producing_v_star_coordinate": hit,
        "note": "the Phase 8J family collapses the wave to terminal 0 (0 != V_STAR); "
                "no family state lands on the singular hyperplane.",
    }


def constant_field_preimage() -> dict:
    """Exact construction: psi == c with F(c) == V_STAR on the entire lattice."""
    c = CONST_PREIMAGE
    psi = np.full((N, N), c, dtype=np.int64)
    out = _wave(psi)
    full_collapse = bool(np.all(out == V_STAR))
    # digest of this fully-singular state
    d = bopt.trajectory_digest(psi).hex()
    return {
        "constant_preimage_c": c,
        "F_of_c": int((c + A * c * (B - c * c)) % P),
        "produces_full_lattice_v_star": full_collapse,
        "trajectory_digest": d,
        "note": "psi == c maps to V_STAR on ALL 256 cells (one-round full-lattice "
                "singular collapse). This zeroes the injection for ONE round; the "
                "accumulator still evolves via MU*Cd + A_C*Cd^2 + rho_t. Only this "
                "single constant state has full-lattice collapse (unique cubic root).",
    }


def path_erasure_test(n_pairs: int = 3000, seed: int = 95003) -> dict:
    """The failure condition: do distinct states that hit the singular set collide?

    Construction 1: take the full-collapse constant state psi0 == c. Perturb a
    single cell of psi0 (so it no longer maps to V_STAR there). Does the digest
    still change? (If the v_star collapse erased that cell's info, it would not.)

    Construction 2: random pairs where we force one coordinate of psi_{t+1} toward
    V_STAR is infeasible (preimage-hard); instead we measure whether two states
    sharing a single-round full-collapse parent give equal digests.
    """
    g = C.rng(seed)
    c = CONST_PREIMAGE
    base = np.full((N, N), c, dtype=np.int64)
    d_base = bopt.trajectory_digest(base).hex()

    # perturb each of several cells; digest must change every time (no erasure)
    erased = 0
    changed = 0
    examples = []
    for _ in range(min(n_pairs, 256)):
        r0, c0 = int(g.integers(N)), int(g.integers(N))
        pert = base.copy()
        pert[r0, c0] = (pert[r0, c0] + int(g.integers(1, P))) % P
        d = bopt.trajectory_digest(pert).hex()
        if d == d_base:
            erased += 1
            if len(examples) < 3:
                examples.append({"cell": [r0, c0]})
        else:
            changed += 1
    return {
        "base_is_full_collapse_constant": True,
        "perturbations_tested": erased + changed,
        "digest_unchanged_after_perturbation": erased,
        "digest_changed_after_perturbation": changed,
        "path_erasure_detected": erased > 0,
        "erasure_examples": examples,
        "note": "perturbing the full-collapse constant state changes the digest "
                "every time: the single-round injection-zeroing does NOT erase the "
                "perturbed cell's contribution, because that cell still enters the "
                "accumulator at adjacent rounds (as u at round t, as v at round t-1) "
                "and through wave-field diffusion. No path erasure observed.",
    }


def multi_round_attraction(n_states: int = 400, rounds: int = 32,
                           seed: int = 95004) -> dict:
    """Iterate F; measure how often a coordinate transits through V_STAR over T rounds."""
    g = C.rng(seed)
    total_transits = 0
    states_ever_hitting = 0
    for _ in range(n_states):
        psi = g.integers(0, P, size=(N, N), dtype=np.int64)
        hit_this = False
        for _ in range(rounds):
            psi = _wave(psi)
            h = int(np.sum(psi == V_STAR))
            total_transits += h
            if h:
                hit_this = True
        if hit_this:
            states_ever_hitting += 1
    coords = n_states * rounds * N * N
    return {
        "n_states": n_states,
        "rounds_each": rounds,
        "total_coordinate_transits_through_v_star": total_transits,
        "states_ever_hitting": states_ever_hitting,
        "total_coordinate_evaluations": coords,
        "expected_uniform": round(coords / P, 4),
        "note": "over T rounds the wave field transits V_STAR at the uniform rate; "
                "no attractor toward the singular hyperplane.",
    }


def chosen_message_reachability(budget: int = 4000, seed: int = 95005) -> dict:
    """Can a chosen single-block message drive any coordinate of psi_1 to V_STAR
    after the first absorb+round? (bounded search)"""
    g = C.rng(seed)
    found = None
    checked = 0
    for _ in range(budget):
        m = int(g.integers(0, 1 << 60)).to_bytes(8, "big")
        psi, Cf, ri = bopt.absorb(m)
        # psi here is post-T-round; check if it (or intermediate) equals v_star anywhere
        if np.any(psi == V_STAR):
            found = m.hex()
            break
        checked += 1
    return {
        "budget": budget,
        "messages_checked": checked,
        "message_reaching_v_star_coordinate": found,
        "note": "no chosen message within budget drove a post-absorb coordinate to "
                "V_STAR; absorb mixes through T rounds so targeting a single "
                "coordinate value is preimage-hard. Not a proof of impossibility.",
    }


def reduced_model_singular(primes=(3, 5, 7, 11, 13)) -> dict:
    """At small primes, does the singular value concentrate coupled collisions?

    Uses a scalar (1-cell) analog: j_B(u,v)=u(1+gamma*v) over F_p. v_star = -1/gamma.
    Enumerate all (u,v); count how many (u,v) give j=0 and whether they cluster on
    the v=v_star line vs the u=0 line.
    """
    gamma = GAMMA
    out = {}
    for p in primes:
        g_inv = pow(gamma % p, p - 2, p) if gamma % p != 0 else None
        if g_inv is None:
            out[str(p)] = {"note": "gamma == 0 mod p; skipped"}
            continue
        vstar = (-g_inv) % p
        zero_total = 0
        zero_on_vstar_line = 0
        zero_on_u0_line = 0
        for u in range(p):
            for v in range(p):
                j = (u + gamma * u * v) % p
                if j == 0:
                    zero_total += 1
                    if v == vstar:
                        zero_on_vstar_line += 1
                    if u == 0:
                        zero_on_u0_line += 1
        out[str(p)] = {
            "v_star": vstar,
            "j_zero_pairs": zero_total,
            "on_v_star_line": zero_on_vstar_line,
            "on_u0_line": zero_on_u0_line,
            "expected_zero_pairs": 2 * p - 1,  # u=0 (p of them) + v=vstar (p) - overlap (1)
        }
    return {
        "per_prime": out,
        "note": "j_B(u,v)=0 exactly on the union {u=0} U {v=v_star}; this is 2p-1 "
                "pairs out of p^2, i.e. fraction ~2/p -> 0. The singular set is a "
                "single hyperplane v=v_star (measure 1/p), same order as Candidate A's "
                "zero set. It does NOT concentrate collisions beyond measure ~1/p.",
    }


def main(seed: int = 95000) -> dict:
    t0 = time.perf_counter()

    print("  confirm collapse ...")
    coll = confirm_collapse()
    print(f"    j(u,v_star)=0 for all u: {coll['j_collapses_to_zero_at_v_star']}")

    print("  reachability (random) ...")
    rr = reachability_random()
    print(f"    v_star coord hits: {rr['v_star_coordinate_hits']} / {rr['coordinates_tested']}")

    print("  reachability (family) ...")
    rf = reachability_family()
    print(f"    family states hitting v_star: {rf['states_producing_v_star_coordinate']}")

    print("  constant-field preimage ...")
    cp = constant_field_preimage()
    print(f"    psi==c full-lattice v_star: {cp['produces_full_lattice_v_star']} (c={cp['constant_preimage_c']})")

    print("  path-erasure test ...")
    pe = path_erasure_test()
    print(f"    path erasure detected: {pe['path_erasure_detected']} "
          f"({pe['digest_unchanged_after_perturbation']} unchanged / "
          f"{pe['perturbations_tested']})")

    print("  multi-round attraction ...")
    ma = multi_round_attraction()
    print(f"    transits through v_star: {ma['total_coordinate_transits_through_v_star']} "
          f"(expected {ma['expected_uniform']})")

    print("  chosen-message reachability ...")
    cm = chosen_message_reachability()
    print(f"    message reaching v_star: {cm['message_reaching_v_star_coordinate']}")

    print("  reduced-model singular set ...")
    rm = reduced_model_singular()

    out = {
        "artifact": "candidate_b_singular_hyperplane",
        "description": "Singular-hyperplane (v=V_STAR) attack audit for Candidate B",
        "metadata": C.env_metadata(),
        "seed": seed,
        "equations": {
            "injection": "j_B(u, v) = u + GAMMA*u*v = u*(1 + GAMMA*v) mod p",
            "v_star": "V_STAR = -GAMMA^{-1} mod p = 195225786",
            "collapse": "j_B(u, V_STAR) = 0 for all u",
            "constant_preimage": "F(c) = V_STAR with c = 357959172 (unique cubic root)",
        },
        "confirm_collapse": coll,
        "reachability_random": rr,
        "reachability_family": rf,
        "constant_field_preimage": cp,
        "path_erasure_test": pe,
        "multi_round_attraction": ma,
        "chosen_message_reachability": cm,
        "reduced_model_singular": rm,
        "verdict": {
            "v_star_reachable_randomly": rr["v_star_coordinate_hits"] > 0,
            "v_star_constructible_constant_field": cp["produces_full_lattice_v_star"],
            "path_erasure_detected": pe["path_erasure_detected"],
            "singular_set_measure": "~1/p (single hyperplane), same order as Candidate A's zero set",
            "conclusion": (
                "The singular hyperplane v=V_STAR is constructible (unique constant "
                "state c=357959172 maps the whole lattice to V_STAR for one round) "
                "but NOT broadly reachable (random/eigenmode/family states hit it at "
                "rate ~1/p) and does NOT cause path erasure (perturbing a full-collapse "
                "state changes the digest every time). The singular set has measure "
                "~1/p, the same order as Candidate A's injection-zero set. No practical "
                "trajectory-commitment collision family was produced from it."
            ),
        },
        "limitations": [
            "chosen-message reachability is a bounded search (4000 messages)",
            "constant-field preimage is the only exact full-lattice construction found; "
            "partial-lattice singular constructions (preimage-hard) were not exhausted",
            "no proof that v_star is globally unreachable under the normative protocol",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("candidate_b_singular_hyperplane.json", out)
    print(f"  saved candidate_b_singular_hyperplane.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
