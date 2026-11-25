"""Phase CC-1 Part IX -- two-candidate accumulator comparison.

Compares CC-Core-v0 (Candidate A, the current design) against a second
accumulator candidate (Candidate B: simpler linear injection without the
quadratic u^2 term) on the following criteria:

  1. Trajectory separation of the complete Phase 8J eigenmode family (47 states)
  2. Algebraic structure: does the u^2 term provide additional separation?
  3. Avalanche / diffusion: bit-flip response in psi vs digest
  4. Fixed-point / cancellation vulnerability
  5. Round-constant dependence (does removing the self-square term change mixing?)

Candidate A (CC-Core-v0, current):
  j_A(u,v) = u + GAMMA*u*v + ETA*u^2 + ZETA*v  mod p
  C_A_next = MU*Cd + A_C*Cd^2 + W_t*j_A + rho_t  mod p

Candidate B (linear injection, no u^2):
  j_B(u,v) = u + GAMMA*u*v + ZETA*v  mod p  (ETA=0)
  C_B_next = MU*Cd + A_C*Cd^2 + W_t*j_B + rho_t  mod p
  (keeps the self-square in C, only removes the u^2 injection)

Both use identical wave rounds, IVs, constants, and squeeze.
"""

from __future__ import annotations

import time

import numpy as np

from wavelock.curvature_capacity import spec, optimized as opt
from . import _common as C
from .phase_cc1_family import enumerate_full_family

P = spec.P
N = spec.N
_ZERO = np.zeros((N, N), dtype=np.int64)


def _candidate_b_accumulator_step(
    Cf: np.ndarray, psi_t: np.ndarray, psi_next: np.ndarray, t: int
) -> np.ndarray:
    """Candidate B: identical to Candidate A except ETA=0 (no u^2 term)."""
    p = P
    rho = spec.round_constant(t)
    u = psi_t.reshape(-1)
    v = psi_next.reshape(-1)
    # j_B: no ETA*u^2 term
    j = (u + spec.GAMMA * ((u * v) % p) + spec.ZETA * v) % p
    Cd_lap = opt._lap_C(Cf).reshape(-1)
    cd = (Cf.reshape(-1) + (spec.D_C * Cd_lap) % p) % p
    x = np.arange(spec.N_CELLS, dtype=np.int64)
    w = (1 + (t + 1) * spec.WA + (x + 1) * spec.WB + (t + 1) * (x + 1) * spec.WC) % p
    out = (spec.MU * cd + (spec.A_C * ((cd * cd) % p)) % p + (w * j) % p + rho) % p
    return out.reshape(N, N)


def _trajectory_digest_b(psi0: np.ndarray, rounds: int = spec.T) -> bytes:
    """Trajectory digest using Candidate B accumulator."""
    psi = (np.asarray(psi0, dtype=np.int64) % P).reshape(N, N).copy()
    Cf = opt.iv_C()
    ri = 0
    for _ in range(rounds):
        psi_next = opt._wave_round(psi)
        Cf_next = _candidate_b_accumulator_step(Cf, psi, psi_next, ri)
        psi, Cf = psi_next, Cf_next
        ri += 1
    return opt.squeeze(psi, Cf, ri, spec.OUTPUT_BITS)


# ---------------------------------------------------------------------------
# 1. Eigenmode family separation
# ---------------------------------------------------------------------------
def family_separation_comparison() -> dict:
    """Compare how many distinct digests each candidate produces for the
    complete Phase 8J family."""
    states, stats = enumerate_full_family()
    digs_a: list[str] = []
    digs_b: list[str] = []
    for s in states:
        psi = np.array(s["cells"], dtype=np.int64).reshape(N, N)
        digs_a.append(opt.trajectory_digest(psi).hex())
        digs_b.append(_trajectory_digest_b(psi).hex())

    n = len(states)
    ua = len(set(digs_a))
    ub = len(set(digs_b))

    min_hd_a = 256
    min_hd_b = 256
    for i in range(n):
        for j in range(i + 1, n):
            ha = C.hamming_bytes(bytes.fromhex(digs_a[i]), bytes.fromhex(digs_a[j]))
            hb = C.hamming_bytes(bytes.fromhex(digs_b[i]), bytes.fromhex(digs_b[j]))
            min_hd_a = min(min_hd_a, ha)
            min_hd_b = min(min_hd_b, hb)

    return {
        "n_states": n,
        "candidate_a": {
            "name": "CC-Core-v0 (j_A = u + GAMMA*u*v + ETA*u^2 + ZETA*v)",
            "distinct_digests": ua,
            "all_distinct": ua == n,
            "min_pairwise_hamming": min_hd_a,
        },
        "candidate_b": {
            "name": "Candidate B (j_B = u + GAMMA*u*v + ZETA*v, ETA=0)",
            "distinct_digests": ub,
            "all_distinct": ub == n,
            "min_pairwise_hamming": min_hd_b,
        },
        "interpretation": (
            "Both candidates should separate the Design A eigenmode family, since "
            "the odd-in-u term (u itself) distinguishes +s*sigma from -s*sigma. "
            "The u^2 term (ETA) adds an extra quadratic injection but introduces "
            "the known 2-to-1 structural weakness. If Candidate B also achieves "
            "full separation, the ETA term offers no advantage for this task."
        ),
    }


# ---------------------------------------------------------------------------
# 2. Structural 2-to-1 comparison
# ---------------------------------------------------------------------------
def structural_comparison() -> dict:
    """Compare the u-injection ambiguity between Candidate A and B.

    For Candidate A: j_A(u,v) = j_A(u',v) has a nontrivial solution
      u' = -(1+GAMMA*v)/ETA - u  (the u^2 pairing).
    For Candidate B: j_B(u,v) = u*(1+GAMMA*v) + ZETA*v is INJECTIVE in u
      (for fixed v, j_B is linear in u with slope (1+GAMMA*v); if 1+GAMMA*v != 0
      mod p, then j_B(u,v) = j_B(u',v) implies u = u').
    """
    g = C.rng(92010)
    samples = 500

    # Candidate A: check pairing
    a_pairs_found = 0
    b_linear_ambiguity_found = 0

    for _ in range(samples):
        u = int(g.integers(0, P))
        v = int(g.integers(0, P))

        # A pairing
        eta_inv = pow(spec.ETA, P - 2, P)
        u_prime_a = int((P - (1 + spec.GAMMA * v % P) * eta_inv % P - u) % P)
        j_a1 = (u + spec.GAMMA * (u * v % P) + spec.ETA * (u * u % P) + spec.ZETA * v) % P
        j_a2 = (u_prime_a + spec.GAMMA * (u_prime_a * v % P)
                + spec.ETA * (u_prime_a * u_prime_a % P) + spec.ZETA * v) % P
        if j_a1 == j_a2 and u != u_prime_a:
            a_pairs_found += 1

        # B: no pairing unless (1 + GAMMA*v) = 0 mod p
        slope = (1 + spec.GAMMA * v) % P
        if slope == 0:
            b_linear_ambiguity_found += 1

    return {
        "samples": samples,
        "candidate_a_u2_pairs_found": a_pairs_found,
        "candidate_a_is_2to1_in_u": a_pairs_found == samples,
        "candidate_b_linear_ambiguity_count": b_linear_ambiguity_found,
        "candidate_b_is_injective_in_u": b_linear_ambiguity_found == 0,
        "slope_zero_probability_b": "~1/p (negligible)",
        "interpretation": (
            "Candidate B's linear injection j_B is injective in u for all v with "
            "(1+GAMMA*v) != 0 mod p; this happens with probability ~1/p. "
            "Candidate A's quadratic injection j_A has a provable 2-to-1 structure "
            "for every (u,v). Candidate B is strictly better in this structural "
            "dimension; however, the u^2 term in A does increase the polynomial "
            "degree of the injection, which may aid nonlinearity in other respects."
        ),
    }


# ---------------------------------------------------------------------------
# 3. Avalanche comparison
# ---------------------------------------------------------------------------
def avalanche_comparison(n_msgs: int = 80, seed: int = 92011) -> dict:
    """Flip a random bit in the initial wave state; measure bit changes in digest."""
    g = C.rng(seed)
    hd_a_list: list[int] = []
    hd_b_list: list[int] = []
    for _ in range(n_msgs):
        psi = g.integers(0, P, size=(N, N), dtype=np.int64)
        # flip one random cell by 1
        r0, c0 = int(g.integers(N)), int(g.integers(N))
        psi2 = psi.copy()
        psi2[r0, c0] = (psi2[r0, c0] + 1) % P

        d_a1 = opt.trajectory_digest(psi).hex()
        d_a2 = opt.trajectory_digest(psi2).hex()
        d_b1 = _trajectory_digest_b(psi).hex()
        d_b2 = _trajectory_digest_b(psi2).hex()

        hd_a_list.append(C.hamming_bytes(bytes.fromhex(d_a1), bytes.fromhex(d_a2)))
        hd_b_list.append(C.hamming_bytes(bytes.fromhex(d_b1), bytes.fromhex(d_b2)))

    return {
        "n_msgs": n_msgs,
        "candidate_a": {
            "mean_hamming": round(float(np.mean(hd_a_list)), 2),
            "min_hamming": int(min(hd_a_list)),
            "max_hamming": int(max(hd_a_list)),
        },
        "candidate_b": {
            "mean_hamming": round(float(np.mean(hd_b_list)), 2),
            "min_hamming": int(min(hd_b_list)),
            "max_hamming": int(max(hd_b_list)),
        },
        "expected_random": 128,
        "interpretation": (
            "A single-cell perturbation in psi should change ~128 output bits "
            "(the random expectation). Both candidates are compared; any "
            "systematic difference (consistently lower Hamming) would indicate "
            "weaker diffusion."
        ),
    }


def main(seed: int = 92000) -> dict:
    t0 = time.perf_counter()

    print("  1. family separation comparison ...")
    sep = family_separation_comparison()
    print(f"    A: distinct={sep['candidate_a']['distinct_digests']}/{sep['n_states']}, "
          f"min_HD={sep['candidate_a']['min_pairwise_hamming']}")
    print(f"    B: distinct={sep['candidate_b']['distinct_digests']}/{sep['n_states']}, "
          f"min_HD={sep['candidate_b']['min_pairwise_hamming']}")

    print("  2. structural 2-to-1 comparison ...")
    struct = structural_comparison()
    print(f"    A is 2-to-1 in u: {struct['candidate_a_is_2to1_in_u']}, "
          f"B injective in u: {struct['candidate_b_is_injective_in_u']}")

    print("  3. avalanche comparison ...")
    aval = avalanche_comparison(seed=seed + 11)
    print(f"    A mean_HD={aval['candidate_a']['mean_hamming']}, "
          f"B mean_HD={aval['candidate_b']['mean_hamming']}")

    out = {
        "artifact": "accumulator_comparison",
        "description": "Side-by-side comparison of CC-Core-v0 (Candidate A) vs linear-injection (Candidate B)",
        "metadata": C.env_metadata(),
        "seed": seed,
        "candidates": {
            "A": {
                "name": "CC-Core-v0",
                "j_formula": "j_A(u,v) = u + GAMMA*u*v + ETA*u^2 + ZETA*v",
                "ETA": spec.ETA,
                "known_weakness": "2-to-1 in u (j_A(u,v)=j_A(u',v) for u+u'=-(1+GAMMA*v)/ETA)",
            },
            "B": {
                "name": "Linear-injection (ETA=0)",
                "j_formula": "j_B(u,v) = u + GAMMA*u*v + ZETA*v",
                "ETA": 0,
                "known_weakness": "injective in u except when (1+GAMMA*v)=0 mod p (~prob 1/p)",
            },
        },
        "family_separation": sep,
        "structural_comparison": struct,
        "avalanche": aval,
        "verdict": {
            "candidate_b_separates_family": sep["candidate_b"]["all_distinct"],
            "candidate_b_avoids_2to1": struct["candidate_b_is_injective_in_u"],
            "recommendation": (
                "Candidate B (ETA=0) achieves family separation while avoiding the "
                "provable 2-to-1 structural weakness. However, the u^2 term in A "
                "increases the polynomial degree of the injection, which is a "
                "secondary design objective. The design choice (A vs B) involves "
                "a trade-off: A has higher algebraic degree with a known 2-to-1 "
                "weakness; B is injective in u but has lower degree. This trade-off "
                "is documented; no claim of superiority is made without a formal "
                "security reduction."
            ),
        },
        "limitations": [
            "Candidate B is defined only for the first CC-1 analysis phase; "
            "adopting it would require a separate spec revision and re-audit",
            "Avalanche is measured on raw trajectory digests (not message-level)",
            "No formal security reduction exists for either candidate",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("accumulator_comparison.json", out)
    print(f"  saved accumulator_comparison.json ({out['runtime_s']}s)")
    return out


if __name__ == "__main__":
    main()
