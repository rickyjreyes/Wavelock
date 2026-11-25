"""Part IV -- explicit resource models (computation / physics / machine).

This module computes concrete numbers for three resource models and states, for
each, exactly what it does and does not bound. The central honesty point:

  * Landauer / Margolus-Levitin / Bremermann / Bekenstein bounds limit a
    *physical implementation* (irreversible-erasure energy, operation rate,
    information density). They are NOT lower bounds on the number of operations
    required to invert, collide, or forge the digest. No such lower bound is
    proved anywhere in this repository.

So the "heat" argument is INTERPRETIVE, not a proof of hardness. We make the
distinction quantitative below.
"""

from __future__ import annotations

import time

import numpy as np

from wavelock.curvature_capacity import spec
from . import _common as C

# physical constants (SI)
K_B = 1.380649e-23            # Boltzmann constant, J/K
LN2 = 0.6931471805599453
HBAR = 1.054571817e-34        # reduced Planck, J*s
H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8        # m/s
T_ROOM = 300.0                # K


# --- Model 1: abstract computation -------------------------------------
def forward_op_count(blocks: int = 1) -> dict:
    """Field-operations per forward digest evaluation (order of magnitude).

    Per coupled round: wave round ~ c1*N^2 ops; accumulator step ~ c2*N^2 ops.
    Per block: T rounds. Squeeze: 4 rounds * (...). We count modular mul/add as
    unit ops with small constants from the spec formulas.
    """
    n2 = spec.N_CELLS
    # wave round: ~ (square + reaction 2 mul + laplacian 4 add + 1 mul + adds) ~ 8 ops/cell
    wave = 8 * n2
    # accumulator: lap(4add+1mul) + cd + cd^2(1mul) + w*j (j: 3mul+adds) ~ 12 ops/cell
    acc = 12 * n2
    per_round = wave + acc
    rounds = blocks * spec.T + 3 * spec.T  # absorb + 3 squeeze re-evolutions
    total = per_round * rounds
    return {"ops_per_round": per_round, "rounds": rounds,
            "field_ops_per_digest": total,
            "asymptotic": "O(blocks * T * N^2) field ops (polynomial in all "
                          "parameters); forward cost is poly(n).",
            "R_comp_definition": "R_comp(n) = field operations as a function of "
                                 "message length n (linear in #blocks)."}


# --- Model 2: information-theoretic physical bounds --------------------
def physical_bounds(n_operations: float, T_env: float = T_ROOM) -> dict:
    """Numbers for the established physical bounds, given a hypothetical attack
    that performs ``n_operations`` irreversible bit operations / state visits.

    Each bound is annotated with what it actually constrains.
    """
    landauer_per_bit = K_B * T_env * LN2           # J per irreversible bit erase
    landauer_total = landauer_per_bit * n_operations
    # Margolus-Levitin: max operations/sec for a system of average energy E:
    #   nu_max = 2E/(pi*hbar)  => to do n_operations in time t needs E >= n*pi*hbar/(2t)
    # Bremermann: max bits/sec processed per joule  ~ 2E/(pi*hbar) as well; the
    # canonical Bremermann limit is ~1.36e50 ops per second per kg.
    bremermann_ops_per_s_per_kg = 2 * (C_LIGHT ** 2) / (np.pi * HBAR)  # ~ c^2/(...)
    return {
        "assumed_irreversible_operations": n_operations,
        "T_env_K": T_env,
        "landauer_energy_per_bit_J": landauer_per_bit,
        "landauer_total_energy_J": landauer_total,
        "landauer_total_energy_note":
            "energy to ERASE this many bits irreversibly at T_env. It bounds an "
            "implementation's dissipation; it does NOT lower-bound the number of "
            "operations an attacker must perform. A reversible circuit can in "
            "principle approach zero erasure energy.",
        "bremermann_ops_per_s_per_kg": bremermann_ops_per_s_per_kg,
        "R_phys_definition":
            "R_phys(E,V,t,T_env): given energy E, volume V, time t and "
            "temperature T_env, the max #irreversible operations is bounded by "
            "min(E/(k_B T_env ln2) [Landauer], (2E/(pi*hbar))*t [Margolus-"
            "Levitin], Bekenstein(V,E) [holographic info]). These bound an "
            "implementation, not the problem.",
        "caveat": "Landauer alone does NOT prove computational hardness.",
    }


def attack_vs_physical(security_bits: int = 256) -> dict:
    """Contrast a generic 2^k attack work figure with physical capacity.

    This is illustrative: IF an attack required 2^security_bits irreversible
    operations, what energy would Landauer imply? We then state plainly that no
    such lower bound on operations is proved -- the construction is only
    *conjectured* (and only empirically, to a shallow truncation) to need
    generic 2^k work.
    """
    n_ops = 2.0 ** security_bits
    pb = physical_bounds(n_ops)
    # energy of the observable universe ~ 4e69 J (mass-energy); the sun outputs
    # ~3.8e26 W. Provide comparison scales.
    return {
        "hypothetical_attack_operations": f"2^{security_bits}",
        "landauer_energy_if_irreversible_J": pb["landauer_total_energy_J"],
        "comparison_scales_J": {
            "annual_world_energy_use": 6.0e20,
            "sun_luminosity_one_year": 1.2e34,
            "mass_energy_observable_universe": 4.0e69,
        },
        "honest_statement":
            "the energy figure is only meaningful IF the attack provably needs "
            "2^k IRREVERSIBLE operations. No lower bound of this kind is proved "
            "for the curvature-capacity core. Generic-cost behaviour was observed "
            "only on truncations <= 24 bits. Therefore the heat/energy argument "
            "is INTERPRETIVE and decorative, not a security proof.",
    }


# --- Model 3: concrete machine model -----------------------------------
def machine_model() -> dict:
    """A reproducible machine-budget model (commodity + large-cluster)."""
    return {
        "commodity_cpu": {
            "ops_per_s": 1e10, "memory_GB": 32, "power_W": 200,
            "wall_clock_budget_s": 86400,
        },
        "large_cluster": {
            "ops_per_s": 1e18, "memory_PB": 10, "power_MW": 30,
            "wall_clock_budget_s": 3.15e7,
        },
        "R_machine_definition":
            "R_machine = (ops/s) * (wall-clock s) bounded by power/cooling; e.g. "
            "a 1 exaflop cluster for one year ~ 3.15e25 field-ops, far below "
            "2^256 ~ 1.16e77. A generic 256-bit digest is out of reach for any "
            "machine model here -- IF the attack is generic, which is unproved.",
        "forward_throughput_measured_hashes_per_s": None,  # filled by main()
    }


def main(seed: int = 90400) -> dict:
    t0 = time.perf_counter()
    # measure forward throughput
    from wavelock.curvature_capacity import optimized as opt
    tt = time.perf_counter()
    for i in range(100):
        opt.cc_hash(i.to_bytes(8, "big"))
    hps = 100 / (time.perf_counter() - tt)

    fwd = forward_op_count(blocks=1)
    mm = machine_model()
    mm["forward_throughput_measured_hashes_per_s"] = round(hps, 1)
    out = {
        "phase": "resource_bounds",
        "metadata": C.env_metadata(),
        "seed": seed,
        "model_1_computation": fwd,
        "model_2_physical": {
            "definition_and_bounds": physical_bounds(fwd["field_ops_per_digest"]),
            "attack_vs_physical_256bit": attack_vs_physical(256),
        },
        "model_3_machine": mm,
        "verdict_on_heat_argument":
            "The heat / Landauer argument is INTERPRETIVE. It quantifies the "
            "dissipation of an implementation, not a lower bound on attack "
            "operations. No exponential lower bound on attack cost is proved; "
            "therefore 'heat makes inversion impossible' is NOT a supported claim.",
        "limitations": [
            "op counts are order-of-magnitude, not gate-exact",
            "physical bounds constrain implementations, not the problem",
            "no proof connects curvature resolution to attack operations",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("resource_bound_analysis.json", out)
    print("  saved resource_bound_analysis.json", f"({out['runtime_s']}s)",
          "forward", mm["forward_throughput_measured_hashes_per_s"], "h/s")
    return out


if __name__ == "__main__":
    main()
