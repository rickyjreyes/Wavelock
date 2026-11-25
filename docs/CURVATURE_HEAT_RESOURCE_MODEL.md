# Curvature / Heat Resource Model

**Branch:** `research/curvature-capacity-wavelock` · **Date:** 2026-06-18
**Artifact:** `curvature_audit/artifacts/resource_bound_analysis.json`
(seed 90400, env recorded in artifact).

Every security statement about WaveLock must name the resource model it uses.
This document defines three, computes concrete numbers, and — crucially —
states for each exactly **what it bounds and what it does not**.

The central conclusion is stated up front:

> The Landauer / thermodynamic ("heat") argument is **interpretive, not a
> proof**. It bounds the dissipation of a *particular implementation*; it does
> **not** lower-bound the number of operations an attacker must perform. No
> exponential lower bound on attack cost is proved, so no heat figure can be
> converted into a security guarantee.

---

## Model 1 — Abstract computation, R_comp(n)

The attacker (and the honest evaluator) is charged in **field operations** over
F_p (p = 2³¹−1): modular multiply, add, reduce, compare; precision is fixed at
31 bits per element; memory in field elements.

Forward evaluation of the curvature-capacity core (measured by op-count in
`resource_bounds.forward_op_count`):

- per coupled round: ≈ 8·N² wave ops + 12·N² accumulator ops = 20·N² ≈ 5120 ops;
- rounds per single-block digest: B·T + 3·T = 4·32 = 128 (absorb T + 3 squeeze
  re-evolutions);
- **field ops per digest (1 block): ≈ 6.55 × 10⁵**, growing **linearly** in the
  number of message blocks.

So **R_comp(n) ∈ poly(n)** for the forward direction: O(blocks · T · N²). This is
the *forward cost*; it is **not** an attack cost.

What R_comp can express that the repo never proved: a lower bound of the form
"any algorithm outputting a preimage/collision uses ≥ L(n) field ops." This work
does **not** establish such a bound (Results, Part V).

---

## Model 2 — Information-theoretic physical computation, R_phys(E,V,t,T_env)

Established bounds, computed in the artifact. Each line says what it limits.

- **Landauer (irreversible erasure):** energy ≥ k_B·T·ln2 per *irreversibly
  erased bit*. At T = 300 K this is **2.87 × 10⁻²¹ J/bit**.
  *Bounds:* the heat an implementation must dump when it discards a bit.
  *Does not bound:* the number of operations to solve the problem; a logically
  reversible circuit can in principle erase nothing and approach zero Landauer
  cost. Evaluating the WaveLock map is reversible-circuit-embeddable.
- **Margolus–Levitin / Bremermann (operation rate):** a system of average energy
  E performs ≤ 2E/(π·ħ) state transitions per second (≈ 1.36 × 10⁵⁰ ops·s⁻¹ per
  kg of mass-energy).
  *Bounds:* speed of any physical computer.
  *Does not bound:* how many ops the *problem* needs.
- **Bekenstein (information density):** a region of radius R and energy E holds
  ≤ 2π·E·R/(ħ·c·ln2) bits.
  *Bounds:* how much state fits in a volume.
  *Does not bound:* problem hardness.

**Illustrative composition.** *If* an attack provably required 2²⁵⁶ irreversible
operations, Landauer would imply **≥ 3.32 × 10⁵⁶ J** — vastly more than annual
world energy use (~6 × 10²⁰ J) or a year of solar output (~1.2 × 10³⁴ J), though
far below the mass-energy of the observable universe (~4 × 10⁶⁹ J). **But the
antecedent is unproved.** Generic-cost behaviour for this construction was
observed only on truncations ≤ 24 bits (Results). The energy figure is therefore
**decorative**: it describes a hypothetical, not a guarantee.

`R_phys(E,V,t,T_env) =` the maximum number of *irreversible* operations
realizable under energy E, volume V, time t, temperature T_env, taken as the
minimum of the Landauer (E/(k_B·T_env·ln2)), Margolus–Levitin ((2E/π ħ)·t), and
Bekenstein (V,E) limits. This bounds a *machine*, not the *problem*.

---

## Model 3 — Concrete machine model, R_machine

| Machine | ops/s | memory | power | budget | total field-ops |
|---|---|---|---|---|---|
| commodity CPU | 10¹⁰ | 32 GB | 200 W | 1 day | ~8.6 × 10¹⁴ |
| exascale cluster | 10¹⁸ | 10 PB | 30 MW | 1 year | ~3.15 × 10²⁵ |

Measured forward throughput of the reference implementation: **49.3 digests/s**
(unoptimized NumPy; ~2× slower than Design A's 97 h/s because two fields are
co-evolved).

A generic 256-bit digest needs ~2²⁵⁶ ≈ 1.16 × 10⁷⁷ evaluations to brute-force —
out of reach for **any** machine in this table. **But "the digest is generic" is
exactly the unproved hypothesis.** R_machine only tells us the design is not
trivially breakable by enumeration *if* it is generic; it cannot establish that
it is.

---

## What the heat argument can and cannot do

| Claim | Status |
|---|---|
| "Forward evaluation dissipates ≥ E_min on irreversible hardware." | plausible/true for a *given irreversible implementation*; implementation-dependent, not a problem bound. |
| "Verification is cheaper than generation." | **not** demonstrated for this construction — there is no separate cheap verifier; verifying a digest means recomputing it (same cost as generating). The asymmetry asserted in `CURVATURELOWERBOUNDS.md` does **not** hold for a hash-style primitive. |
| "Heat makes inversion impossible." | **forbidden phrasing** — unsupported. Inversion cost is not lower-bounded at all. |
| "Attack cost exceeds physical capacity." | unsupported; requires a proved 𝒞_attack lower bound that does not exist. |

**Bottom line:** the resource models are well-defined and the forward cost is
honestly polynomial, but **no model yields a proved exponential attack-cost or
heat lower bound.** The thermal narrative is interpretive.
