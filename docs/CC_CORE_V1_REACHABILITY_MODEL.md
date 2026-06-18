# CC-Core-v1 / Candidate B — Reachability Model

**Status:** Phase CC-3, Parts II–III. Defines the singular-reachability questions
and characterizes the message-absorption image. Labels exact vs approximate.

---

## 1. The four reachability questions (Part II)

Candidate B injection `j_B(u,v) = u·(1+γv)`; singular value
`v_star = −γ⁻¹ mod p = 195225786`; `j_B(u, v_star) = 0` for all u.

Let `ψ_t(m)` be the wave state at round `t` for valid message `m` (normative
protocol). The questions are kept **separate**:

- **(A) Coordinate reachability.** ∃ valid `m`, round `t`, coordinate `x` with
  `ψ_t(m)[x] = v_star`?
- **(B) Structured-subset reachability.** Can a valid `m` put `v_star` on a row /
  column / checkerboard / periodic / positive-density subset of coordinates at
  some round?
- **(C) Full-lattice reachability.** ∃ valid `m`, round `t` with
  `ψ_t(m)[x] = v_star` for **every** `x`?
- **(D) Persistent reachability.** Can a valid trajectory enter the singular set
  and remain on / repeatedly return to it across rounds?

These are NOT collapsed into one yes/no answer.

---

## 2. The post-absorption image R₀ (exact, Part III)

After absorbing block 0 into the IVs (before the first wave round), the state
`ψ_0(m)` has, per coordinate:

| coordinate | value | controllable range |
|---|---|---|
| `x ∈ 0..63` (rate) | `ψ_IV[x] + elem[x] mod p`, `elem[x] ∈ [0, 2²⁴)` | `[ψ_IV[x], ψ_IV[x]+2²⁴−1]` ⊂ `[54, 16777397]` |
| `x = 64` (CAP0) | `ψ_IV[64] + (k+1)·G mod p` | counter-fixed (small for bounded `m`) |
| `x = 65` (CAP1) | `ψ_IV[65] + D_TAG mod p = 1464026030` | fixed (last block) |
| `x = 66` (CAP2) | `ψ_IV[66] + ((k+1)//p)·G mod p` | counter-fixed (0 for short `m`) |
| `x ∈ 67..255` | `ψ_IV[x]` | **fixed** (∈ `[123, 374]`, never injected) |

**Exact properties of R₀ (one block):**
- **Affine:** `ψ_0` is an affine function of the injected bytes (additive
  injection mod p). *Exact.*
- **Strict subset / not surjective:** each rate cell ranges over exactly `2²⁴`
  consecutive residues out of `p`; 192 coordinates are fixed. `R₀` occupies a
  measure-`(2²⁴/p)⁶⁴ · 0` (literally a 64-dimensional, byte-bounded slab) corner
  of `F_p²⁵⁶`. *Exact.*
- **Independent control:** the 64 rate cells are independently controllable
  (each by its own 3 message bytes); cells 64,66 are counter-fixed, 65 tag-fixed,
  67..255 are uncontrollable. *Exact.*
- **Padding limits:** the empty message already occupies 2 blocks; the trailing
  length block constrains the last block's bytes (they encode the bit length),
  reducing controllability of the final block. *Exact.*
- **Multiple blocks enlarge R:** after block 0's `T` rounds the state is
  full-range; block 1 then adds `elem ∈ [0,2²⁴)` to cells 0..63 of that
  full-range state. So `R` grows with blocks, but each block only *additively*
  controls 64 cells over a `2²⁴` window. *Exact.*

**Immediate corollary (exact):** since every coordinate of `ψ_0(m)` is either
`≤ 16777397 < v_star`, or one of the fixed constants `{1464026030 (CAP1),
ψ_IV[64]+7, ψ_IV[66], ψ_IV[67..255] ∈ [123,374]}` — none of which equals
`v_star = 195225786` — **no valid message reaches `v_star` at round 0.** (This is
the bounded theorem of `CC_CORE_V1_VSTAR_REACHABILITY.md`, Option B.)

---

## 3. The forward-orbit reachable set R = ⋃ₜ Fᵗ(R₀) (approximate)

For `t ≥ 1` the wave round `F` is a nonlinear, globally-coupled, degree-3 map and
spreads coordinate values across all of `F_p`. We do **not** have a closed-form
description of `R_t` for `t ≥ 1`; the following are *approximations / empirical
characterizations*, explicitly labeled:

- **(approx)** `F` mixes the byte-bounded slab `R₀` toward an apparently uniform
  distribution over `F_p` per coordinate within a few rounds (consistent with the
  Phase 8A avalanche/saturation findings for Design A). Under a uniform-image
  heuristic, the probability that a given coordinate equals `v_star` at a given
  round is `≈ 1/p ≈ 4.66×10⁻¹⁰`.
- **(approx)** Expected singular coordinate-hits across one full message
  trajectory (`T·N² ≈ 8192` coordinate evaluations) `≈ 8192/p ≈ 3.8×10⁻⁶` — i.e.
  essentially never for any single message, absent a structural attractor.
- **(empirical)** No attractor toward `v_star` was observed: `v_star` is not a
  wave fixed point (`F(v_star) = 1006784219 ≠ v_star`), and 0 transits were seen
  over `T` rounds from 400 random states (Phase CC-2).

`R` is therefore conjectured to intersect the singular hyperplane only at
measure-`~1/p` incidental coordinates, with **no** valid-message construction of a
structured or full-lattice singular state known. Parts IV–VIII test this with
exact preimage analysis, solvers, exhaustive enumeration, and guided search.

---

## 4. What is exact vs approximate

| Statement | Status |
|---|---|
| `ψ_0(m)` is an affine, byte-bounded image; cells 67..255 fixed | **exact** |
| No coordinate of `ψ_0(m)` equals `v_star` (round 0) | **exact** |
| Each rate cell ranges over exactly `2²⁴` residues | **exact** |
| `F` drives `R₀` toward uniform per-coordinate by a few rounds | approximate (heuristic) |
| Per-coordinate `v_star` incidence `≈ 1/p` at rounds ≥ 1 | approximate (heuristic) |
| No structured/full-lattice singular state is valid-message reachable | conjecture (tested in Parts IV–IX, not proved for `t ≥ 1`) |
