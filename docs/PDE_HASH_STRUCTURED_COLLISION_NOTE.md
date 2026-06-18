# WaveLock-PDE-256-v0 — Structured Laplacian-Eigenmode Collision (Phase 8J)

**Status:** verified internal-state finding. Branch
`research/pde-eigenmode-collision-audit`. Artifact:
`pde_audit/artifacts/phase8j_eigenmode_collisions.json`. Regression tests:
`pde_audit/test_eigenmode_collisions.py`.

> **Observed attack cost is not a lower bound and is not a proof of one-wayness,
> collision resistance, or cryptographic security.**

## 1. Summary

There is an **exact, constructive collision family in the normative one-round
state transformation** `F` (and therefore in `evolve_T`, `T=32`). At least
**46 distinct structured states**, plus the all-zero state, map to the all-zero
state after a single round. This was independently re-derived and verified by
**element-by-element state equality** in the pure-Python reference round, the
optimized NumPy round, the parameterized audit harness, and `evolve_T`.

Consequence: **the state transformation is NOT globally injective** (and not
globally one-way at these structured points). However, **no end-to-end message
collision or message-level preimage of zero was demonstrated**: message
absorption controls only 64 of 256 state cells, and the 189 capacity cells
`67..255` are pinned to fixed IV-derived values that no structured collision
state can match. The complete digest verdict therefore remains *Unresolved
experimental candidate*, now carrying this explicit internal-non-injectivity
finding.

## 2. Independent derivation (recomputed, not trusted)

`F(ψ) = ψ + D·L(ψ) + a·ψ·(b − ψ²) (mod p)`, with the 5-point toroidal Laplacian
`L`. For a sign field `ψ = s·σ`, `σ∈{−1,+1}`, where every site has exactly `r`
opposite-sign neighbors:

```
L(σ) = (4−2r)σ − 4σ = −2r·σ          (σ is a Laplacian eigenvector, λ = −2r)
F(s·σ) = s·σ·[ 1 − 2rD + a(b − s²) ]  (mod p)
```

`F(s·σ) = 0` (all-zero) iff the bracket vanishes:

```
s² ≡ b − (2rD − 1)·a⁻¹   (mod p)          [division = modular inverse of a]
```

With `p = 2³¹−1, D = 5, a = 3, b = 1431655765`: independently,
`a⁻¹ = 3⁻¹ = 1431655765 = b` (a consequence of `b ≈ 2p/3`). Per-`r` results
(all recomputed and verified):

| r | s² = b − (2rD−1)/a | QR? | roots (±s) | valid sign config on 16×16 |
|---|---|---|---|---|
| 0 | 715827883 | **no** | — | (constant field) — no collision |
| 1 | 1431655762 (=b−3) | yes | 1217065103, 930418544 | period-4 cols `[+,−,−,+]` |
| 2 | 2147483641 (=p−6) | yes | 1395627816, 751855831 | row stripes / col stripes |
| 3 | 715827873 | **no** | — | — |
| 4 | 1431655752 (=b−13) | yes | 151946369, 1995537278 | checkerboard `(−1)^(i+j)` |

All reported candidate constants (checkerboard, stripe, r=1 square/roots) match
exactly. **r = 0 and r = 3 are quadratic non-residues**, so those classes have
no collision (in particular the constant field does **not** collide to zero).

## 3. Verified collisions (exact, all implementations)

For each of `checkerboard (r=4)`, `row stripes (r=2)`, `column stripes (r=2)`,
`period-4 columns (r=1)`, and for **both** amplitudes `+s` and `−s`:

```
one_round(s·σ)  == all_zero   (reference, optimized, harness)
evolve_T(s·σ)   == all_zero   (reference, harness; T=32)
one_round(0) == evolve_T(0) == all_zero   (zero is a fixed point)
```

Verification is exact element equality — no floating point, no tolerance, no
digest-of-state. See `test_eigenmode_collisions.py` (18 tests).

## 4. Constructive lower bound on preimages of zero

Enumerating periodic sign tiles up to 4×4 that are Laplacian eigenvectors with a
QR amplitude yields **46 distinct full sign patterns** (`r=1`: 8, `r=2`: 36,
`r=4`: 2) → **46 distinct nonzero states** mapping to zero in one round, plus the
all-zero state = **≥ 47 explicit one-round preimages of zero**. Under the
symmetry group (global sign flip, toroidal translation, rotation, reflection)
these reduce to **6 orbits**.

This is a **lower bound from ≤4×4 periodic tiles only**. The full set of ±1
Laplacian eigenvectors over 𝔽_p (hence preimages of zero) is **conjectured
larger** but is **not claimed exponential** without a constructive argument.

## 5. Jacobian at collision states — full rank ≠ injective

The one-round modular Jacobian is **full rank (256/256)** at *every* collision
representative tested: `zero`, `checkerboard(+s)`, `checkerboard(−s)`,
`row stripes`, `column stripes`, `period-4`.

Crucially, `checkerboard(+s)` and `checkerboard(−s)` have **identical** ψ²
(both equal `s²`), hence **identical Jacobians**, yet are **distinct states that
both map to zero**. This is a direct, constructive demonstration that a
**full-rank (locally non-singular) Jacobian does NOT imply global injectivity**.
Phase 8B's observation that sampled full-state Jacobians were full-rank was
therefore *only* evidence of local non-singularity at those random points, never
of global injectivity (see §7).

## 6. Three distinct claims

1. **The state transformation is globally injective.** → **FALSE** (this note).
2. **The state transformation is hard to invert for arbitrary targets.** →
   unresolved in general; but for the *specific* target `0` there is an explicit,
   simple, low-complexity preimage set (the eigenmodes above).
3. **The complete 256-bit message digest has practical collision/preimage
   resistance.** → **not refuted here**; see §7. A compression function may be
   internally many-to-one without that yielding a message-level collision.

## 7. Message-level reachability (why this is not an end-to-end break)

Message absorption writes only the **64 rate cells** `0..63` (additively), plus
the block counter into `cap0=64`, `cap2=66`, and finalization into `cap1=65`.
The **189 capacity cells `67..255` receive no injection ever.**

- **Before the first `evolve_T` (one-block message):** cells `67..255` hold fixed
  IV values `IV[c] = (1 + c + tag[c mod 19]) mod p` — all `< ~300`. A structured
  collision state requires every cell to be `±s` (`s ≈ 1.5×10⁸` or `1.2×10⁹`).
  All 189 uncontrolled cells mismatch the required `±s` pattern, so **no
  structured collision state is reachable as the pre-first-round state for any
  message** — an exact structural reason, not a failed search.
- **Multi-block / pre-final-round:** cells `67..255` then hold the previous
  block's `evolve_T` output, which is not controllable and would have to equal
  `±s·σ` in 189 coordinates simultaneously (≈ `p^−189`); no steering mechanism is
  known.
- **Bounded search:** 50 000 sequential messages produced **no** pre-squeeze
  zero state and **no** final-absorbed structured-collision state (exact state
  equality). Absence of a hit is reported with its budget; it is not a proof.
- The all-zero pre-squeeze state, *if reachable*, squeezes to the fixed digest
  `00…00` (all comparisons tie). No message reaching it was found.

**Why Phase 8B's random sampling missed this family:** the eigenmodes are a
vanishingly small, highly structured subset (≈46 of `p^256 ≈ 10^2400` states).
Uniform random full-state sampling and random message hashing have effectively
zero probability of landing on a Laplacian sign-eigenvector with the exact QR
amplitude, so the duplicate-search and Jacobian-sampling in 8B could not
encounter them. This attack is *mathematically targeted*, not statistical.

## 8. Verdict impact

- **State transformation:** *constructively non-injective*, with simple
  algebraic preimages of the all-zero state. It must **not** be described as
  collision-resistant or globally one-way.
- **Message digest:** remains **Unresolved experimental candidate** — no
  message-level collision or message preimage of zero was demonstrated, for the
  exact structural reason in §7. This is recorded as a separate, explicit
  finding rather than upgraded to "Broken".
