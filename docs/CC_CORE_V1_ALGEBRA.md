# CC-Core-v1 / Candidate B — Algebraic Structure Analysis

**Status:** Research analysis — Phase CC-2, Part III. Performed BEFORE any
preference claim. No forbidden claim.

Candidate B's injection is
```
j_B(u, v) = u + GAMMA·u·v = u·(1 + GAMMA·v)   (mod p),    GAMMA = 11, p = 2³¹−1.
```
This document answers the eight mandatory structural questions (Part III) and
records the singular-hyperplane analysis that gates Candidate B's promotion.

---

## 1. Injectivity in u for fixed v

For fixed v, `j_B(·, v)` is **linear** in u with slope `s(v) = (1 + GAMMA·v) mod p`:
```
j_B(u, v) = s(v)·u.
```
- If `s(v) ≠ 0`: the map `u ↦ s(v)·u` is a bijection of F_p (multiplication by a
  nonzero element), hence **injective in u**.
- If `s(v) = 0`: the map is the zero map — `j_B(u, v) = 0` for **all** u.

**Contrast with Candidate A:** j_A had a symmetric `η·u²` term making it generically
**2-to-1** in u for *every* v. Candidate B is **injective in u for every v except the
single value where s(v) = 0**. This is the central structural improvement, traded
for the singular value below.

## 2. Where is `1 + GAMMA·v ≡ 0 (mod p)`?

Exactly one value, since GAMMA is invertible mod p:
```
v_star = −GAMMA⁻¹ mod p = 195225786.
```
(GAMMA⁻¹ = 1952257861; `1 + 11·195225786 = 2147483647 ≡ 0 mod p`. Verified.)

## 3. Does v_star create a full collapse?

Yes: `j_B(u, v_star) = u·0 = 0` for every u (confirmed for 2000 random u). At any
coordinate where the *later* wave value equals v_star, the injection contributes
**nothing** that round, independent of the *earlier* value u.

## 4. Are v_star values reachable as wave-state coordinates?

- **Random states:** 0 hits in 1,280,000 wave-output coordinates (expected uniform
  rate ≈ 1/p ≈ 4.7×10⁻¹⁰). No structural attraction.
- **Phase 8J family / eigenmodes:** 0 of 47 states produce a v_star coordinate
  (they collapse the wave to terminal 0, and 0 ≠ v_star).
- **Constant-field construction (exact):** solving the cubic
  `F(c) = c + A·c·(B − c²) = v_star (mod p)` over GF(p) has a **unique root**
  `c = 357959172`. For ψ ≡ c, `F(ψ) = v_star` on **all 256 cells** — a one-round
  full-lattice singular collapse. This is the strongest singular construction found.

## 5. Can a structured field force many coordinates into v_star?

The constant state ψ ≡ 357959172 forces **all** coordinates to v_star for one
round (§4). However this is a **single isolated state** (unique cubic root); there
is no family of distinct states with full-lattice collapse via the constant ansatz.
Partial-lattice singular constructions require solving wave-round preimage
constraints (coupled, preimage-hard) and none was found by bounded search.

## 6. Can the singular factor propagate round to round?

No sustained propagation observed. v_star is **not a fixed point**:
`F(v_star) = 1006784219 ≠ v_star`. Starting from ψ ≡ c, the next state is ψ ≡ v_star
(constant), and `F(v_star)` is a different constant — the lattice leaves the singular
hyperplane after one round. Over T = 32 rounds from 400 random states, 0 coordinate
transits through v_star (expected ≈ 0.0015). There is no attractor.

## 7. Does the accumulator's other mixing remove or preserve the collapse?

Even when `j_B = 0` at a cell, the accumulator still updates via
`C_{t+1}[x] = MU·Cd[x] + A_C·Cd[x]² + rho_t` (the self-diffusion, self-square, and
round constant are all nonzero). So a singular round zeroes only the *injection
contribution* at that cell that round; C keeps evolving. **Path-erasure test:**
perturbing the full-collapse constant state ψ ≡ c at any single cell changed the
trajectory digest in **256 / 256** trials — the perturbed cell still enters the
accumulator at adjacent rounds (as u at round t, as the v-multiplier at round t−1)
and through wave diffusion. **No path erasure.**

## 8. Can an attacker choose (u, v) jointly to create equal injections?

The zero set of j_B is exactly the union of two lines:
```
{ j_B = 0 } = { u = 0 } ∪ { v = v_star }.
```
This is `2p − 1` pairs out of `p²` (fraction ≈ 2/p → 0). More generally
`j_B(u, v) = j_B(u', v')` whenever `u·(1+γv) = u'·(1+γv')`, a codimension-1 variety.
**But u and v are not free:** v = F(ψ)[x] is determined by the wave round, and u is
the previous wave value. The attacker cannot independently set (u, v) at a
coordinate without solving wave-round preimage constraints. No digest collision was
produced from this structure (Part IV, Part VI).

---

## 9. Singular-set measure comparison (Candidate A vs B)

| | Candidate A (j_A) | Candidate B (j_B) |
|---|---|---|
| Injection degree in u | 2 (quadratic) | 1 (linear) |
| Generic multiplicity in u | **2-to-1 everywhere** (proved) | 1-to-1 except on v = v_star |
| Zero set of j | variety of size ~p (the parabola) | `{u=0} ∪ {v=v_star}`, size 2p−1 |
| Singular hyperplane | none distinguished | v = v_star (measure 1/p) |
| Reachability of singular set | n/a | ~1/p random; one exact constant construction |
| Path erasure from singular set | n/a | none observed |

Both zero sets have measure of order 1/p relative to the (u,v) plane. Candidate B
replaces a **generic, everywhere-present** 2-to-1 fold with a **measure-1/p**
singular hyperplane that is hard to reach and does not erase path information.

## 10. Reduced-model confirmation

Exhaustive over the (u,v) plane at p ∈ {3,5,7,11,13}: the zero set is exactly
`{u=0} ∪ {v=v_star}` with `2p−1` pairs, matching the algebra. The singular line
does not concentrate collisions beyond measure ~1/p.

## 11. Conclusion of the algebraic analysis

Candidate B is **injective in u away from a single measure-1/p hyperplane**, in
exchange for removing Candidate A's **generic 2-to-1 fold**. The singular
hyperplane v = v_star is:
- constructible (unique constant state c = 357959172, one-round full collapse),
- not broadly reachable (random/eigenmode/family hit rate ~1/p, no attractor),
- not path-erasing (256/256 perturbations change the digest).

**Candidate B is NOT yet declared algebraically stronger** — that depends on the
full-family regression (Part V), A/B comparison (Part VI), joint-map analysis
(Part VII), solver attacks (Part VIII), and shortcut audit (Part IX). This section
establishes only that the singular set has been fully characterized and is not a
broad, reachable, path-erasing weakness.
