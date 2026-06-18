# CC-Core-v1 / Candidate B — Shortcut Computation Audit

**Status:** Research analysis — Phase CC-2, Part IX. No forbidden claim.

A shortcut computes the trajectory commitment C_T without constructing the full
ordered trajectory at comparable cost. Candidate B uses a **linear** injection
(`j_B = u·(1+γv)`, degree 1 in u) where Candidate A used a **quadratic** one
(degree 2 in u). The concern (Part IX): does the lower injection degree make
Candidate B *easier* to shortcut?

---

## 1. The decisive finding: degree growth is identical

Scalar-model symbolic degree of the accumulator in the initial wave value u₀
(sympy, reduced rounds):

| round t | Candidate A degree | Candidate B degree |
|---|---|---|
| 1 | 4 | 4 |
| 2 | 12 | 12 |
| 3 | 36 | 36 |
| 4 | 108 | 108 |

**They are identical.** The reason: the degree is set by the *cross term*
`γ·u·v` (shared by both candidates), where `v = F(ψ)` has degree 3 in u₀, so
`γ·u·v` has degree 4 — strictly higher than A's removed `η·u²` (degree 2) and
`ζ·v` (degree 3) terms. Removing those lower-degree terms does **not** reduce the
leading degree. The accumulator self-square `A_C·cd²` (identical in both) then
triples the degree each round.

**Conclusion:** Candidate B is *not* lower-degree in the variable that matters
(u₀). The lower *injection* degree does not translate into a lower *trajectory*
degree.

---

## 2. Shortcut attempts and classifications

| Shortcut | Result | Classification |
|---|---|---|
| Symbolic recurrence compression | degree growth identical to A (self-square dominates) | no asymptotic improvement |
| Transfer-operator composition | wave round nonlinear; no linear operator | no asymptotic improvement |
| Blockwise composition | per-block map depends on global round index (rho_t, W_t continue across blocks); not a fixed monoid action | no asymptotic improvement |
| Affine factor extraction | not affine even at 2 rounds (high "affine" fraction is unreached cells, not structure) | failed within budget |
| Low-rank / spectral diagonalization | map is nonlinear; no applicable linear spectrum | not applicable |
| Checkpoint elimination | round-dependent constants prevent reuse | no asymptotic improvement |
| Meet-in-the-middle on C | requires inverting Φ_t^(B): a 256-variable **degree-2** system (from the shared `A_C·cd²`); B does NOT lower this degree | no asymptotic improvement |
| Direct polynomial composition | degree grows via shared self-square (108 by round 4) | infeasible |
| Linear predictor (psi0 bits → digest) | accuracy A 0.4998, B 0.4986 (~random) | no shortcut |

---

## 3. Why the linear injection does not help the attacker

One might expect: with the wave trajectory fixed (all `v` known), Candidate B's
accumulator is *linear in the injected u-sequence* through the `W_t·j_B` term,
whereas Candidate A's is quadratic. This is true of the **injection term in
isolation**, but:

- The accumulator carries a **self-square** `A_C·cd²` every round (shared with A),
  which is degree 2 in C. Across T rounds this makes C_T a degree-`2^T`-ish
  polynomial in the injected values regardless of the injection's own degree.
- The cross term `γ·u·v` couples u to `v = F(ψ)`, and v is degree 3 in the wave
  history, so even the injection is degree 4 in u₀ (not 1).

So the "linear injection" is linear only in the *instantaneous* u for a *fixed*
v; it is not linear in the message or in the wave history.

---

## 4. Backward-inversion (MITM) degree is shared

Inverting one accumulator round (needed for MITM) means solving
```
MU·cd + A_C·cd² + W_t·j_B + rho_t = C_{t+1}   (cd = C + D_C·Lap(C))
```
for C. The quadratic term `A_C·cd²` is **identical** in Candidate A and B, so the
backward system is degree 2 in C for both. Candidate B's linear injection does
**not** lower the inversion degree. MITM is no easier for B.

---

## 5. Verdict

**Candidate B's shortcut resistance is no worse than Candidate A's.** The lower
injection degree does not yield any exact or partial shortcut, because the
high-degree contributions (wave round degree 3, accumulator self-square degree 2)
are shared and dominate. No shortcut was found within budget. This is **not** a
proof that no shortcut exists.

This addresses the explicit Part IX concern: removing the 2-to-1 flaw did **not**
make Candidate B easier to shortcut.
