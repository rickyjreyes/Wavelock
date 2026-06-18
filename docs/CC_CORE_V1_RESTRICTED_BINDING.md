# CC-Core-v1 / Candidate B — Restricted Binding Results

**Status:** Research analysis — Phase CC-2, Part X. Results are labeled by rigor.
No forbidden claim. No unconditional global cryptographic proof is attempted.

We establish bounded, explicit results under stated assumptions. Each is labeled:
**[theorem]** (proved analytically), **[computer-assisted theorem]** (proved with
exact-arithmetic finite verification), or **[exhaustive finite verification]**.

---

## Theorem 1 (injectivity of the injection in u, off the singular hyperplane) — [theorem]

**Statement.** Fix v ∈ F_p with `1 + γ·v ≠ 0` (equivalently v ≠ v_star). Then the
map `u ↦ j_B(u, v) = u·(1 + γ·v)` is a bijection of F_p; in particular it is
injective in u.

**Proof.** `1 + γ·v` is a nonzero element of the field F_p. Multiplication by a
nonzero element of a field is a bijection (it has inverse multiplication by
`(1+γ·v)⁻¹`). Hence `u ↦ u·(1+γ·v)` is a bijection. ∎

**Solver corroboration.** z3 confirms `j_B(a,v)=j_B(b,v) ∧ a≠b` is UNSAT for a
concrete v = 12345678 ≠ v_star (Part VIII, `candidate_b_algebraic_solver.json`).

**Contrast.** Candidate A is *never* injective in u (its `η·u²` term makes it
generically 2-to-1 for every v). Candidate B is injective in u for **all** v
except the single value v_star (measure 1/p).

---

## Theorem 2 (first-round sign-pair separation) — [theorem]

This is the central restricted binding result for the Phase 8J family.

**Lemma (oddness of the wave round).** The Design A round
`F(ψ) = ψ + D·Lap(ψ) + A·ψ·(B − ψ²)` is an **odd** function: `F(−ψ) = −F(ψ)`.

*Proof.* Each term is odd in ψ: ψ is odd; `Lap` is linear hence odd; and
`A·ψ·(B − ψ²) = A·B·ψ − A·ψ³` is a sum of odd-degree monomials. Sum of odd
functions is odd. ∎ (Verified numerically on 50 random states.)

**Statement.** Let σ ∈ {−1, +1}^256 be any sign field and s ≠ 0 an amplitude,
giving the Phase 8J sign pair Ψ⁺ = s·σ and Ψ⁻ = −s·σ (both elements of the
47-state zero-collapse family come in such ± pairs). Run **one** Candidate B
coupled round from the accumulator IV. Then the injection vectors differ at
**every** cell x:
```
j_B(Ψ⁺) [x] − j_B(Ψ⁻)[x] = 2·s·σ[x]  ≠ 0   (mod p),    for all x.
```

**Proof.** Write u = s·σ[x] and v = F(Ψ⁺)[x]. By the Lemma, F(Ψ⁻)[x] = F(−s·σ)[x]
= −F(s·σ)[x] = −v. Therefore
```
j_B(Ψ⁺)[x] = u·(1 + γ·v) = u + γ·u·v,
j_B(Ψ⁻)[x] = (−u)·(1 + γ·(−v)) = −u + γ·u·v.
```
Subtracting: `j_B(Ψ⁺)[x] − j_B(Ψ⁻)[x] = 2u = 2·s·σ[x]`. Since 2 ≠ 0 mod p, s ≠ 0,
and σ[x] ∈ {−1,+1} ≠ 0, the difference is nonzero at every cell. ∎ (Verified by
exact arithmetic.)

**Significance.** The injection is the *only* channel through which the wave
trajectory enters the accumulator C. Theorem 2 shows that for every Phase 8J sign
pair, that channel separates the two states at **every coordinate in the first
round**, and the separation amount `2·s·σ[x]` is **independent of the wave output
v** — so it cannot be cancelled by any choice of wave dynamics.

**Contrast with Candidate A.** For Candidate A the analogous first-round
separation is `2·s·σ[x] + 2·ζ·F(s·σ)[x]`, which *could* vanish at a cell where
`s·σ[x] = −ζ·F(s·σ)[x]`. Candidate B's separation `2·s·σ[x]` is structurally
incapable of vanishing. In this restricted sense Candidate B separates sign pairs
**more robustly** than Candidate A.

**Limitation.** Theorem 2 is an *injection-level* (first-round) separation. It does
not by itself prove the *digest-level* separation, because the accumulator's
subsequent self-mixing (`A_C·cd²`, diffusion) could in principle re-merge two
states. The digest-level separation for the full family is established by exhaustive
verification (below), not by Theorem 2 alone.

---

## Exhaustive finite verification (full 47-state family, digest level) — [exhaustive finite verification]

**Statement.** For the complete Phase 8J zero-collapse family {Γ⁽¹⁾, …, Γ⁽⁴⁷⁾}
(all 46 nonzero periodic-tile eigenmode states + the zero state),
```
Γ⁽ⁱ⁾ ≠ Γ⁽ʲ⁾  ⟹  C_{T,B}(Γ⁽ⁱ⁾) ≠ C_{T,B}(Γ⁽ʲ⁾)   for all i, j,
```
with minimum pairwise Hamming distance 105 / 256.

**Method.** Exact enumeration of all 47 states and all 1081 pairwise digest
comparisons (`candidate_b_full_family_binding.json`, regression test
`test_candidate_b.py`). This is a finite, exhaustive check over the *known*
family — **not** a general collision-resistance theorem.

---

## What is NOT proved

- No claim that Candidate B is collision-resistant, one-way, or "provably secure".
- Theorem 2 is injection-level for one round; it is not lifted to a digest-level
  theorem for arbitrary (non-sign-pair) state pairs.
- The 47-state separation is an exhaustive finite check, not a statement about
  all of F_p^256.
- General trajectory uniqueness and hardness of inversion remain **unresolved**
  (see `WAVELOCK_PROVER_VERIFIER_PROTOCOL.md` layer separation).

---

## Summary of rigor labels

| Result | Label |
|---|---|
| Injectivity of j_B in u for v ≠ v_star | theorem (+ z3 corroboration) |
| Oddness of F | theorem (+ numeric check) |
| First-round sign-pair separation = 2·s·σ[x] ≠ 0 | theorem (+ exact-arithmetic check) |
| 47-state digest separation (min HD 105) | exhaustive finite verification |
| General collision resistance / hardness | NOT proved (open) |
