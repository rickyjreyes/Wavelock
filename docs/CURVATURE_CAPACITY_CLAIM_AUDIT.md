# Curvature-Capacity Claim Audit

**Branch:** `research/curvature-capacity-wavelock`
**Date:** 2026-06-18
**Status:** audit of *existing* repository claims. Makes no new security claim.

This document extracts every load-bearing statement in the repository's
P-vs-NP / dimensional-lock / curvature / signature / thermal-capacity material
and classifies each into exactly one of four classes:

1. **Exact definition** — a precisely stated mathematical object.
2. **Proved theorem** — a statement with a complete proof in the repo.
3. **Numerical / experimental observation** — measured, finite-sample.
4. **Conjecture / physical intuition** — asserted, motivated, not proved.

The mandate: **do not upgrade a conjecture into a theorem.** Where a source
document uses theorem-like language for a class-4 statement, this audit records
the *actual* status and flags the gap.

Sources reviewed:

- `docs/inevitability/README.md` (frozen primitive definition + invariant)
- `docs/inevitability/CURVATURELOWERBOUNDS.md`
- `docs/inevitability/ATTACKBOUNDS.md`
- `docs/inevitability/FALSIFIABILITY.md`
- `docs/PDE_HASH_SPEC.md`, `docs/PDE_HASH_RESULTS.md`,
  `docs/PDE_HASH_MULTIBLOCK_REACHABILITY.md`,
  `docs/PDE_HASH_STRUCTURED_COLLISION_NOTE.md`
- `Wavelock A Curvature-Locked One Way Function ... .pdf` (motivational)

---

## 1. The frozen primitive definition and invariant

> "WaveLock is an irreversible state-evolution primitive such that no
> verifier-accepted commitment exists unless it arises from executing a
> curvature-bounded evolution whose terminal state ψ★ is reproducible under the
> declared kernel."

and the acceptance invariant

> 𝓘(ψ★; K) ≡ ( Serialize_K(ψ★) is canonical ∧ Budget_K(ψ★) ≤ C ).

**Classification: (1) exact definition** — but a definition of *acceptance
semantics*, not a hardness statement. It defines when a commitment is accepted;
it does **not** by itself assert that producing an accepting ψ★ is hard. The
word "irreversible" inside the definition is **(4) conjecture/intuition** until a
separate argument establishes it.

**Gap:** the definition is declarative ("no accepted commitment exists
unless…"). It presumes the conclusion (that ψ★ can only arise from executing the
evolution). Nothing in the definition rules out an algebraic shortcut that
produces an accepting ψ★ without running the declared evolution. The whole
security question is exactly whether such a shortcut exists; the definition
cannot answer it.

---

## 2. Curvature / energy lower-bound arguments (`CURVATURELOWERBOUNDS.md`)

The document's own header states they are "**not tight bounds and are not formal
complexity proofs.**" That self-assessment is correct. Itemized:

| Statement | As written | Actual class |
|---|---|---|
| "Key Claim: any accepted ψ★ must dissipate a strictly positive minimum curvature/energy budget E_min > 0." | asserted "non-negotiable" | **(4) conjecture** — no proof that an *attacker's* method must dissipate E_min; only that the *declared forward evolution* does. |
| Lyapunov descent ⇒ E_min ≥ L[ψ₀] − L[ψ★] > 0 | theorem-like | **(2) proved, but only for the forward trajectory**, AND **conditional** on a Lyapunov functional existing for the kernel. No Lyapunov functional is exhibited for the F_p Design-A map; over F_p the real-analytic Lyapunov theory does **not** transfer (see PDE_HASH_SPEC §1). So even the conditional theorem is **not instantiated**. |
| Curvature accumulation is global ⇒ E_parallel ≥ E_serial ≥ E_min | theorem-like | **(4) conjecture** — assumes the quantity an attacker must pay equals the forward accumulation. Unproven. |
| Entropy production ⇒ E_min ≳ k_B ΔS | physics analogy | **(4) physical intuition** — Landauer bounds *irreversible erasure* by an implementation, not the number of operations to solve the problem (see Resource Model doc). |
| Exact serialization ⇒ cost floor | argument | **(1)/(4)** — exact serialization is a real definitional fact (1); "forces cost" on an attacker is (4). |
| Cost_verify ≪ Cost_generate ≤ E_min | conclusion | **(4) conjecture** — the asymmetry is asserted, not derived. |

**Critical gap:** every lower-bound argument bounds the cost of the **honest
forward evaluation** (or the cost of a *reversible physical implementation*).
None bounds from below the cost of an **arbitrary algorithm** that produces an
accepting output. The leap from "the forward map dissipates energy" to "any
attacker must dissipate energy" is the entire unproven step. This is the gap the
curvature-capacity program must close and, per the present work, does **not**
close.

---

## 3. Attack-bounding arguments (`ATTACKBOUNDS.md`)

Five attack classes (adjoint inversion, gradient backprop, coarse-graining,
parallel decomposition, learned surrogates) are each argued to fail.

**Classification: (4) conjecture / heuristic** for all five. The document itself
says they are "attack-bounding arguments, **not** formal proofs." Each argument
has the same logical shape: "method X produces an *approximate* state, but
acceptance requires *exact* serialization, so X fails." This is valid **only
against approximate attacks**. It says nothing about an *exact* algebraic attack
— and Phase 8J found exactly such an attack at the state level (the
Laplacian-eigenmode family below), which the five-class taxonomy does not cover.

---

## 4. Phase 8J / 8K — the eigenmode collision family (PRESERVED)

These are the most rigorous results in the repository and are preserved
verbatim by this work. From `PDE_HASH_RESULTS.md` §8J/§8K:

| Statement | Class |
|---|---|
| For ψ = s·σ with σ a ±1 toroidal-Laplacian eigenvector (each site has r opposite-sign neighbours, L(σ) = −2r·σ), `F(s·σ) = 0` exactly iff `s² ≡ b − (2rD−1)/a (mod p)`. | **(2) proved theorem** (exact, verified element-by-element across all implementations). |
| For normative constants, r ∈ {1,2,4} give quadratic-residue amplitudes; ≥ 47 explicit states map to the all-zero state in one round (≥ 46 nonzero + zero). | **(2) proved** (constructive, enumerated). |
| The internal state transformation is **constructively non-injective**; "full-rank Jacobian ⇒ injective" is **false** (the ±s checkerboard pair shares an identical Jacobian and both map to 0). | **(2) proved** (constructive refutation). |
| Direct first-block reachability of a structured collision state is **impossible**. | **(2) proved** (cells 67..255 are fixed IV values that cannot equal ±s). |
| Multi-block reachability, generic arbitrary-target state inversion, full message collision / preimage resistance. | **(unresolved)** — neither proved nor broken. |
| Reduced-model message-level lifting: internal non-injectivity lifts to message-level collisions at ≈3 blocks in tiny fields (z3-verified); not extrapolated to N=16. | **(3) experimental observation** on toy fields. |

These are the **frozen baseline facts** the curvature-capacity core is built
against, and they are reproduced (independently re-derived) in
`curvature_audit/eigenmode_attacks.py`.

---

## 5. The three target inequalities

The task asks specifically whether the existing P-vs-NP / curvature work
establishes any of:

- **𝒞_attack(n) ≥ 2^{Ω(n)}** (attack cost grows exponentially):
  **NOT established.** No lower bound on attack cost of any kind is proved
  anywhere in the repository. The strongest related empirical fact is that
  *truncated* digest collisions and preimages track the generic birthday /
  brute-force rate up to ≤ 28 / ≤ 16 bits (Design A, Phase 8F/8G) — an
  **observation (3)** on small truncations, explicitly "not a lower bound."

- **Q_heat(n) ≥ 2^{Ω(n)}** (dissipated heat grows exponentially):
  **NOT established.** Forward evaluation heat is *polynomial* (linear in #blocks
  × T × N², see Resource Model doc). No exponential heat lower bound on *attacks*
  exists, because no exponential operation lower bound on attacks exists.

- **𝒞_attack(n) > 𝒞_physical(n)** (attack exceeds physical capacity):
  **NOT established**, and cannot be without first establishing 𝒞_attack. The
  comparison is only meaningful once a *proved* lower bound on 𝒞_attack exists;
  it does not.

**Conclusion of Part I:** the repository contains
- one genuine proved-theorem cluster (the eigenmode collision family, §4),
- a precise acceptance definition (§1),
- and a body of *structural arguments and physical intuitions* (§2, §3) that are
  explicitly self-described as not proofs.

No exponential lower bound (𝒞_attack, Q_heat) is proved. The curvature/heat
"cannot be cheap" property is, at present, a **class-4 conjecture**, not a
theorem. The curvature-capacity core must therefore be evaluated as an
experimental construction, not as a proven primitive.

---

## 6. Missing definitions / logical gaps to be supplied

1. **A measurable "curvature signature"** — supplied (candidate functionals) in
   `docs/CURVATURE_CAPACITY_CLAIM_AUDIT` → see
   `curvature_audit/curvature_metrics.py`. Finding: curvature *magnitude* is
   lifting-convention dependent and is a diagnostic only.
2. **"Curvature-resolution cost" 𝒞_required(n)** — never defined in the repo.
   No definition makes the central inequality 𝒞_required > 𝒞_available
   well-typed. This work proposes candidate definitions but finds **no proof**
   linking any of them to attack cost.
3. **The reduction** "breaking H ⇒ resolving L(n) independent curvature
   distinctions" — absent. This work attempts and **fails** to construct it (see
   Results §Part V); it states plainly that no such reduction exists yet.
4. **Resource model selection** — never pinned. Supplied in
   `docs/CURVATURE_HEAT_RESOURCE_MODEL.md`.
