Falsifiability Conditions for the WaveLock Primitive
Purpose

This document enumerates explicit falsifiability conditions for the WaveLock primitive.

These conditions define what observations would invalidate WaveLock, either partially or entirely.
They are stated deliberately and narrowly to increase trust, not to evade scrutiny.

WaveLock is not a probabilistic security claim.
It is a constraint claim: if the constraint is violated, the primitive is false.

Frozen Reference (Context)

WaveLock is defined by the frozen primitive sentence:

WaveLock is an irreversible state-evolution primitive such that no verifier-accepted commitment exists unless it arises from executing a curvature-bounded evolution whose terminal state ψ★ is reproducible under the declared kernel.

Acceptance is governed solely by the invariant:

𝐼
(
𝜓
⋆
;
𝐾
)
≡
(
Serialize
𝐾
(
𝜓
⋆
)
 is canonical
  
∧
  
Budget
𝐾
(
𝜓
⋆
)
≤
𝐶
)
I(ψ
⋆
;K)≡(Serialize
K
	​

(ψ
⋆
) is canonical∧Budget
K
	​

(ψ
⋆
)≤C)
Principle of Falsifiability

WaveLock is falsifiable if any verifier-accepted commitment can be produced without satisfying the invariant under the declared kernel.

Falsification does not require breaking all embodiments.
Breaking one valid instance is sufficient.

Falsification Class 1: Invariant Violation
Condition

A verifier accepts a commitment 
𝐶
C for which invariant 
𝐼
(
𝜓
⋆
;
𝐾
)
I(ψ
⋆
;K) does not hold.

Examples

Canonical serialization mismatch

Curvature or energy budget exceeded

Kernel metadata altered without rejection

Consequence

WaveLock is invalid, as acceptance no longer enforces the primitive constraint.

Falsification Class 2: Reproducibility Failure
Condition

Two executions of the declared kernel 
𝐾
K from identical ψ₀ produce terminal states ψ★ whose canonical serializations differ, yet both are accepted.

Consequence

Deterministic reproducibility is violated.
WaveLock collapses into a probabilistic or heuristic system and is invalid as a primitive.

Falsification Class 3: Shortcut Generation
Condition

A verifier-accepted ψ★ is produced without executing the declared evolution under 
𝐾
K.

This includes (non-exhaustive):

algebraic construction

learned surrogate prediction

statistical emulation

adjoint or inverse-time reconstruction

partial or parallel trajectory stitching

Consequence

WaveLock is invalid, as irreversibility is violated.

Falsification Class 4: Cost Floor Violation
Condition

A method produces a verifier-accepted ψ★ while demonstrably incurring less total curvature/energy cost than required by the declared kernel 
𝐾
K.

Consequence

The “cannot be cheap” property is violated.
WaveLock reduces to algorithmic hardness and is invalid as a physical constraint.

Falsification Class 5: Approximate Acceptance
Condition

A verifier accepts an approximately reconstructed ψ★ (numerical tolerance, coarse-graining, reduced precision) rather than requiring exact canonical serialization.

Consequence

Acceptance ceases to be exact.
Shortcut approximation becomes admissible.
WaveLock is invalid.

Falsification Class 6: Kernel Ambiguity
Condition

Two materially different kernels are treated as equivalent for verification without explicit declaration or migration rules.

Consequence

Acceptance semantics become ambiguous.
WaveLock loses invariant coherence and is invalid.

Falsification Class 7: Witness Substitution
Condition

A verifier accepts a commitment based on any witness other than ψ★ produced under the declared kernel, including:

hashes alone

signatures alone

metadata alone

external attestations

Consequence

WaveLock degenerates into a conventional cryptographic scheme and is invalid as a new primitive.

Partial vs Total Falsification

Total falsification: any accepted commitment violating Classes 1–3.

Scope-limited falsification: violations limited to a specific kernel or embodiment invalidate that embodiment but not the primitive definition.

This distinction preserves scientific rigor without weakening the claim.

Experimental Falsification Pathways

WaveLock invites falsification via:

adversarial shortcut construction

reproducibility stress tests

curvature/energy accounting audits

independent re-implementations of 
𝐾
K

Failure to falsify under these tests strengthens, but does not prove, the primitive.

Why Explicit Falsifiability Matters

Explicit falsifiability:

prevents interpretive drift

prevents post-hoc claim tightening

increases regulatory trust

distinguishes WaveLock from heuristic cryptographic systems

A primitive that cannot be falsified is not scientific.
WaveLock is intentionally falsifiable.

Conclusion

WaveLock stands or falls on a small set of explicit, testable conditions.

If any verifier-accepted commitment can be produced while violating the invariant, reproducibility, irreversibility, or cost floor, WaveLock is wrong.

If not, WaveLock defines a new class of irreversible verification primitive.

Status:
This document completes the inevitability framework and is intended to be read alongside:

attack_bounds.md

curvature_lower_bounds.md

the frozen primitive definition and invariant