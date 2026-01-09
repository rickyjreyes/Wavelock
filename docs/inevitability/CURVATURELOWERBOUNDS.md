Curvature and Energy Lower-Bound Arguments for the WaveLock Primitive
Scope and Intent

This document provides structural lower-bound arguments showing that any method capable of producing a verifier-accepted terminal state ψ★ must incur a nonzero, irreducible cost under the declared kernel 
𝐾
K.

These are not tight bounds and are not formal complexity proofs.
They establish that WaveLock verification is not merely computationally expensive, but physically constrained by curvature dissipation and entropy production.

The arguments are anchored to the frozen invariant:

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
Key Claim (Non-Negotiable)

Any trajectory that reaches a verifier-accepted terminal state ψ★ must dissipate a strictly positive minimum curvature/energy budget under the declared kernel 
𝐾
K.

This minimum cost cannot be bypassed, amortized away, or parallelized into zero.

Definitions

Curvature Functional 
𝐸
[
𝜓
]
E[ψ]:
A non-negative scalar functional measuring curvature, gradient energy, entropy, or feedback cost accumulated during evolution. Examples include (non-limiting):

integrated gradient energy

curvature feedback accumulation

entropy production

Lyapunov descent

Declared Kernel 
𝐾
K:
A fixed evolution rule under which both forward execution and verification are defined.

Evolution Cost:
The total accumulated curvature/energy dissipation required to transform ψ₀ into ψ★ under 
𝐾
K.

Lower-Bound Argument 1: Lyapunov Descent Implies Nonzero Dissipation
Statement

If the evolution admits a Lyapunov functional 
𝐿
[
𝜓
]
L[ψ] such that:

𝐿
[
𝜓
𝑡
+
1
]
<
𝐿
[
𝜓
𝑡
]
for forward evolution
,
L[ψ
t+1
	​

]<L[ψ
t
	​

]for forward evolution,

then any trajectory reaching ψ★ must dissipate:

𝐸
min
≥
𝐿
[
𝜓
0
]
−
𝐿
[
𝜓
⋆
]
>
0.
E
min
	​

≥L[ψ
0
	​

]−L[ψ
⋆
]>0.
Implication

ψ★ is separated from ψ₀ by a finite Lyapunov gap.

No valid trajectory can traverse this gap at zero cost.

Consequence for Shortcuts

Any shortcut that claims to reach ψ★ with lower cost must either:

violate monotonicity, or

violate reproducibility under 
𝐾
K,

and thus fails 
𝐼
I.

Lower-Bound Argument 2: Curvature Accumulation Is Global and Path-Dependent
Statement

The declared evolution accumulates curvature globally across the domain. The total curvature budget satisfies:

𝐸
[
𝜓
⋆
]
=
∫
trajectory
𝜅
(
𝜓
(
𝑡
)
)
 
𝑑
𝑡
E[ψ
⋆
]=∫
trajectory
	​

κ(ψ(t))dt

where 
𝜅
(
𝜓
)
κ(ψ) is non-negative under 
𝐾
K.

Implication

Curvature cost depends on the entire trajectory, not only endpoints.

Partial or shortcut paths cannot reconstruct global accumulation.

Consequence for Parallelization

Parallel or decomposed execution may accelerate time, but cannot reduce total accumulated curvature below a fixed lower bound.

Thus:

𝐸
parallel
≥
𝐸
serial
≥
𝐸
min
.
E
parallel
	​

≥E
serial
	​

≥E
min
	​

.
Lower-Bound Argument 3: Entropy Production Prevents Reversible Shortcuts
Statement

The evolution increases entropy or coarse-grained disorder monotonically:

Δ
𝑆
≥
0
ΔS≥0

for all forward steps under 
𝐾
K.

Implication

Information about ψ₀ is irreversibly discarded.

Reverse or shortcut trajectories must recreate lost information.

Consequence

Recreating discarded information requires energy at least proportional to entropy loss:

𝐸
min
≳
𝑘
𝐵
Δ
𝑆
.
E
min
	​

≳k
B
	​

ΔS.

Any shortcut claiming lower cost violates thermodynamic consistency or determinism under 
𝐾
K.

Lower-Bound Argument 4: Exact Reproducibility Enforces Cost Floors
Statement

Verification requires byte-exact reproduction of:

Serialize
𝐾
(
𝜓
⋆
)
.
Serialize
K
	​

(ψ
⋆
).
Implication

Approximate trajectories are insufficient.

Error tolerance is zero.

Consequence

Any method that avoids full evolution must still reproduce ψ★ exactly, which requires satisfying all curvature constraints encountered along the original trajectory.

Thus, exactness forces cost.

Synthesis: “Cannot Be Cheap”

Combining the above:

Lyapunov descent ⇒ finite dissipation

Curvature accumulation ⇒ path dependence

Entropy production ⇒ irreversibility

Exact serialization ⇒ no approximation slack

We obtain the structural result:

There exists a kernel-dependent constant 
𝐸
min
>
0
E
min
	​

>0 such that no verifier-accepted ψ★ can be produced with total cost less than 
𝐸
min
E
min
	​

.

This holds regardless of:

algorithmic strategy

hardware parallelism

statistical or learned approximation

adversarial foreknowledge

Relation to Verification Cost Asymmetry

Verification checks:

invariant 
𝐼
I,

reproducibility under 
𝐾
K,

budget satisfaction.

These checks are bounded and do not scale with 
𝐸
min
E
min
	​

.

Thus:

Cost
verify
≪
Cost
generate
≤
𝐸
min
.
Cost
verify
	​

≪Cost
generate
	​

≤E
min
	​

.

This asymmetry is structural, not contingent.

Falsifiability Statement

WaveLock is falsified if any method produces a verifier-accepted ψ★ while demonstrably expending less than the declared curvature/energy budget required by 
𝐾
K.

Such a result would imply violation of at least one of:

Lyapunov monotonicity,

curvature accumulation,

entropy production,

deterministic reproducibility.

Conclusion

These arguments establish that WaveLock’s irreversibility is grounded in nonzero physical cost, not merely algorithmic hardness.

WaveLock commitments are therefore not just expensive to compute — they are physically constrained to incur irreducible curvature and energy dissipation.

Status:
This document provides structural lower-bound justification for WaveLock’s “cannot be cheap” property and is intended to be read alongside the frozen primitive definition and attack-bounding arguments.