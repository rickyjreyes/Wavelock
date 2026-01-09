# Attack-Bounding Arguments for the WaveLock Primitive

## Scope and Intent

This document enumerates classes of shortcut attacks that might plausibly be attempted against the WaveLock primitive and provides structural reasons why each class fails to produce a **verifier-accepted** commitment without executing the declared curvature-bounded evolution.

These are **attack-bounding arguments**, not formal proofs.  
They are intended to demonstrate where and why inversion shortcuts fail under the frozen WaveLock invariant:

\[
\mathcal{I}(\psi^\star;K)\ \equiv\ 
\bigl(
\mathrm{Serialize}_K(\psi^\star)\ \text{is canonical}
\ \wedge\
\mathrm{Budget}_K(\psi^\star)\le C
\bigr).
\]

A commitment is accepted only if the terminal state **ψ★** is reproducible under the declared kernel **K** and satisfies **\(\mathcal{I}\)**.

---

## Threat Model

An adversary is assumed to have:

- Full knowledge of the declared kernel **K**
- Full knowledge of serialization and verification rules
- Access to arbitrary compute resources
- Ability to use classical, statistical, learned, and parallel methods

The adversary’s goal is to produce a **verifier-accepted** commitment **without executing** the declared evolution to **ψ★**.

---

## Attack Class 1: Adjoint / Reverse-Time PDE Inversion

### Description
Attempt to recover **ψ₀** or **ψ★** by integrating the evolution backward using adjoint methods, reverse-time PDEs, or sensitivity analysis.

### Why It Appears Plausible
Many PDE systems admit adjoint formulations for optimization, control, or inverse problems.

### Failure Mode
The WaveLock evolution is irreversible by construction under the declared kernel:

- The evolution is contractive in forward time and expansive under reversal.
- Information is destroyed via curvature dissipation and entropy production.
- Reverse-time integration is ill-posed: small perturbations amplify exponentially.

### Invariant Violation
Any reverse-constructed state fails at least one of:

- canonical serialization reproducibility
- curvature/energy budget bound

Thus, adjoint inversion cannot satisfy \(\mathcal{I}(\psi^\star;K)\).

---

## Attack Class 2: Gradient Backpropagation / Differentiable Inversion

### Description
Treat the evolution as a differentiable map and use gradient descent or backpropagation to recover **ψ★** or a surrogate state.

### Why It Appears Plausible
Modern ML systems routinely invert or approximate complex nonlinear mappings using gradients.

### Failure Mode
WaveLock’s evolution breaks gradient-based inversion because:

- Gradients are not conserved through curvature-regulated dissipation.
- Backpropagated gradients collapse under entropy-increasing steps.
- Local gradient information does not encode global trajectory constraints.

Gradient methods may converge to approximate states but not to a state whose canonical serialization is reproducible under **K**.

### Invariant Violation
Approximate states fail deterministic reproducibility and therefore fail \(\mathcal{I}\).

---

## Attack Class 3: Coarse-Graining and Reduced-Order Approximation

### Description
Approximate **ψ★** using spectral truncation, reduced bases, low-rank decompositions, or coarse-grained representations.

### Why It Appears Plausible
Many physical systems admit reduced-order models that preserve macroscopic behavior.

### Failure Mode
WaveLock verification is exact, not statistical:

- Canonical serialization is byte-level exact.
- Verification tolerates no approximation error.
- Coarse-grained states do not map to identical serialized bytes.

Even infinitesimal deviation in **ψ★** invalidates the commitment.

### Invariant Violation
Reduced representations fail canonical serialization and thus fail \(\mathcal{I}\).

---

## Attack Class 4: Parallel Decomposition of Evolution

### Description
Attempt to partition the evolution spatially or temporally and compute **ψ★** via parallel sub-evolutions.

### Why It Appears Plausible
Many PDE solvers use domain decomposition and parallel execution.

### Failure Mode
While forward execution may be parallelized, shortcut execution cannot:

- Curvature budgets are global, not separable.
- Boundary coupling enforces cross-domain dependencies.
- Partial trajectories cannot be recombined without violating curvature constraints.

Any decomposition that avoids full evolution fails to preserve the global invariant.

### Invariant Violation
Recombined states exceed curvature budgets or fail reproducibility, violating \(\mathcal{I}\).

---

## Attack Class 5: Learned Surrogates and Statistical Emulation

### Description
Train a neural or statistical model to predict **ψ★** directly from **ψ₀** or from partial evolution data.

### Why It Appears Plausible
Surrogate models can approximate expensive simulations in many domains.

### Failure Mode
Learned models fail because:

- They approximate distributions, not exact trajectories.
- They cannot guarantee deterministic reproducibility under **K**.
- They do not encode curvature budget enforcement.

Even a perfect predictor in expectation cannot satisfy exact serialization equality.

### Invariant Violation
Surrogates produce non-reproducible **ψ★** and thus fail \(\mathcal{I}\).

---

## Summary of Closed Attack Classes

| Attack Class | Fails Because |
|---|---|
| Adjoint inversion | Ill-posed reverse evolution |
| Gradient backprop | Entropy destroys gradient information |
| Coarse-graining | Exact serialization required |
| Parallel shortcut | Global curvature budgets |
| Learned surrogate | No exact reproducibility |

---

## Conclusion

All examined shortcut classes fail not because they are inefficient, but because they violate the acceptance invariant.

A verifier-accepted commitment requires execution of the declared curvature-bounded evolution. Any shortcut that avoids this execution necessarily violates \(\mathcal{I}(\psi^\star;K)\) and is rejected.

These arguments establish that WaveLock is not merely hard to compute, but structurally resistant to shortcut inversion under its acceptance rules.

---

## Falsifiability Statement

If any method produces a verifier-accepted commitment without executing the declared evolution under **K** while satisfying \(\mathcal{I}\), the WaveLock primitive is invalid.

---

## Status

This document provides attack-bounding justification for irreversibility claims and is intended to be read alongside the frozen primitive definition and invariant.
