# WaveLock Lean 4 proof project

This is an **experimental formal verification track** for [#37](https://github.com/rickyjreyes/Wavelock/issues/37) and [#32](https://github.com/rickyjreyes/Wavelock/issues/32). It uses the same Lean/Mathlib version as `rickyjreyes/math`: Lean 4.23.0 and Mathlib v4.23.0. It does **not** assert that WaveLock is a proven one-way function, signature scheme, collision-resistant hash, or sequential-work primitive.

```bash
cd lean
lake update
lake exe cache get
lake build
```

## What is proved

- `Algebra.lean`: injectivity of Candidate B's per-cell injection away from its zero multiplier; exact singular collapse; the odd polynomial wave equation under an odd Laplacian; nonzero sign-pair separation at the injection and first-round accumulator when its position weight is nonzero; a symbolic eigenmode collapse criterion; conditional noninjectivity of the internal wave round when a nonzero collapse witness exists; and the actual 16x16 periodic neighbor Laplacian's oddness.
- `Encoding.lean`: injective three-byte packing, the exact 24-bit packed range, initial rate-cell singular-value exclusion, no modular wrap, the injective two-digit block-counter arithmetic, and the fact that the first padded block is never the final block.
- `Protocol.lean`: a symbolic coupled 16x16 Candidate B round using the stated coefficients, global round index, position weight and accumulator polynomial; deterministic recursion; abstract replay acceptance and wrong-version rejection; abstract version/length framing lemmas.
- `SecurityGames.lean`: definitions of collision, second preimage, inversion, adversaries; sufficient but generally unavailable injectivity assumptions; proof that an injection-level result cannot guarantee digest injectivity after noninjective postprocessing; a finite-output pigeonhole theorem; the exact condition needed to lift internal merging to messages.

## Boundaries

The finite-field equations are instantiated over arbitrary fields `F`, not yet an independently verified executable implementation of the Python `ZMod 2147483647` code. The coupled-round definition is faithful to the documented algebra but has **not** been proved to match Python's byte-oriented interpreter. Padding, bytes-to-field packing, block counters, IVs, last-block tag, comparison squeeze, signatures, OTS replay persistence, and hash algorithms are NOT formalized. The historical 2025 publication, PDE-256-v0, CC-Core-v0-A, CC-Core-v1-B, and WaveLock-OTS are **distinct** constructions. In particular, a theorem about Candidate B cannot be transplanted to the older release.

No hardness theorem can be obtained merely from finite-state contraction, observed avalanche, absence of attacks in a bounded budget, or a conditional statement with an unproved hardness assumption.

See [claim inventory](../docs/LEAN_CLAIM_MAP.md) for proof statements, assumptions, open obligations and non-claims.
