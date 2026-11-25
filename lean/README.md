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
- `OTSKeyConsumption.lean`, `LedgerLinearizability.lean`, `KeyReuseCounterexample.lean`, `OTSVerification.lean`, `AuthenticatedBlockBinding.lean`, `OTSConformance.lean`: WaveLock-OTS stateful acceptance (issue #41). Same-host single use of `one_time_key_id`, `public_key_fingerprint` and the commitment table (`ots_material_id`) over arbitrary interleavings with crashes, restarts and side-file loss; one winner per identity; no commitment table authorises two accepted blocks; fail-closed verification; block transcript binding up to explicit collisions; kernel-checked counterexamples for rollback, independent hosts, missing locks, four key-id defects of `b310ea5` and PR #47's key-material reuse (all corrected except rollback and independent hosts); 25 traces checked against both Lean and Python. See [OTS formal verification](../docs/WAVELOCK_OTS_FORMAL_VERIFICATION.md).
- `OTSSignerGuard.lean`, `OTSSignerConformance.lean`: the WaveLock-OTS signing guard (`sign_ots`, host-local key-state registry): no two signatures with the same key id or signing material on one registry over any interleaving with claim failures, crashes, restarts and POSIX power loss; boundaries (`allow_reuse`, second or wiped registry, Windows power loss, migration); 14 traces checked against Lean and Python. See [signer guard](../docs/WAVELOCK_OTS_SIGNER_GUARD.md).
- `RCAAgentState.lean`, `RCATransitions.lean`, `RCASafetyInvariants.lean`, `RCADriftMetrics.lean`, `RCACounterexamples.lean`, `RCAConformance.lean`: the PROPOSED RCA reference safety kernel (issues #42/#43; not preexisting behaviour): typed state, fail-closed safety memory, declarative specification and refinement; an inductive invariant over every interleaving with crashes and restarts; Theorems A–D; authorization strictly before execution and blocking; drift-metric properties and limits; twelve adversarial cases; rollback results; 25 traces. `scripts/RCAModelCheck.lean`: bounded model checks. See [RCA formalization](../docs/RCA_FORMALIZATION.md).
- `scripts/AxiomAudit.lean` (+ `AxiomAuditNegativeControl.lean`): strict audit of every constant of the OTS and RCA modules against `propext`, `Classical.choice`, `Quot.sound`; receipt `axiom_audit_receipt.txt`, compared in CI.
- `SecurityGames.lean`: definitions of collision, second preimage, inversion, adversaries; sufficient but generally unavailable injectivity assumptions; proof that an injection-level result cannot guarantee digest injectivity after noninjective postprocessing; a finite-output pigeonhole theorem; the exact condition needed to lift internal merging to messages.

## Boundaries

The finite-field equations are instantiated over arbitrary fields `F`, not yet an independently verified executable implementation of the Python `ZMod 2147483647` code. The coupled-round definition is faithful to the documented algebra but has **not** been proved to match Python's byte-oriented interpreter. Padding, bytes-to-field packing, block counters, IVs, last-block tag, comparison squeeze, and hash algorithms are NOT formalized. OTS verification and replay persistence are modelled with opaque hashes (see the OTS document); no unforgeability or hash security property is proved. The historical 2025 publication, PDE-256-v0, CC-Core-v0-A, CC-Core-v1-B, and WaveLock-OTS are **distinct** constructions. In particular, a theorem about Candidate B cannot be transplanted to the older release.

No hardness theorem can be obtained merely from finite-state contraction, observed avalanche, absence of attacks in a bounded budget, or a conditional statement with an unproved hardness assumption.

See [claim inventory](../docs/LEAN_CLAIM_MAP.md) for proof statements, assumptions, open obligations and non-claims.
