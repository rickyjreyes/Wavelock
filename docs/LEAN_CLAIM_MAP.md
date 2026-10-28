# WaveLock formalization map: issues #37 and #32

**Status:** Partial, rigorously scoped verification track. **Not** a cryptographic security proof.
**Historical publication:** [Release 22](https://doi.org/10.5281/zenodo.19122146), the published PDE-derived one-way-function proposal. Its exact historical assertions must be independently inventoried from the archived release; this mapping does **not** certify their original wording.
**Current experimental protocols:** `WaveLock-PDE-256-v0`, `CC-Core-v0-A`, `CC-Core-v1-B`, and `WaveLock-OTS` are separate and remain frozen.

| Source statement / category | Lean declaration | Required assumptions | Assessment |
| --- | --- | --- | --- |
| Candidate B injection `j_B(u,v)=u(1+γv)` (algebraic identity) | `candidateB_factor` | Field | Formalized |
| Candidate B injective for fixed nonsingular `v` (theorem) | `candidateB_injective` | Field; `1+γv ≠ 0` | Formalized |
| Zero injection on singular hyperplane (theorem) | `candidateB_singular`, `candidateB_zero_iff` | Field; for the former, zero factor | Formalized |
| Opposite-sign wave states and first-round injection separation (theorem) | `waveRound_odd`, `torusLap_odd`, `torusSignPair_injection_difference`, `candidateB_sign_separated` | Odd Laplacian; `2 ≠ 0`; nonzero amplitude | Formalized |
| One-round accumulator separation (conditional theorem) | `accumulator_sign_difference`, `accumulator_sign_separated` | Same starting accumulator/diffused value; nonzero weight; `2 ≠ 0`; nonzero amplitude | Formalized |
| Design A structured zero eigenmode collapse (symbolic theorem) | `waveRound_collapse` | Eigenvalue, squared amplitude, exact balance | Formalized conditionally; no exact 47-state enumeration |
| Global internal wave-round injectivity (false when witness exists) | `waveRound_not_injective_of_collapse` | A nonzero collapse witness and `2 ≠ 0` | Formalized conditional counterexample theorem |
| Deterministic Candidate B coupled round (definition) | `candidateBRound`, `runFrom` | Field; constants from CC-Core-v1-B | Algebraic model, not verified Python parity |
| Replay verifier acceptance (logical correctness) | `replay_accepts_honest`, `replay_rejects_wrong_version` | Same deterministic model/output supplied | Formalized for abstract output, **not** cryptographic authenticity |
| Abstract version/length framing (mathematical model) | `versionFrame_payload_injective`, `versionFrame_separates_versions` | List of natural-number symbols | Formalized; **not** byte encoder |
| Collision/second-preimage/inversion games (definitions) | `Collision`, `SecondPreimage`, `Inverts`, `SecondPreimageGame`, `InversionGame` | Explicit `H`, challenge, adversary | Specified; no security advantage bound |
| Intermediate separation cannot imply digest-level separation (counterexample) | `intermediate_injective_does_not_imply_digest_injective` | Noninjective downstream map | Formalized counterexample |
| Global injectivity fails if message set is larger than digest range (theorem) | `finite_output_collision` | Finite input/output with `n+1>n` | Formalized, **not** an efficient attack |
| Digest collision from equal reachable states (theorem) | `collision_if_same_absorbed_state`, `collision_if_same_evolved_state` | **Two distinct valid messages** reach same intermediate state | Formalized conditional bridge |
| 47-state digest separation / HD ≥ 105 (finite verification result) | None | Frozen dataset and complete replay / reflection | **Open Lean obligation**; documented exhaustive computational finding only |
| Round-0 singular unreachability (analytical claim) | None | Exact byte range, IV, padding/counters | **Open Lean obligation** |
| Normative reference vectors (implementation regression) | None | Byte-exact Python model and certificate | **Open Lean obligation** |
| One-wayness, second-preimage resistance, collision resistance | None | Security parameter family, distribution and attacker model, explicit reduction or quantitative bound | **Open research questions** |
| Sequential work lower bound / no shortcut | None | Restricted computational cost model or complexity assumption | **Open research question** |
| OTS unforgeability and replay protection | None | Full stateful verifier and adversary oracle / key-security model | **Open research question** |

## Distinguishing the different levels

A proof of `candidateB_injective` establishes a **single-coordinate** algebraic fact with fixed `v`. It does not establish injectivity of the coupled round or of the digest. The experimental 47-state result documents a *restricted family*, not a universal proof. A proof that the internal wave map merges arbitrary chosen states does not supply two valid byte-string messages that reach those states. Any reduction must explicitly connect the relevant layers and account for the adversary's computation and observables.

## Remaining work for #37

1. Retrieve the immutable Release 22 PDF, inventory all original mathematical claims and distinguish their historical definitions from later code revisions.
2. Extend the formal model to exact `ZMod 2147483647` constants, normative byte padding, field packing, initialized IVs, per-block counter/tag injection and comparison squeeze; prove a trace relation to executable reference vectors.
3. Obtain a green pinned Lean CI run and keep CI's theorem checking separate from simulation tests.

## Remaining work for #32

1. Define parameterized families, attacker randomness/distributions, oracle interfaces and explicit work/memory accounting **separately** for PDE-256-v0, CC-Core-v1-B and OTS.
2. Bridge wave collisions to **valid-message reachable** collisions, or prove a quantified obstruction under exact protocol bounds. Neither has been done for the full field.
3. Prove a genuine conditional security reduction under auditable assumptions, or give a constructive full-protocol attack. Do not equate first-round binding with final digest binding.
4. Prove a sequential lower bound in a stated computational model, if possible. Deterministic iteration alone is not a complexity lower bound.

This checkpoint must not close #32 or #37 as complete.
