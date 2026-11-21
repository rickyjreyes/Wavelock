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
| Deterministic Candidate B coupled round (definition) | `candidateBRound`, `runFrom`, `NormativeConstants.ccCoreRho0`, `ccCoreRho1` | Field; exact Python hexadecimal and modular constants, with no byte-to-execution translation proof | Corrected algebraic model; **full Python parity still open** |
| Replay verifier acceptance (logical correctness) | `replay_accepts_honest`, `replay_rejects_wrong_version` | Same deterministic model/output supplied | Formalized for abstract output, **not** cryptographic authenticity |
| Abstract version/length framing (mathematical model) | `versionFrame_payload_injective`, `versionFrame_separates_versions` | List of natural-number symbols | Formalized; **not** byte encoder |
| Collision/second-preimage/inversion games (definitions) | `Collision`, `SecondPreimage`, `Inverts`, `SecondPreimageGame`, `InversionGame` | Explicit `H`, challenge, adversary | Specified; no security advantage bound |
| Intermediate separation cannot imply digest-level separation (counterexample) | `intermediate_injective_does_not_imply_digest_injective` | Noninjective downstream map | Formalized counterexample |
| Global injectivity fails if message set is larger than digest range (theorem) | `finite_output_collision` | Finite input/output with `n+1>n` | Formalized, **not** an efficient attack |
| Digest collision from equal reachable states (theorem) | `collision_if_same_absorbed_state`, `collision_if_same_evolved_state` | **Two distinct valid messages** reach same intermediate state | Formalized conditional bridge |
| 47-state digest separation / HD ≥ 105 (finite verification result) | None | Frozen dataset and complete replay / reflection | **Open Lean obligation**; documented exhaustive computational finding only |
| First-block singular-value exclusion for all 256 wave sites (restricted theorem) | `firstBlockRate_below_singular`, `firstAbsorbedWave_not_singular`, `absorbWaveBlock_first_matches_initial` | Bounded IV < 375; 3 byte rate elements; first block is not last | Full 256-site **pre-evolution** wave state formalized; later rounds and messages remain open |
| Exact two-digit counter arithmetic (theorem) | `counterDigits_injective` | Natural-number digits, prior to field reduction | Formalized, with field encoding bound still open |
| At least two padded blocks; no first-block finalization tag (theorem) | `paddedBlockCount_at_least_two`, `firstBlock_not_last` | Normative marker, zero-fill and length-block rule | Formalized integer-level count |
| Normative reference vectors (implementation regression) | None | Byte-exact Python model and certificate | **Open Lean obligation** |
| One-wayness, second-preimage resistance, collision resistance | None | Security parameter family, distribution and attacker model, explicit reduction or quantitative bound | **Open research questions** |
| Sequential work lower bound / no shortcut | None | Restricted computational cost model or complexity assumption | **Open research question** |
| WaveLock-OTS same-host single use: no accepted transition consumes an already-consumed `one_time_key_id`, fingerprint or commitment table, any interleaving, crashes, restarts, side-file loss (theorem) | `inv_run`, `chain_single_use`, `consume_fresh`, `at_most_one_acceptance`, `wavelock_ots_single_use`, `wavelock_ots_material_single_use`, `wavelock_ots_fresh_triple` | Trusted durable single-host store; `AcceptSound`; `LosslessRecon` ∨ no side-file loss (both proved for the Python instance) | Formalized; see [OTS formal verification](WAVELOCK_OTS_FORMAL_VERIFICATION.md) |
| One commitment table authorises at most one accepted block (theorem) | `distinct_material_distinct_tables`, `materialId_eq_of_decode_eq`, `relabelled_replays_keep_material`, `psi_relabel_verifies` | None for this direction; the converse needs Merkle-root collision resistance (`material_binding`) | Formalized; model ↔ `ots_material_id` by conformance traces |
| OTS acceptance never makes the node's own files unloadable (liveness theorem) | `loadable_run`, `wavelock_ots_no_self_dos` | No torn writes; `RecordAdmissible`, `ReconTotal` | Formalized; false at `b310ea5` (`legacy_*_key_id_*`), corrected |
| `verify_ots` fails closed and binds replay identities to the verified key (theorem) | `verifyOTS_sound`, `verifyOTS_identity`, `fingerprint_binding` | Opaque hashes; uniqueness only modulo an explicit fingerprint `Collision` | Formalized |
| OTS block signature binds the received block (theorem) | `verifyOTSBlock_binds`, `block_binding`, `block_relabel_needs_second_message` | Opaque transcript digest; conclusions are disjunctions with `Collision` / `TableUsedTwice` | Formalized, conditional |
| Global one-time use; rollback or cross-host safety; secret values recommitted under a new table (counterexample / boundary) | `rollback_reaccepts_identity`, `independent_nodes_both_accept`; PR #47 `copied_key_material_accepted_twice` (closed by `copied_key_material_rejected`) | — | **Counterexamples**; not provided by the implementation |
| WaveLock-OTS signing guard: no two signatures leave `sign_ots` with the same key id or the same signing material `(params_hash, seed)`, any interleaving, claim failures, crashes, restarts, POSIX power loss (theorem) | `signer_single_use`, `durable_markers_monotone` | One registry directory; no `allow_reuse`; registry not deleted; POSIX directory `fsync` | Formalized; 14 traces against `sign_ots`; see [signer guard](WAVELOCK_OTS_SIGNER_GUARD.md) |
| No two signatures reveal the same secret table through `sign_ots` (conditional theorem) | `signer_values_distinct` | Injective slice derivation (SHAKE-256 collision resistance) | Formalized, conditional |
| Signer boundaries: `allow_reuse`, a second registry, a wiped registry, Windows power loss, material signed before the change; one seed under two key ids at `c3cb0d1` (counterexamples) | `allow_reuse_bypasses_guard`, `second_registry_bypasses_guard`, `wiped_registry_forgets`, `windows_power_loss_forgets`, `legacy_material_signs_once_more`, `c3cb0d1_material_reused` | — | **Counterexamples**; the last is corrected (`current_material_refused`) |
| OTS unforgeability (EUF-CMA, one-time) | None | Adversary oracle model, explicit reduction to SHAKE256 properties | **Open research question** |

## CC-Core constant mismatch correction, October 29, 2026

The normative Python expressions have **not** changed. The earlier Lean model and
several written decimal annotations used incorrect conversions:

| Parameter | Normative Python expression | Correct decimal | Previously used incorrect decimal |
| --- | --- | ---: | ---: |
| RHO0 | `0x57434330` | 1464025904 | 1463898160 |
| RHO1 | `2654435761 % 2147483647` | 506952114 | 507468114 |
| Candidate B D_TAG | `0x57434332` | 1464025906 | 1463898162 |
| Candidate A D_TAG | `0x57434331` | 1464025905 | 1463898161 |

New `NormativeConstants.lean` certifies the exact integer equalities
and `gcd(RHO1, p-1)=186`. The previously documented claim that
the affine additive `rho_t` schedule has period `p-1` because its
slope is coprime to `p-1` was incorrect. Its actual standalone period
modulo `p` is `p`, because `p` is prime and RHO1 is nonzero.

`Protocol.lean` now draws the round constant from those exact expressions;
`BlockAbsorption.lean` now uses the correct finalization tag.
Python regression tests lock down the literal values and schedule.

**Scope:** correcting literal parameters is necessary but insufficient for
source-level verification. Historic prior CI runs with the wrong Lean
parameters do not prove the round model matches the frozen Python program.
The corrected source must pass full Lean and Python CI at its own final
commit. Full message processing, multi-block replay and comparison-squeeze
correspondence remain open.

## Distinguishing the different levels

A proof of `candidateB_injective` establishes a **single-coordinate** algebraic fact with fixed `v`. It does not establish injectivity of the coupled round or of the digest. The experimental 47-state result documents a *restricted family*, not a universal proof. A proof that the internal wave map merges arbitrary chosen states does not supply two valid byte-string messages that reach those states. Any reduction must explicitly connect the relevant layers and account for the adversary's computation and observables.

## Remaining work for #37

1. Retrieve the immutable Release 22 PDF, inventory all original mathematical claims and distinguish their historical definitions from later code revisions.
2. Extend the formal model to exact `ZMod 2147483647` constants and full byte-oriented padding, initialized IVs, counter/tag field injection and comparison squeeze (building on the proved packing and block-count lemmas); prove a trace relation to executable reference vectors.
3. Obtain a green pinned Lean CI run and keep CI's theorem checking separate from simulation tests.

## Remaining work for #32

1. Define parameterized families, attacker randomness/distributions, oracle interfaces and explicit work/memory accounting **separately** for PDE-256-v0, CC-Core-v1-B and OTS.
2. Bridge wave collisions to **valid-message reachable** collisions, or prove a quantified obstruction under exact protocol bounds. Neither has been done for the full field.
3. Prove a genuine conditional security reduction under auditable assumptions, or give a constructive full-protocol attack. Do not equate first-round binding with final digest binding.
4. Prove a sequential lower bound in a stated computational model, if possible. Deterministic iteration alone is not a complexity lower bound.

This checkpoint must not close #32 or #37 as complete.
