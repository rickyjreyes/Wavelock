# RCA agent state, safety memory and pre-action intervention: formalization report

**Issues:** #42 (semantic core), #43 (safety invariants, drift, intervention).

**Baseline:** `master` at `c3cb0d1`, after merged PR #47.

**Companion:** the WaveLock-OTS signing-guard hardening ([signer guard](WAVELOCK_OTS_SIGNER_GUARD.md)),
submitted as a separate pull request.

**Toolchain:** Lean 4.23.0, Mathlib v4.23.0.

## Classifications used

| Label | Meaning |
|---|---|
| `LEAN_PROVED` | Kernel-checked theorem about the stated model. |
| `IMPLEMENTATION_CONFORMANCE_TESTED` | The same finite traces are checked in Lean by `decide` and against the Python code. |
| `CONDITIONAL` | Proved under a named hypothesis that is an assumption, not a fact about the code. |
| `COUNTEREXAMPLE` | A kernel-checked reachable execution violating a property. |
| `REFUTED` | A claim shown false. |
| `PROPOSED_MODEL` | A model of new functionality introduced by this work. |
| `OPEN` | Not established. |

## 1. Repository existence audit

The whole repository at `c3cb0d1` was searched: Python, docs, tests, tools, scripts and audit.
The search covered agent state, memory, policy decisions, drift detection, action authorization
and intervention.

| Component | Python implementation at `c3cb0d1` | Persistent state | Tests | Status |
|---|---|---|---|---|
| Agent state | None. The word "RCA" occurs only as a patent docket (`PATENT_NOTICE.md:35`, `REYES-RCA-2025`). Nearest: the OTS signing process state, the secret-key dict in `wavelock/crypto/wavelock_ots.py:sign_ots`. | — | — | **Missing** |
| Safety memory | None for agents. For one-time keys: the signer registry (`wavelock_ots._claim_one_time_key`, `WAVELOCK_OTS_STATE_DIR`), the advisory `used` flag (`keyfiles.mark_secret_used`), and the verifier ledger and chain (`ots_ledger.PersistentOTSReplayLedger`). | OTS only | `tests/test_ots_redteam.py::test_secure_copied_key_cannot_sign_twice`, `tests/test_bounty_durability.py::test_process_exit_after_signer_claim_burns_key`, `tests/test_ots_formal_conformance.py` | **Missing** (agents); implemented (OTS keys) |
| State transitions | None for agents. For OTS: `sign_ots`, `PersistentOTSReplayLedger.accept`, `network/server.py:try_accept_block`. | OTS only | `tests/test_supported_workflow.py`, `tests/test_ots_consensus.py` | **Missing** (agents) |
| Drift measurement | None. `audit/run_bounty_contract.py:191`, Critical 7 FAIL: "No behavior-wide drift oracle, observation contract, or detection verifier exists". The `tests/scientific/*::test_drift` tests measure commitment sensitivity to perturbed ψ (avalanche), not behaviour. | no | as named | **Missing** |
| Authorization guard | None for agent actions. For OTS: the `sign_ots` registry claim, `ots_cli.cmd_sign` (`used`), `chain/cli._mine` (`is_consumed`) and `try_accept_block`. The node policy `chain/config.py:Config.require_ots` is static configuration, not persisted safety memory. | OTS only | `tests/test_ots_security.py`, `tests/test_supported_workflow.py::test_real_config_loader_enforces_ots` | **Missing** (agents); OTS gate implemented, defect found (§9) |
| Intervention execution | None for agents. For OTS: refusal by exception (`OTSKeyReuseError`, CLI exit 2) and block rejection. | — | as above | **Missing** (agents) |
| Curvature/coherence feedback (named in #42) | The PDE evolution in `chain/WaveLock.py` (`CurvatureKeyPair.evolve`, `_curvature_functional`, `classify_wcc_run`) and `wavelock_ots.evolve_psi_star`; the PoV reward in `chain/pov.py`. | no | `tests/scientific/*` | Implemented as commitment dynamics. **No feedback into any decision.** |

**Critical existence test.** RCA safety memory and a pre-action intervention mechanism for agents
**do not exist** in this repository. The gap is architectural, not mathematical: there is no agent
loop, no observation channel and no action path to reason about.

The only preexisting mechanism of the requested shape is the WaveLock-OTS signing guard. It is
persistent memory of claimed keys, consulted before the unsafe action (revealing slices twice). It
is formalized and corrected in the companion pull request (§9).

## 2. Implemented versus proposed functionality

| Category | Content | Label |
|---|---|---|
| Existing implementation | The WaveLock-OTS signing guard and verifier ledger. | `LEAN_PROVED` model + `IMPLEMENTATION_CONFORMANCE_TESTED` (companion PR; PR #47) |
| Proposed formal abstraction | Typed agent state, safety memory, declarative safety specification, transition system and disciplines (`RCAAgentState`, `RCATransitions`). | `PROPOSED_MODEL` |
| New functionality | `wavelock/rca_reference`, a minimal reference safety kernel (`STATUS = "PROPOSED_REFERENCE"`): hash-chained fail-closed memory, guarded execution, integer drift metrics, monotone policy, operator-credential reset, optional anchor. It is not wired into the chain, network or OTS code and changes none of their behaviour. | `PROPOSED_MODEL`, conformance-tested |

Every theorem below about the RCA kernel concerns this proposed reference. None of them verifies
preexisting production behaviour, because none existed.

## 3. Typed state and transition specification

Lean `RCAAgentState`, `RCATransitions`; Python `wavelock/rca_reference/memory.py`, `kernel.py`.
The prompt's illustrative state $S_t = (X_t, M_t, O_t, P_t, D_t, I_t)$ maps as follows.

| Component | Lean | Python |
|---|---|---|
| Agent id, context | `Rec.genesis agent baseline policy`, `View.agent` | genesis record `agent`; `GuardedAgent.directory` |
| Operational state $X_t$ | `Phase`: `idle`, `dead`, `holding pre v`, `committed pre v k`; lock owner | one `GuardedAgent` instance; `_lock` held in `_transact` |
| Persistent memory $M_t$ | `Disk` (`log : List Rec`, `torn`) | `safety-memory.jsonl`, hash-chained canonical JSON lines |
| Observations $O_t$ | `Rec.observe seq values` (`List ℤ`) | `observe(seq, values)`: integer vectors, `seq` strictly increasing |
| Policy $P_t$ | `Policy` (`version`, `threshold`, `prohibited`) | `Policy`, `successor` |
| Deviation $D_t$ | `View.lastDev = l1 values baseline` | `View.last_dev`, `drift.l1` |
| Intervention $I_t$ | `View.locked` (sticky latch) | `View.locked` |
| Authorization | `View.authorizes`, `admissible` | `View.authorizes`, `memory.admissible` |
| Execution, recovery | events `submit` / `effect` / `crash` / `restart` | `_submit` (write-ahead, then effect), new instance |

Events: `provision`, `begin`, `submit q r o`, `effect q`, `crash`, `restart`, `issue`, and
`anchorNow`. The out-of-boundary events `directEffect`, `rollback` and `tamper` stay in the model
so that every hypothesis is explicit.

Each event is classified by whether it is observable, durable or reversible (table in
`RCATransitions`). Pure parts (`admissible`, `next`, `load`, `specOf`) are functions; I/O
outcomes are explicit event parameters:
- `ok`;
- `fail`, with nothing written, a complete line, or a torn line;
- `crash`, with the same three effects.

No determinism is assumed across processes: every interleaving is a different trace.

## 4. Persistent safety-memory semantics

**Records.** The record kinds are `genesis`, `observe`, `policy`, `execute` and `reset`.

**Loading.** `load` folds the records and fails closed (`MemoryCorrupt`) on any of:
- a missing file or a torn tail;
- non-canonical JSON or a broken hash chain;
- an inadmissible record;
- a log that does not extend the externally supplied anchor.

**Writing.** The kernel appends a record only if it is admissible for the view loaded under the
lock, so it never writes a memory its own loader rejects. A durable append uses `write` + `fsync`.

**Monotonicity.** The memory is append-only inside the boundary:
- `log_prefix_step`: no in-boundary transition, in any discipline, deletes, compacts or rewrites
  a record (`LEAN_PROVED`);
- $M_t \subseteq M_{t+1}$ therefore holds for the record set.

The *derived* latch is not monotone: a reset record clears it. That transition is modelled, not
hidden. A reset needs an operator credential bound to the exact prefix, made while the last fresh
deviation is below the threshold.

**Recovery contract** (Theorem C):
- a restarted process recovers exactly the fold of the durable log;
- every acknowledged record is in that log, in order;
- records written but not acknowledged (write errors, crashes) may or may not be recovered.

A crash between a durable `execute` intent and its effect leaves the intent without the effect
(at-most-once).

**Logical consistency versus durability.** The theorems assume honest `fsync`. Storage that
loses acknowledged writes, or is restored to an older copy, is the out-of-boundary `rollback`;
§8 case 1 shows what it does.

## 5. Authorization and action-execution semantics

`GuardedAgent.request(action, effect)` runs entirely inside one critical section:

1. Take the memory lock.
2. Reload and validate the whole memory.
3. Evaluate `admissible` (`View.authorizes`) on that fresh view.
4. Append the `execute` intent with `fsync`.
5. Invoke the effect.
6. Release the lock.

No authorization artifact outlives the call, so there is nothing stale to reuse. In Lean the
decision with the durable intent is `submit` and the effect is a separate `effect` event; the lock
is held between them.

**Declarative specification.** `specOf` folds the same records under the *intended* semantics and
trusts none of the kernel's checks:
- prohibitions accumulate (union), and the threshold only decreases (min);
- stale or out-of-order observations carry no information;
- the hazard is sticky, and only a reset with a credential issued for exactly its prefix, at a
  resolved deviation, clears it.

`specPermits L k` is the safety condition. `load_refines_spec` proves that on every log the loader
accepts, the operational view and the specification agree (`LEAN_PROVED`). The guard therefore
decides exactly the specification (`authorizes_iff_spec`).

**Trust boundary**, stated and not enforced:
- effects are reachable only through `request`;
- memory is not rolled back, deleted or edited, except as detected by an anchor;
- `fsync` is honest;
- the operator key is not available to the agent;
- observations faithfully report behaviour.

## 6. Main Lean invariants

All theorems below are in `RCASafetyInvariants` unless noted. Every one passes the strict audit
(§11). "Reference" means `Discipline.reference`; "in-boundary" means every event of the trace
satisfies `Event.inBoundary`.

| Theorem | Hypotheses | Conclusion | Implementation mapping | Limitations / counterexample |
|---|---|---|---|---|
| `inv_run` | in-boundary finite trace from `Sys.init`, reference | `Inv`: validity, anchor prefix, lock ownership, fresh views, committed intents, effect safety, acknowledged ⊆ durable, credential binding | `GuardedAgent` | All interleavings, crashes, write failures, restarts. Out-of-boundary events: §8 |
| `state_validity` (A) | as above | an intact memory loads; views and committed intents are folds of the durable log | `_load`, `_submit` | Torn tail: every process fails closed (liveness, not safety) |
| `safety_memory_preserved` (B), `log_prefix_step` | as above; `log_prefix_step` holds for every discipline | acknowledged records are durable, in order; append-only | `_write_line` + `fsync` | Honest `fsync`; volatile memory and rollback: cases 1, 10 |
| `restart_consistency` (C) | `Inv`, a crashed process, intact non-empty memory | recovered view = fold of the durable log ⊇ acknowledged records; latch = specification hazard | new `GuardedAgent` | — |
| `replay_is_inert` (D), `credential_bound` | unique credential names; a loadable log | a replayed record is refused, except a replayed action request (no state change, judged afresh); every durable reset carries a credential issued for exactly its prefix | `seq` check, `successor`, `operator_credential` | Unbound credentials and missing sequence checks: model-check witnesses |
| `effect_step_source`, `step_enters_committed` | **every** discipline | effects arise only from a committed intent or `directEffect`; intents are committed only by a `submit` whose guard admitted on the process's view | `_submit` ordering | `directEffect`: case 5 |
| **`authorized_before_execution`** | in-boundary trace, reference, step `j` changes the effects | ∃ i < j: step `i` is `submit q (execute k) ok` with the guard admitting `k` on the view folded from the then-current durable memory `pre` (so `specPermits pre k`); step `j` is `effect q`; for all i < m ≤ j, `q` stays committed and the durable log is `pre ++ [execute k]` | the code order lock → reload → guard → `fsync` → effect → unlock (`test_effect_runs_only_after_durable_intent`) | — |
| **`pre_action_intervention`** | `Inv`; `specPermits S.log k = false` | the `execute k` request is refused: nothing written or committed | `_submit` returns `False` | — |
| **`effect_blocked`** | `Inv`; `specPermits S.log k = false`; in-boundary event | no new effect of kind `k` | — | — |
| `safe_run` | in-boundary trace, reference | `Safe`: every effect was permitted by the specification on the durable memory when it ran, and acknowledged ⊆ durable | — | §8 |

`authorized_before_execution` is the central result in the requested form:

$$\operatorname{Executed}(T,a) \Rightarrow \operatorname{AuthorizedBeforeExecution}(T,a).$$

The authorization it names is a *decision event* at an earlier step, evaluated on the pre-state's
durable memory. It is not inferred from the execution record. The blocking direction,
$\neg\operatorname{Authorized}(S,a) \Rightarrow \neg\operatorname{ExecuteTransition}(S,a)$, is
`pre_action_intervention` together with `effect_blocked`.

**Levels established.**

1. Model-level safety: `LEAN_PROVED` for all finite in-boundary traces.
2. Correspondence with the Python reference kernel: `IMPLEMENTATION_CONFORMANCE_TESTED` (25
   traces, §10). A universal refinement proof is `OPEN`.
3. Protection of an existing WaveLock production execution path: **none**. No production agent
   path exists, and the reference kernel is not wired into one.
4. General autonomous-agent safety: **not established**.

## 7. Drift-metric results

Lean `RCADriftMetrics`; Python `wavelock/rca_reference/drift.py`. All inputs are integer vectors
in caller-defined units, so Lean and Python compute identical numbers.

| Result | Inputs and assumptions | Status |
|---|---|---|
| `l1_comm`, `l1_self`, `l1_triangle`, `l1_eq_zero_iff`; non-negativity by type (ℕ) | Vectors of one dimension | `LEAN_PROVED` |
| `path_snoc`, `dev_le_path`: accumulated drift grows, and final deviation ≤ initial deviation + path | Same dimension as the baseline | `LEAN_PROVED` |
| `path_drift_without_deviation`: accumulated drift is not a hazard measure (oscillation) | — | `COUNTEREXAMPLE`; the correct statement is the upper bound |
| `anchoredDetects_mono` (lower threshold ⇒ more detections); `tighten_restricts` (a successor policy never authorizes anything new) | — | `LEAN_PROVED` |
| `step_detector_evaded`: for every step threshold θ ≥ 2 and target D, a ramp with all steps < θ reaches deviation D | — | `COUNTEREXAMPLE` (case 7); the kernel's anchored latch fires (`kernel_latches_on_anchored`) |
| `predict_sound`: if the predictor says no breach within h, none occurs | **Every step deviation ≤ rate.** The assumption is about the process, not the predictor's output. | `CONDITIONAL` |
| `predict_not_complete` (false alarms); `predict_unsound_without_rate_bound` | — | `COUNTEREXAMPLE` |
| `no_false_negative_outside_band`, `no_false_positive_outside_band`: correct latch when the true deviation is ≥ θ + ε or < θ − ε | Measurement error ≤ ε (L1) | `CONDITIONAL` |
| `false_positive_in_band`, `false_negative_in_band` | ε = 1, θ = 5 | `COUNTEREXAMPLE` (cases 11, 12) |
| `manipulated_reports_hide_any_deviation` | A reporter that lies | `COUNTEREXAMPLE` (case 8) |

**Mathematical measurement versus empirical detection.** These results describe the metrics. They
say nothing about whether a real agent's dangerous behaviour shows up in the observation vector.
That is the observation contract, which is `OPEN` and needs empirical validation. No zero
false-positive or zero false-negative claim is made: both occur inside the noise band.

## 8. Twelve adversarial cases

All cases are in `RCACounterexamples`, as kernel-checked statements about reachable runs from
`Sys.init`. Bracketed names are the Python-replayed traces. Shortest witnesses come from the
bounded search (§10).

| # | Intended invariant | Missing hypothesis | Lean | Python | Reference kernel |
|---|---|---|---|---|---|
| 1 | acknowledged safety records survive restart (B, C) | persistence; no storage rollback | `ce1_volatile_memory_forgets_latch`, `ce1_rollback_forgets_latch`, `ce1_anchor_detects_rollback` | [`ablate_volatile_memory`, `kernel_rollback_without_anchor_loses_latch`, `kernel_anchor_detects_rollback`] | persists; **rollback detected only with an anchor** |
| 2 | no action without drift authorization | guard consults the latch | `ce2_action_without_drift_gate` | [`ablate_no_drift_gate`] | prevents |
| 3 | prohibitions are never lifted | monotone policy (+ reload) | `ce3_policy_weakening_executes_prohibited`, `ce3_reload_validation_stops_weakening` | [`ablate_policy_weakening`, `reload_validation_stops_weakened_policy`] | prevents; the loader also fails closed |
| 4 | no stale authorization | reload per critical section | `ce4_stale_view_reused`, `ce4_shortest_stale_view` | [`ablate_no_reload_stale_view`] | prevents (`authorized_before_execution`) |
| 5 | effects only through the guard | trust boundary | `ce5_direct_effect_bypasses_guard`; `effect_step_source` (only bypass) | [`kernel_direct_effect_bypasses_guard`] | **cannot prevent** (Python cannot isolate effects) |
| 6 | an authorization stays valid until the effect | reload + lock | `ce6_policy_change_after_authorization` | [`ablate_no_reload_policy_change`] | prevents |
| 7 | cumulative drift is bounded | anchored (not per-step) deviation | `step_detector_evaded`, `ce7_kernel_catches_slow_drift` | [`kernel_slow_drift_cumulative`] | prevents |
| 8 | detection reflects behaviour | faithful observation channel | `manipulated_reports_hide_any_deviation`, `ce8_manipulated_report_executes` | — | **cannot prevent** (`OPEN`) |
| 9 | no race on a restricted action | lock held from decision to effect | `ce9_race_without_lock`, `ce9_race_between_intent_and_effect`, `ce9_reference_serializes` | [`ablate_no_lock_race`]; `test_concurrent_processes_never_execute_after_latch` (real processes) | prevents (`inv_run`) |
| 10 | durable ⊇ acknowledged | fail-stop, durable acknowledgement | case 1a; `ce10_reference_divergence_is_safe`, `ce10_failed_write_not_acknowledged` | [`kernel_interrupted_writes`, `kernel_crash_between_intent_and_effect`] | prevents loss; over-approximates on crash (safe direction) |
| 11 | safe actions are not blocked | true deviation < θ − ε | `false_positive_in_band`, `ce11_false_positive_blocks_safe_action` | — | inherent inside the band |
| 12 | unsafe actions are blocked | true deviation ≥ θ + ε | `false_negative_in_band`, `ce12_false_negative_permits_unsafe_action` | — | inherent inside the band |

Two findings come from the bounded search:
- **The lock must cover the intent-to-effect window, not only the decision.** A latch recorded
  between the durable intent and the effect makes the effect unsafe
  (`ce9_race_between_intent_and_effect`).
- **Revalidating the memory on every transaction defends against several single-writer defects.**
  Accepting a weakening policy, a replayed observation or an unbound credential is caught by the
  loader at the next transaction, and every process fails closed. Harm needs those defects
  combined with skipping the reload. A missing drift gate or a missing lock is harmful on its own.

**Rollback** (strongest results):
- **Invisible without external state.** `load_take`: every non-empty prefix of a loadable memory
  loads. A restored older copy is, to the kernel, an honest shorter history. An append-only local
  file gives **no** protection against hostile storage rollback (`LEAN_PROVED`).
- **Below an anchor it fails closed.** `rollback_below_anchor_fails_closed` holds for every
  discipline.
- **An anchor retains what precedes it.** `anchor_retains_prohibitions`: every prohibition before
  the anchor survives in any memory the kernel accepts. `anchor_retains_latch`: so does the latch,
  unless a reset follows the anchor.
- **Required assumption:** an anchor that the adversary cannot roll back, refreshed after every
  safety-relevant record, plus collision-resistant digests. The reference only *accepts* an anchor
  (`GuardedAgent(anchor=...)`); it does not provide that external store.

## 9. Signer-side security findings

See [WaveLock-OTS signing guard](WAVELOCK_OTS_SIGNER_GUARD.md); this is a separate pull request.

**Defect.** At `c3cb0d1` the preexisting signing guard recorded only key ids, so one seed under
two key ids signed twice.

**Correction.** `sign_ots` now also claims `(params_hash, seed)` before revealing.

**Guarantees.**
- Single use of derivation inputs and key ids: `LEAN_PROVED`, conformance-tested on 14 traces,
  with 8-thread and 6-process race tests.
- Secret values reused through `sign_ots`: `CONDITIONAL` on SHAKE-256 collision resistance.
- Secret values recommitted by custom code: `OPEN`.

## 10. Python-to-Lean conformance

`lean/rca_conformance_traces.json` holds 25 traces. Each is checked by `decide` in
`RCAConformance.lean` against `RCA.runOut` and replayed against `GuardedAgent` by
`tests/test_rca_formal_conformance.py`. Each trace compares:
- the final durable log and torn flag;
- loadability under the anchor;
- every effect's kind (Lean also checks the specification's verdict);
- the acknowledged records;
- every request's verdict.

The traces cover:
- initialization, memory updates, repeated updates and policy updates;
- reload and restart;
- duplicate and stale observations;
- torn, failed and crashed writes, including a crash between the intent and the effect;
- conflicting policy versions, operator reset and credential replay;
- anchors and rollback;
- direct effects;
- the nine ablated disciplines.

Negative controls (all must fail, and do):
- mutated expectations;
- the guard replaced by "admit everything", on 5 guard-sensitive reference traces;
- persistence and reload removed from a restart trace;
- each ablation's events under the reference, which must yield a different and safe outcome (a
  loadable memory with one durable intent per effect).

Two Lean-side mutants were rejected by `decide`. A real three-process race test checks that no
execute record follows the first latching observation.

**Python harness caveats.**
- In the harness a blocked lock acquisition is a no-op, as in the model.
- Ablated disciplines are emulations: they replace `kernel.admissible` and drive the critical
  section without the lock or the reload. The loader is never changed.
- The hash chain, canonical encoding and HMAC are abstracted in Lean (a log *is* its record list).
  Edits that only the hash chain detects are tested in Python only (`test_edited_records_fail_closed`).

A universal refinement proof from Python to Lean is `OPEN`.

**Bounded model checking.** `lean/scripts/RCAModelCheck.lean` (receipt
`lean/rca_model_check_receipt.txt`) runs a breadth-first search over `RCA.step` itself:

| Search | Result |
|---|---|
| Reference, 23-event alphabet (two processes, latching and benign observations, action, tightening and weakening policies, reset, effects, crashes, restarts, credential) | no violation up to depth 8 (4,332 states) |
| Reference, write faults | none up to depth 8 (948 states) |
| Reference, replay alphabet | none up to depth 10 (202 states) |
| Shortest violation per ablation and boundary event | lengths 2 to 10 |
| Weakening with the reload kept | none up to depth 7 (3,206 states) |

These are bounded results with the stated alphabets. They do not replace the unbounded theorems.

## 11. Strict axiom audit

`lean/scripts/AxiomAudit.lean` (receipt `lean/axiom_audit_receipt.txt`; run in CI and compared)
enumerates **every constant** of the 14 OTS and RCA modules. That is 2,760 constants: theorems,
definitions and generated auxiliaries. It collects each constant's axioms and fails on anything
outside {`propext`, `Classical.choice`, `Quot.sound`}. That set includes `sorryAx` and the
`Lean.ofReduceBool` / `Lean.trustCompiler` axioms that `native_decide` introduces. It also fails
on declared axioms or opaques, and on missing central roots.
`scripts/AxiomAuditNegativeControl.lean` shows that a `sorryAx` proof and a `native_decide` proof
are both flagged.

| Measure | Value |
|---|---|
| Lean modules audited | 14 (6 from PR #47; 2 signer; 6 RCA) |
| Theorems | 402 in total: 47 signer, 177 RCA, 178 PR #47 |
| Definitions / types / derived instances / auxiliary | 526 / 43 / 50 / 1,739 |
| Audited central roots | 55 |
| Approved axioms | `propext`, `Classical.choice`, `Quot.sound` |
| Unexpected axioms | none |
| Failed declarations | none |
| `sorry` / `admit` / `native_decide` in the library | none |
| Unproved obligations presented as theorems | none (open items are listed in §13) |

## 12. Test and CI results

All counts are on the final local tree.

| Check | Result |
|---|---|
| `lake build` (pinned toolchain) | success (7,242 jobs); no warnings in the new modules |
| Strict audit and negative control | pass |
| `pytest tests -m "not slow"`, container interpreter | 604 passed, 10 skipped, 1 failed: `test_visualize_psi`, because matplotlib is not installed (environment) |
| The same suite with matplotlib installed (virtualenv; matplotlib is a declared dependency and CI installs it) | **605 passed, 10 skipped, 0 failed** |
| `audit/run_bounty_contract.py --check` | 13/13, no evidence mismatches |
| `hello_wavelock.py` | PASS |
| New tests | RCA conformance 42, reference kernel 22 (including a 3-process race), signer conformance 18, signer regressions 13 |
| Slow tests | not run (`-m "not slow"`, as in CI) |

CI results on the final pushed commits are recorded in the pull requests.

## 13. Remaining safety limitations

1. No RCA agent exists; nothing here protects a production action path.
2. Effects are isolated only by convention (case 5). Python offers no capability boundary.
3. Rollback is undetectable without an external monotonic anchor (`load_take`). The reference
   accepts an anchor but does not provide one.
4. The observation contract is open (case 8). Detection quality depends on what is measured.
5. False positives and false negatives are unavoidable inside the noise band (cases 11, 12).
6. There is liveness cost: a torn tail stops the agent until an operator repairs it, and a false
   positive latches until an operator reset.
7. The HMAC operator verifier is symmetric. A deployment needs an asymmetric verifier whose signing
   key the agent never holds.
8. A universal Python-to-Lean refinement is open; the evidence is 25 traces plus unit and process
   tests.
9. Only a single host is covered. Multi-host agreement is out of scope, as for OTS (#28).

## 14. Required future architecture

The minimal changes needed to make the guarantees hold in a real deployment:

- **Capability isolation for effects.** Run effects in a separate process or service that accepts
  only kernel-issued, intent-bound requests, closing case 5.
- **An external monotonic anchor** (TPM counter, chain anchor or witness service), refreshed after
  every safety-relevant record.
- **An authenticated observation channel** with a validated observation contract (sensor
  attestation; cross-checked measurements), plus an empirical evaluation of FP/FN rates.
- **An asymmetric operator credential** (for example a WaveLock-OTS signature over the prefix
  digest).
- **Torn-tail repair tooling** that cannot drop acknowledged records.
- **A refinement proof**, or an executable extraction of the Lean model, used as the kernel.

## Phase 13: what the results mean

| Level | Established? |
|---|---|
| 1. Mathematically consistent abstract state machine | **Yes**: `LEAN_PROVED` invariants over all finite traces. |
| 2. Faithful model of current Python behaviour | For the reference kernel and the signing guard: conformance-tested on finite traces (refinement `OPEN`). For a preexisting RCA agent: not applicable (absent). |
| 3. Verified safety-memory mechanism | **For the model**, under the trust boundary (Theorems B, C; signer `signer_single_use`). |
| 4. Verified intervention gate | **For the model**, under the trust boundary (`authorized_before_execution`, `pre_action_intervention`, `effect_blocked`, `safe_run`). |
| 5. Evidence of reliable drift detection | **No.** The metrics are defined and their properties proved, but empirical validation and an observation contract are missing. |
| 6. Real-world AI alignment or prevention of autonomous misbehaviour | **No.** Nothing here establishes it; a local transition invariant does not prove the absence of future AI failures. |

## Issue assessment

**#42 (semantic core).** The *actual* RCA semantics are **absent** from the implementation. Item
by item:

- **Implemented-state inventory:** complete (§1).
- **Typed formal semantic core:** complete for the proposed reference (types, memory, transitions,
  restart, replay).
- **Inductive invariants:** proved (Theorems A–D).
- **Python correspondence:** conformance-tested (25 traces); refinement `OPEN`.
- **Remaining architectural gaps:** §13–14.

In engineering terms, the reference-system formalization is complete for its stated scope, and a
real RCA agent architecture does not exist. The issue's acceptance criteria are met for the
reference, but the issue asks for the RCA system grounded in existing code. **Recommendation:**
keep #42 open until a real RCA agent exists, or re-scope it to the reference kernel.

**#43 (safety invariants and intervention).**

- **Proved for the reference model:**
  - safety-memory preservation and restart consistency;
  - replay inertness;
  - authorization strictly before execution, and blocking of unauthorized requests and effects;
  - inductive safety (`safe_run`);
  - the drift-metric properties of §7;
  - 12 adversarial cases plus rollback results;
  - bounded searches.
- **Unverified:** reliability of drift detection on real behaviour; the observation contract;
  effect isolation; external anchoring; production integration; refinement.
- **False positives and negatives:** characterized (band theorems), not eliminated.

**Recommendation:** keep #43 open for the empirical and architectural items.
