import WaveLockProofs.RCASafetyInvariants
import WaveLockProofs.RCADriftMetrics

/-!
# RCA reference safety kernel: twelve adversarial cases and rollback (issue #43)

**Status.**  Every case below is a kernel-checked statement about a reachable execution of the
`PROPOSED_MODEL` (`RCA.run` from `Sys.init`, so no unreachable state is assumed), or about the
drift metrics.  Each removes exactly one hypothesis of the positive results of
`RCASafetyInvariants` — an element of the reference discipline, or the trust boundary — and shows
that the guarantee then fails.  Traces with an executable counterpart are replayed against Python
by `tests/test_rca_formal_conformance.py` (trace names in brackets).  Historical counterexamples
are kept even where the reference rejects the attack.

| # | case | missing hypothesis | reference kernel |
|---|---|---|---|
| 1 | safety memory lost after restart | persistence; no storage rollback | persists; rollback detected only with an anchor |
| 2 | action before (drift) authorization | the guard consults the latch | prevents (`pre_action_intervention`) |
| 3 | policy weakening | monotone policy updates | prevents; the loader also fails closed |
| 4 | stale authorization replayed | reload in every critical section | prevents (`authorized_before_execution`) |
| 5 | alternate execution path | effects reachable only through the kernel | cannot prevent (trust boundary) |
| 6 | state change invalidates an authorization | reload + lock | prevents |
| 7 | small deviations accumulate | anchored (not per-step) deviation | prevents |
| 8 | manipulated measurement | faithful observation channel | cannot prevent (`OPEN`) |
| 9 | concurrent race | the lock | prevents (`inv_run`, all interleavings) |
| 10 | persistence failure divergence | fail-stop, durable acknowledgement | prevents loss; over-approximates on crash |
| 11 | false positive | true deviation outside the noise band | inherent inside the band |
| 12 | false negative | true deviation outside the noise band | inherent inside the band |
-/

namespace WaveLockProofs
namespace RCA

/-- Decidable form of `Safe`. -/
def safeB (S : Sys) : Bool :=
  S.effects.all (fun d => specPermits (credOK d.issued) d.log d.kind) &&
    S.acked.isSublist S.disk.log

theorem safeB_iff (S : Sys) : safeB S = true ↔ Safe S := by
  simp [safeB, Safe, List.isSublist_iff_sublist, List.all_eq_true]

/-- A request through the API: lock + reload, guard + append, and the effect of an action. -/
def req (q : ℕ) (r : Rec) : List Event :=
  match r with
  | .execute _ => [.begin q, .submit q r .ok, .effect q]
  | _ => [.begin q, .submit q r .ok]

/-- Agent 1, baseline `[0]`, latch threshold 5, nothing prohibited. -/
def prov : Event := .provision 1 [0] ⟨0, 5, []⟩

def genesis5 : Rec := .genesis 1 [0] ⟨0, 5, []⟩

/-! ## 1. Safety memory lost after restart -/

/-- Volatile memory: records are acknowledged without being written. -/
def ce1Volatile : List Event :=
  [prov] ++ req 0 (.observe 1 [9]) ++ [.crash 0, .restart 0] ++ req 0 (.execute 3)

/-- **Case 1a** (`ablate_volatile_memory`).  *Invariant:* acknowledged records are durable
(Theorem B).  *Missing:* persistence.  The acknowledged latching observation is not in the memory
the restarted process reloads; the action executes although the acknowledged history forbids it. -/
theorem ce1_volatile_memory_forgets_latch :
    let S := run { persist := false, reload := false } ce1Volatile
    safeB S = false ∧ S.effects.map Done.kind = [3] ∧
      specPermits (credOK S.issued) [genesis5, .observe 1 [9]] 3 = false := by
  decide

/-- **Case 1b** (`kernel_rollback_without_anchor_loses_latch`).  *Missing:* storage that is not
rolled back.  The reference kernel itself, after the memory file is restored to its genesis copy,
forgets the latch and executes. -/
theorem ce1_rollback_forgets_latch :
    let S := run .reference ([prov] ++ req 0 (.observe 1 [9]) ++ [.rollback 1] ++ req 0 (.execute 3))
    safeB S = false ∧ S.effects.map Done.kind = [3] := by
  decide

/-- **Case 1c** (`kernel_anchor_detects_rollback`).  With an anchor taken after the latch, the same
rollback makes every process fail closed. -/
theorem ce1_anchor_detects_rollback :
    let S := run .reference
      ([prov] ++ req 0 (.observe 1 [9]) ++ [.anchorNow, .rollback 1] ++ req 0 (.execute 3))
    S.effects = [] ∧ S.disk.view S.issued S.anchor = none := by
  decide

/-! ## 2. An action executes without the drift authorization -/

/-- **Case 2** (`ablate_no_drift_gate`).  *Invariant:* no effect of an action the specification
forbids (`safe_run`).  *Missing:* the guard consults the drift latch.  The action executes after a
latching observation was recorded.  (In every discipline an effect still needs a committed intent
— `effect_step_source`, `step_enters_committed` — so what is missing is the drift evaluation
inside the authorization, not the authorization step.) -/
theorem ce2_action_without_drift_gate :
    let S := run { driftGate := false } ([prov] ++ req 0 (.observe 1 [9]) ++ req 0 (.execute 3))
    safeB S = false ∧ S.effects.map Done.kind = [3] := by
  decide

/-! ## 3. Policy weakening -/

def prov7 : Event := .provision 1 [0] ⟨0, 5, [7]⟩

/-- **Case 3** (`ablate_policy_weakening`).  *Missing:* monotone policy updates (and reload).  An
update that drops the prohibition of action 7 lets it execute. -/
theorem ce3_policy_weakening_executes_prohibited :
    let S := run { monotonePolicy := false, reload := false }
      ([prov7] ++ req 0 (.policy ⟨1, 5, []⟩) ++ req 0 (.execute 7))
    safeB S = false ∧ S.effects.map Done.kind = [7] := by
  decide

/-- With the reload kept (`reload_validation_stops_weakened_policy`), the reference loader rejects
the weakened memory at the next request and every process fails closed: detection, not
prevention of the write — but the prohibited action does not run. -/
theorem ce3_reload_validation_stops_weakening :
    let S := run { monotonePolicy := false } ([prov7] ++ req 0 (.policy ⟨1, 5, []⟩) ++ req 0 (.execute 7))
    S.effects = [] ∧ S.disk.view S.issued S.anchor = none := by
  decide

/-! ## 4. A stale authorization is replayed -/

/-- **Case 4** (`ablate_no_reload_stale_view`).  *Missing:* the reload at the start of every
critical section.  Process 0 acts on the view it formed before process 1 recorded a latching
observation. -/
theorem ce4_stale_view_reused :
    let S := run { reload := false }
      ([prov] ++ req 0 (.execute 3) ++ req 1 (.observe 1 [9]) ++ req 0 (.execute 3))
    safeB S = false ∧ S.effects.map Done.kind = [3, 3] := by
  decide

/-- The shortest stale-view witness found by `scripts/RCAModelCheck.lean` (length 7): process 1
tightens the policy to prohibit action 3 while process 0 still holds the older view. -/
theorem ce4_shortest_stale_view :
    let S := run { reload := false }
      [prov, .begin 0, .submit 0 (.observe 2 [0]) .ok, .begin 1, .submit 1 (.policy ⟨1, 5, [3]⟩) .ok,
        .begin 0, .submit 0 (.execute 3) .ok, .effect 0]
    safeB S = false ∧ S.effects.map Done.kind = [3] := by
  decide

/-! ## 5. An alternate execution path -/

/-- **Case 5** (`kernel_direct_effect_bypasses_guard`).  *Missing:* the trust boundary "effects are
reachable only through `GuardedAgent.request`".  Python cannot enforce it; `effect_step_source`
proves this is the only model-level bypass. -/
theorem ce5_direct_effect_bypasses_guard :
    let S := run .reference ([prov] ++ req 0 (.observe 1 [9]) ++ [.directEffect 3])
    safeB S = false ∧ S.effects.map Done.kind = [3] := by
  decide

/-! ## 6. A state change invalidates a prior authorization -/

/-- **Case 6** (`ablate_no_reload_policy_change`).  *Missing:* reload under the lock.  Action 7 was
authorized; a policy update then prohibits it; the stale view executes it again. -/
theorem ce6_policy_change_after_authorization :
    let S := run { reload := false }
      ([.provision 1 [0] ⟨0, 9, []⟩] ++ req 0 (.execute 7) ++ req 1 (.policy ⟨1, 9, [7]⟩) ++
        req 0 (.execute 7))
    safeB S = false ∧ S.effects.map Done.kind = [7, 7] := by
  decide

/-! ## 7. Small deviations accumulate -/

/-- **Case 7** (`kernel_slow_drift_cumulative`).  Every step deviation is 1, below the threshold 3
(`step_detector_evaded`: a per-step detector never fires on such a ramp), but the reference
kernel's anchored latch fires once the state has moved by 3, and the action is then refused. -/
theorem ce7_kernel_catches_slow_drift :
    let S := run .reference
      ([.provision 1 [0] ⟨0, 3, []⟩] ++ req 0 (.observe 1 [1]) ++ req 0 (.execute 4) ++
        req 0 (.observe 2 [2]) ++ req 0 (.observe 3 [3]) ++ req 0 (.execute 4))
    S.effects.map Done.kind = [4] ∧ safeB S = true ∧ stepDetects 3 (ramp 3) = false := by
  decide

/-! ## 8. A manipulated measurement -/

/-- **Case 8.**  *Missing:* a faithful observation channel.  The reporter reports the baseline
while the true state has drifted arbitrarily far (`manipulated_reports_hide_any_deviation`); the
kernel, which only sees reports, executes, and its own specification — written over the reports —
cannot object.  Unresolved by any report-based kernel (`OPEN`: authenticated, faithful sensing). -/
theorem ce8_manipulated_report_executes :
    let S := run .reference ([prov] ++ req 0 (.observe 1 [0]) ++ req 0 (.execute 3))
    S.effects.map Done.kind = [3] ∧ safeB S = true ∧ 5 ≤ l1 [1000] [0] := by
  decide

/-! ## 9. A concurrent race -/

/-- **Case 9** (`ablate_no_lock_race`).  *Missing:* the memory lock.  Both processes load before
either writes; process 1 records the latch, process 0 then executes on its older view. -/
theorem ce9_race_without_lock :
    let S := run { lock := false }
      [prov, .begin 0, .begin 1, .submit 1 (.observe 1 [9]) .ok, .submit 0 (.execute 3) .ok,
        .effect 0]
    safeB S = false ∧ S.effects.map Done.kind = [3] := by
  decide

/-- The shortest race found by `scripts/RCAModelCheck.lean` (length 5): without the lock, a latching
observation is recorded *between* process 0's durable intent and its effect, so the effect runs on
a memory that already forbids it.  The reference holds the lock from the decision until the effect
has run; this witness shows that covering only the decision would not suffice. -/
theorem ce9_race_between_intent_and_effect :
    let S := run { lock := false }
      [prov, .begin 0, .submit 0 (.execute 3) .ok, .begin 1, .submit 1 (.observe 1 [9]) .ok, .effect 0]
    safeB S = false ∧ S.effects.map Done.kind = [3] := by
  decide

/-- The same interleaving under the reference: process 1 cannot enter while process 0 holds the
lock, so its observation is not recorded before the action (and is refused, not lost: the caller
retries). -/
theorem ce9_reference_serializes :
    let S := run .reference
      [prov, .begin 0, .begin 1, .submit 1 (.observe 1 [9]) .ok, .submit 0 (.execute 3) .ok,
        .effect 0]
    safeB S = true ∧ S.effects.map Done.kind = [3] ∧ S.disk.log = [genesis5, .execute 3] := by
  decide

/-! ## 10. Persistence failure and divergence -/

/-- **Case 10.**  *Invariant:* the acknowledged history is a sub-history of the durable one
(Theorem B).  Without persistence (case 1a) they diverge.  The reference's only divergence is in
the safe direction: a crash between the durable intent and the effect leaves an `execute` record
whose effect never ran (at-most-once; `kernel_crash_between_intent_and_effect`). -/
theorem ce10_reference_divergence_is_safe :
    let S := run .reference
      ([prov, .begin 0, .submit 0 (.execute 3) .ok, .crash 0] ++ req 1 (.execute 3))
    safeB S = true ∧ S.effects.map Done.kind = [3] ∧
      S.disk.log = [genesis5, .execute 3, .execute 3] ∧ S.acked = [.execute 3] := by
  decide

/-- Write failures never acknowledge: an `OSError` after the record reached the file leaves a
durable, unacknowledged record and no effect. -/
theorem ce10_failed_write_not_acknowledged :
    let S := run .reference [prov, .begin 0, .submit 0 (.execute 3) (.fail .line), .effect 0]
    S.effects = [] ∧ S.acked = [] ∧ S.disk.log = [genesis5, .execute 3] := by
  decide

/-! ## 11, 12. False positives and false negatives -/

/-- **Case 11** (`false_positive_in_band`).  The true state `[4]` is safe for threshold 5; the
report `[5]` (measurement error 1) latches; a safe action is refused. -/
theorem ce11_false_positive_blocks_safe_action :
    let S := run .reference ([prov] ++ req 0 (.observe 1 [5]) ++ req 0 (.execute 3))
    S.effects = [] ∧ l1 [4] [0] < 5 ∧ l1 [5] [4] ≤ 1 := by
  decide

/-- **Case 12** (`false_negative_in_band`).  The true state `[5]` reaches threshold 5; the report
`[4]` does not latch; the action executes. -/
theorem ce12_false_negative_permits_unsafe_action :
    let S := run .reference ([prov] ++ req 0 (.observe 1 [4]) ++ req 0 (.execute 3))
    S.effects.map Done.kind = [3] ∧ 5 ≤ l1 [5] [0] ∧ l1 [4] [5] ≤ 1 := by
  decide

/-! ## Rollback -/

section Rollback

variable {vf : CredOK}

theorem fold_take : ∀ (pre : List Rec) (v : View) (rs : List Rec) (w : View),
    fold vf pre v rs = some w → ∀ m, ∃ w', fold vf pre v (rs.take m) = some w'
  | _, v, [], _, _, _ => ⟨v, by simp [fold]⟩
  | pre, v, r :: rs, w, h, m => by
    cases m with
    | zero => exact ⟨v, by simp [fold]⟩
    | succ m =>
      simp only [fold] at h
      split_ifs at h with ha
      obtain ⟨w', hw'⟩ := fold_take (pre ++ [r]) (next v r) rs w h m
      exact ⟨w', by simp only [List.take_succ_cons, fold, if_pos ha]; exact hw'⟩

/-- **Rollback is invisible to the loader.**  Every non-empty prefix of a loadable memory is itself
loadable: a restored older copy of the file is, to the kernel, an honest history that ended there.
An append-only local file therefore gives no protection against hostile storage rollback. -/
theorem load_take {L : List Rec} {v : View} (h : load vf L = some v) {n : ℕ} (hn : 1 ≤ n) :
    ∃ w, load vf (L.take n) = some w := by
  cases L with
  | nil => simp [load] at h
  | cons g rs =>
    obtain ⟨m, rfl⟩ := Nat.exists_eq_add_of_le' hn
    simp only [load] at h
    cases hg : genesisView g with
    | none => simp [hg] at h
    | some v0 =>
      simp only [hg, Option.bind_some] at h
      obtain ⟨w, hw⟩ := fold_take [g] v0 rs v h m
      exact ⟨w, by simp [List.take_succ_cons, load, hg, hw]⟩

/-- **Rollback below an anchor fails closed** (every discipline): after the head is anchored, any
restore to a strictly shorter prefix leaves no loadable memory. -/
theorem rollback_below_anchor_fails_closed (D : Discipline) (S : Sys) {n : ℕ}
    (hn : n < S.disk.log.length) :
    let S' := step D (step D S .anchorNow) (.rollback n)
    S'.disk.view S'.issued S'.anchor = none := by
  simp only [step, Disk.view]
  rw [if_neg]
  rintro ⟨-, hp⟩
  have := (List.isPrefixOf_iff_prefix.1 hp).length_le
  simp only [List.length_take] at this
  omega

theorem specFold_append_list : ∀ (pre : List Rec) (s : Spec) (xs ys : List Rec),
    specFold vf pre s (xs ++ ys) = specFold vf (pre ++ xs) (specFold vf pre s xs) ys
  | _, _, [], _ => by simp [specFold]
  | pre, s, x :: xs, ys => by
    simp only [List.cons_append, specFold]
    rw [specFold_append_list (pre ++ [x]) _ xs ys]
    simp [List.append_assoc]

theorem specOf_append {L : List Rec} (hL : L ≠ []) (rs : List Rec) :
    specOf vf (L ++ rs) = (specOf vf L).map fun s => specFold vf L s rs := by
  obtain ⟨g, gs, rfl⟩ := List.exists_cons_of_ne_nil hL
  simp only [List.cons_append, specOf]
  cases specInit g with
  | none => rfl
  | some s => simp [specFold_append_list]

theorem specFold_proh_mono : ∀ (pre : List Rec) (s : Spec) (rs : List Rec) (k : ℕ),
    k ∈ s.proh → k ∈ (specFold vf pre s rs).proh
  | _, _, [], _, h => h
  | pre, s, r :: rs, k, h => by
    apply specFold_proh_mono (pre ++ [r]) _ rs k
    cases r with
    | observe q xs => simp only [specStep]; split_ifs <;> exact h
    | policy p => simp [specStep, h]
    | reset c => simp only [specStep]; split_ifs <;> exact h
    | execute _ => exact h
    | genesis _ _ _ => exact h

theorem specFold_hazard_sticky : ∀ (pre : List Rec) (s : Spec) (rs : List Rec),
    s.hazard = true → (∀ c, Rec.reset c ∉ rs) → (specFold vf pre s rs).hazard = true
  | _, _, [], h, _ => h
  | pre, s, r :: rs, h, hr => by
    apply specFold_hazard_sticky (pre ++ [r]) _ rs _ (fun c hc => hr c (List.mem_cons_of_mem _ hc))
    cases r with
    | observe q xs => simp only [specStep]; split_ifs <;> simp [h]
    | policy p => simp [specStep, h]
    | reset c => exact absurd List.mem_cons_self (hr c)
    | execute _ => exact h
    | genesis _ _ _ => exact h

/-- **What an anchor retains: prohibitions.**  If the memory loads under anchor `A`, every action
kind prohibited anywhere in `A` is prohibited in the loaded view.  No rollback, rewrite or
extension that the kernel accepts can lift a prohibition recorded before the anchor. -/
theorem anchor_retains_prohibitions {I : Issued} {A : List Rec} {d : Disk} {v : View}
    (h : d.view I A = some v) {s : Spec} (hs : specOf (credOK I) A = some s) :
    ∀ k ∈ s.proh, k ∈ v.policy.prohibited := by
  intro k hk
  obtain ⟨-, hpre, hload⟩ := disk_view_some h
  obtain ⟨rest, hrest⟩ := hpre
  have hA : A ≠ [] := by rintro rfl; simp [specOf] at hs
  obtain ⟨s', hs', hrel⟩ := load_refines_spec hload
  rw [← hrest, specOf_append hA, hs] at hs'
  simp only [Option.map_some, Option.some.injEq] at hs'
  exact (hrel.2.2.1 k).1 (hs' ▸ specFold_proh_mono A s rest k hk)

/-- **What an anchor retains: the latch.**  If the memory loads under anchor `A`, the latch of `A`
is set, and no reset record follows `A` (a reset needs an operator credential bound to its exact
prefix), then the loaded view is locked. -/
theorem anchor_retains_latch {I : Issued} {A : List Rec} {d : Disk} {v : View}
    (h : d.view I A = some v) {s : Spec} (hs : specOf (credOK I) A = some s) (hz : s.hazard = true)
    (hnr : ∀ c, Rec.reset c ∉ d.log.drop A.length) : v.locked = true := by
  obtain ⟨-, hpre, hload⟩ := disk_view_some h
  obtain ⟨rest, hrest⟩ := hpre
  have hA : A ≠ [] := by rintro rfl; simp [specOf] at hs
  obtain ⟨s', hs', hrel⟩ := load_refines_spec hload
  have hdrop : d.log.drop A.length = rest := by rw [← hrest]; simp
  rw [← hrest, specOf_append hA, hs] at hs'
  simp only [Option.map_some, Option.some.injEq] at hs'
  rw [← hrel.2.2.2.2.2, ← hs']
  exact specFold_hazard_sticky A s rest hz (by rw [← hdrop]; exact hnr)

end Rollback

end RCA
end WaveLockProofs
