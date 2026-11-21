import WaveLockProofs.RCATransitions

/-!
# RCA reference safety kernel: safety memory, pre-action intervention and preservation

**Status.**  `LEAN_PROVED` statements about the `PROPOSED_MODEL` of the proposed reference kernel
`wavelock/rca_reference` (issues #42 and #43).  They are not statements about preexisting
production behaviour: none existed (`docs/RCA_FORMALIZATION.md`, Phase 0).

`Inv` is an inductive invariant of the reference discipline over *every* finite interleaving of
any number of processes, with lock contention, write failures and crashes at every modelled point
(nothing, a complete record, or a torn tail reaching the file; a crash between the durable intent
and the effect), process restarts, operator credentials and anchors.  Out-of-boundary events
(`directEffect`, `rollback`, `tamper`) are excluded by the hypothesis `∀ e ∈ es, e.inBoundary`;
`RCACounterexamples` shows that each exclusion, and each element of the reference discipline, is
necessary.

## Results

* **Theorem A** `state_validity` — an intact durable memory always loads, and every process in the
  critical section holds the view folded from the durable log.
* **Theorem B** `safety_memory_preserved` — every acknowledged record is in the durable log, in
  acknowledgment order; `log_prefix_step` — no in-boundary transition (of any discipline) deletes,
  compacts or rewrites a durable record.
* **Theorem C** `restart_consistency` — a restarted process recovers exactly the fold of the
  durable log, which contains every acknowledged record and whose latch is the specification's
  hazard.
* **Theorem D** `replay_is_inert`, `credential_bound` — replayed records are refused (a replayed
  action request changes no state and is judged afresh); every durable reset carries a credential
  issued for exactly its prefix.
* **Authorization before execution** `authorized_before_execution` — in every in-boundary run,
  every step that performs an effect of kind `k` is the `effect` event of a process that, at a
  strictly earlier step, submitted `execute k`, had its guard evaluated on the view folded from the
  then-current durable memory (inside the critical section), was admitted, and made the intent
  durable; from that step until the effect it stayed committed and the durable log was that memory
  plus the intent (no other record was written in between).
* **Blocking** `pre_action_intervention` — an `execute` request that the specification does not
  permit on the current durable memory is refused: nothing is written and no intent is committed;
  `effect_blocked` — no in-boundary step performs an effect of a kind the current durable memory
  does not permit.  `effect_step_source`, `step_enters_committed` (every discipline) — effects
  arise only from committed intents (or the out-of-boundary `directEffect`), and intents are
  committed only by a guarded `submit`.
* **Preservation** `safe_run` — every effect of every in-boundary run was permitted by the
  specification on the durable memory when it ran, and no acknowledged record was lost.
-/

namespace WaveLockProofs
namespace RCA

/-! ## Basic lemmas -/

section Basic

theorem Sys.set_self {S : Sys} {q : ℕ} {ph : Phase} : S.set q ph q = ph := by simp [Sys.set]

theorem Sys.set_ne {S : Sys} {q x : ℕ} {ph : Phase} (h : x ≠ q) : S.set q ph x = S.procs x := by
  simp [Sys.set, h]

theorem load_singleton (vf : CredOK) (g : Rec) : load vf [g] = genesisView g := by
  cases g <;> simp [load, genesisView, fold]

theorem credOK_mono {I : Issued} {x : ℕ × List Rec} :
    ∀ c pre, credOK I c pre = true → credOK (I ++ [x]) c pre = true := by
  intro c pre h; simp only [credOK, decide_eq_true_eq] at h ⊢; exact List.mem_append_left _ h

theorem credAny_false {I : Issued} {c : ℕ} : credAny I c = false ↔ c ∉ I.map Prod.fst := by
  simp only [credAny, List.any_eq_false, beq_iff_eq, List.mem_map]
  constructor
  · rintro h ⟨x, hx, rfl⟩; exact h x hx rfl
  · intro h x hx hc; exact h ⟨x, hx, hc⟩

theorem disk_view_some {I : Issued} {A : List Rec} {d : Disk} {v : View}
    (h : d.view I A = some v) :
    d.torn = false ∧ A <+: d.log ∧ load (credOK I) d.log = some v := by
  unfold Disk.view at h
  split_ifs at h with hc
  exact ⟨hc.1, List.isPrefixOf_iff_prefix.1 hc.2, h⟩

theorem disk_write_log (d : Disk) (r : Rec) (e : Effect) :
    (d.write r e).log = d.log ∨ (d.write r e).log = d.log ++ [r] := by
  cases e <;> simp [Disk.write]

theorem disk_write_prefix (d : Disk) (r : Rec) (e : Effect) : d.log <+: (d.write r e).log := by
  cases e
  · exact List.prefix_refl _
  · exact List.prefix_append _ _
  · exact List.prefix_refl _

end Basic

/-! ## Step equations of the reference discipline -/

section StepEq

variable {S : Sys}

theorem step_begin_some {q : ℕ} {pre : List Rec} {v : View} (hq : S.procs q = .idle)
    (hl : S.lock = none) (hf : S.fresh = some (pre, v)) :
    step .reference S (.begin q) =
      { S with procs := S.set q (.holding pre v), lock := some q,
               cache := S.setCache q (some (pre, v)) } := by
  simp [step, hq, hl, hf]

theorem step_begin_none {q : ℕ} (hq : S.procs q = .idle) (hf : S.fresh = none) :
    step .reference S (.begin q) = S := by
  simp [step, hq, hf]

theorem step_begin_busy {q : ℕ} (hl : S.lock ≠ none) : step .reference S (.begin q) = S := by
  cases hq : S.procs q <;> simp [step, hq, hl]

theorem step_begin_not_idle {q : ℕ} (hq : S.procs q ≠ .idle) : step .reference S (.begin q) = S := by
  cases h : S.procs q with
  | idle => exact absurd h hq
  | _ => simp [step, h]

theorem step_submit_refused {q : ℕ} {pre : List Rec} {v : View} {r : Rec} {o : WriteOutcome}
    (hq : S.procs q = .holding pre v) (hg : admissible (credOK S.issued) pre v r = false) :
    step .reference S (.submit q r o) =
      { S with procs := S.set q .idle, lock := release q S.lock } := by
  have hg' : guard .reference S.issued pre v r = false := by rw [guard_reference]; exact hg
  simp [step, hq, hg']

theorem step_submit_ok_execute {q : ℕ} {pre : List Rec} {v : View} {k : ℕ}
    (hq : S.procs q = .holding pre v)
    (hg : admissible (credOK S.issued) pre v (.execute k) = true) :
    step .reference S (.submit q (.execute k) .ok) =
      { S with disk := S.disk.write (.execute k) .line, procs := S.set q (.committed pre v k),
               cache := S.setCache q (some (pre ++ [.execute k], next v (.execute k))) } := by
  have hg' : guard .reference S.issued pre v (.execute k) = true := by
    rw [guard_reference]; exact hg
  simp [step, hq, hg']

theorem step_submit_ok_other {q : ℕ} {pre : List Rec} {v : View} {r : Rec}
    (hq : S.procs q = .holding pre v) (hg : admissible (credOK S.issued) pre v r = true)
    (hr : ∀ k, r ≠ .execute k) :
    step .reference S (.submit q r .ok) =
      { S with disk := S.disk.write r .line, procs := S.set q .idle, lock := release q S.lock,
               cache := S.setCache q (some (pre ++ [r], next v r)),
               acked := S.acked ++ [r] } := by
  have hg' : guard .reference S.issued pre v r = true := by rw [guard_reference]; exact hg
  cases r with
  | execute k => exact absurd rfl (hr k)
  | _ => simp [step, hq, hg']

theorem step_submit_fail {q : ℕ} {pre : List Rec} {v : View} {r : Rec} {e : Effect}
    (hq : S.procs q = .holding pre v) (hg : admissible (credOK S.issued) pre v r = true) :
    step .reference S (.submit q r (.fail e)) =
      { S with disk := S.disk.write r e, procs := S.set q .idle, lock := release q S.lock } := by
  have hg' : guard .reference S.issued pre v r = true := by rw [guard_reference]; exact hg
  simp [step, hq, hg']

theorem step_submit_crash {q : ℕ} {pre : List Rec} {v : View} {r : Rec} {e : Effect}
    (hq : S.procs q = .holding pre v) (hg : admissible (credOK S.issued) pre v r = true) :
    step .reference S (.submit q r (.crash e)) =
      { S with disk := S.disk.write r e, procs := S.set q .dead, lock := release q S.lock } := by
  have hg' : guard .reference S.issued pre v r = true := by rw [guard_reference]; exact hg
  simp [step, hq, hg']

theorem step_submit_not_holding {q : ℕ} {r : Rec} {o : WriteOutcome} (D : Discipline)
    (hq : ∀ pre v, S.procs q ≠ .holding pre v) : step D S (.submit q r o) = S := by
  cases h : S.procs q with
  | holding pre v => exact absurd h (hq pre v)
  | _ => simp [step, h]

theorem step_effect_committed {q : ℕ} {pre : List Rec} {v : View} {k : ℕ} (D : Discipline)
    (hq : S.procs q = .committed pre v k) :
    step D S (.effect q) =
      { S with procs := S.set q .idle, lock := release q S.lock,
               acked := S.acked ++ [.execute k],
               effects := S.effects ++ [⟨k, pre, S.disk.log, S.issued⟩] } := by
  simp [step, hq]

theorem step_effect_not_committed {q : ℕ} (D : Discipline)
    (hq : ∀ pre v k, S.procs q ≠ .committed pre v k) : step D S (.effect q) = S := by
  cases h : S.procs q with
  | committed pre v k => exact absurd h (hq pre v k)
  | _ => simp [step, h]

end StepEq

/-! ## The invariant -/

/-- Inductive invariant of the reference discipline. -/
structure Inv (S : Sys) : Prop where
  /-- Theorem A: an intact durable memory is empty or loads -/
  valid : S.disk.torn = false → S.disk.log = [] ∨ ∃ v, load (credOK S.issued) S.disk.log = some v
  anchor : S.anchor <+: S.disk.log
  /-- a process inside the critical section owns the lock -/
  owner : ∀ q pre v, S.procs q = .holding pre v → S.lock = some q
  ownerC : ∀ q pre v k, S.procs q = .committed pre v k → S.lock = some q
  /-- a holder's view is the fold of the current durable log -/
  fresh : ∀ q pre v, S.procs q = .holding pre v →
    pre = S.disk.log ∧ S.disk.torn = false ∧ load (credOK S.issued) pre = some v
  /-- a committed intent was admitted on the fold of the memory it extends -/
  committed : ∀ q pre v k, S.procs q = .committed pre v k →
    S.disk.log = pre ++ [.execute k] ∧ S.disk.torn = false ∧ load (credOK S.issued) pre = some v ∧
      v.authorizes k = true ∧ List.Sublist S.acked pre
  /-- every effect was permitted by the specification on the durable memory when it ran, and that
  memory was the decision's memory plus the intent -/
  safe : ∀ e ∈ S.effects, specPermits (credOK e.issued) e.log e.kind = true ∧
    e.log = e.decided ++ [.execute e.kind]
  /-- Theorem B: acknowledged records are durable, in order -/
  mem : List.Sublist S.acked S.disk.log
  credNodup : (S.issued.map Prod.fst).Nodup
  credPast : ∀ x ∈ S.issued, x.2 <+: S.disk.log

section Preservation

variable {S : Sys}

theorem inv_init : Inv Sys.init where
  valid := fun _ => Or.inl rfl
  anchor := List.prefix_refl _
  owner := by simp [Sys.init]
  ownerC := by simp [Sys.init]
  fresh := by simp [Sys.init]
  committed := by simp [Sys.init]
  safe := by simp [Sys.init]
  mem := List.Sublist.slnil
  credNodup := List.nodup_nil
  credPast := by simp [Sys.init]

/-- Being inside the critical section. -/
def InCS (S : Sys) (q : ℕ) : Prop :=
  (∃ pre v, S.procs q = .holding pre v) ∨ ∃ pre v k, S.procs q = .committed pre v k

theorem lock_of_inCS (h : Inv S) {q : ℕ} (hq : InCS S q) : S.lock = some q := by
  rcases hq with ⟨pre, v, hq⟩ | ⟨pre, v, k, hq⟩
  · exact h.owner q pre v hq
  · exact h.ownerC q pre v k hq

theorem not_inCS_of_lock_none (h : Inv S) (hl : S.lock = none) (q : ℕ) : ¬ InCS S q := by
  intro hq; have := lock_of_inCS h hq; rw [hl] at this; cases this

theorem inCS_unique (h : Inv S) {q x : ℕ} (hq : InCS S q) (hx : InCS S x) : x = q := by
  have h1 := lock_of_inCS h hq; have h2 := lock_of_inCS h hx
  rw [h1] at h2; exact (Option.some.inj h2).symm

/-- After `q` leaves the critical section (to a phase outside it), nobody is inside. -/
theorem none_inCS_after (h : Inv S) {q : ℕ} (hq : InCS S q) (ph : Phase)
    (hph : (∀ pre v, ph ≠ .holding pre v) ∧ ∀ pre v k, ph ≠ .committed pre v k) :
    (∀ x pre v, S.set q ph x ≠ .holding pre v) ∧ ∀ x pre v k, S.set q ph x ≠ .committed pre v k := by
  constructor
  · intro x pre v hx
    by_cases hxq : x = q
    · subst hxq; rw [Sys.set_self] at hx; exact hph.1 pre v hx
    · rw [Sys.set_ne hxq] at hx; exact hxq (inCS_unique h hq (Or.inl ⟨pre, v, hx⟩))
  · intro x pre v k hx
    by_cases hxq : x = q
    · subst hxq; rw [Sys.set_self] at hx; exact hph.2 pre v k hx
    · rw [Sys.set_ne hxq] at hx; exact hxq (inCS_unique h hq (Or.inr ⟨pre, v, k, hx⟩))

theorem idle_out :
    (∀ pre v, Phase.idle ≠ .holding pre v) ∧ ∀ pre v k, Phase.idle ≠ .committed pre v k :=
  ⟨fun _ _ hc => (by cases hc), fun _ _ _ hc => (by cases hc)⟩

theorem dead_out :
    (∀ pre v, Phase.dead ≠ .holding pre v) ∧ ∀ pre v k, Phase.dead ≠ .committed pre v k :=
  ⟨fun _ _ hc => (by cases hc), fun _ _ _ hc => (by cases hc)⟩

theorem inv_provision (h : Inv S) (a : ℕ) (b : List ℤ) (p : Policy) :
    Inv (step .reference S (.provision a b p)) := by
  simp only [step]
  split_ifs with hc
  · obtain ⟨hlog, -, hg⟩ := hc
    have hnone : ∀ q pre v, S.procs q ≠ .holding pre v := by
      intro q pre v hq
      obtain ⟨rfl, -, hl⟩ := h.fresh q pre v hq
      rw [hlog] at hl; simp [load] at hl
    have hnoneC : ∀ q pre v k, S.procs q ≠ .committed pre v k := by
      intro q pre v k hq
      have := (h.committed q pre v k hq).1; rw [hlog] at this; simp at this
    obtain ⟨v, hv⟩ := Option.isSome_iff_exists.1 hg
    refine ⟨fun _ => Or.inr ⟨v, by rw [load_singleton]; exact hv⟩, ?_, ?_, ?_, ?_, ?_, h.safe, ?_,
      h.credNodup, ?_⟩
    · have := h.anchor; rw [hlog, List.prefix_nil] at this; rw [this]; exact List.nil_prefix
    · intro q pre v hq; exact absurd hq (hnone q pre v)
    · intro q pre v k hq; exact absurd hq (hnoneC q pre v k)
    · intro q pre v hq; exact absurd hq (hnone q pre v)
    · intro q pre v k hq; exact absurd hq (hnoneC q pre v k)
    · have := h.mem; rw [hlog, List.sublist_nil] at this; rw [this]; exact List.nil_sublist _
    · intro x hx; have := h.credPast x hx; rw [hlog, List.prefix_nil] at this
      rw [this]; exact List.nil_prefix
  · exact h

theorem inv_begin (h : Inv S) (q : ℕ) : Inv (step .reference S (.begin q)) := by
  by_cases hq : S.procs q = .idle
  · by_cases hl : S.lock = none
    · cases hf : S.fresh with
      | none => rw [step_begin_none hq hf]; exact h
      | some pv =>
        obtain ⟨pre, v⟩ := pv
        rw [step_begin_some hq hl hf]
        simp only [Sys.fresh, Option.map_eq_some_iff, Prod.mk.injEq] at hf
        obtain ⟨w, hw, rfl, rfl⟩ := hf
        obtain ⟨ht, -, hload⟩ := disk_view_some hw
        have hno := not_inCS_of_lock_none h hl
        refine ⟨h.valid, h.anchor, ?_, ?_, ?_, ?_, h.safe, h.mem, h.credNodup, h.credPast⟩
        · intro x pre' v' hx
          dsimp only at hx ⊢
          by_cases hxq : x = q
          · subst hxq; rfl
          · rw [Sys.set_ne hxq] at hx; exact absurd (Or.inl ⟨pre', v', hx⟩) (hno x)
        · intro x pre' v' k hx
          dsimp only at hx ⊢
          by_cases hxq : x = q
          · subst hxq; simp [Sys.set_self] at hx
          · rw [Sys.set_ne hxq] at hx; exact absurd (Or.inr ⟨pre', v', k, hx⟩) (hno x)
        · intro x pre' v' hx
          dsimp only at hx ⊢
          by_cases hxq : x = q
          · subst hxq; rw [Sys.set_self] at hx; cases hx; exact ⟨rfl, ht, hload⟩
          · rw [Sys.set_ne hxq] at hx; exact absurd (Or.inl ⟨pre', v', hx⟩) (hno x)
        · intro x pre' v' k hx
          dsimp only at hx ⊢
          by_cases hxq : x = q
          · subst hxq; rw [Sys.set_self] at hx; cases hx
          · rw [Sys.set_ne hxq] at hx; exact absurd (Or.inr ⟨pre', v', k, hx⟩) (hno x)
    · rw [step_begin_busy hl]; exact h
  · rw [step_begin_not_idle hq]; exact h

theorem inv_submit (h : Inv S) (q : ℕ) (r : Rec) (o : WriteOutcome) :
    Inv (step .reference S (.submit q r o)) := by
  by_cases hhold : ∃ pre v, S.procs q = .holding pre v
  · obtain ⟨pre, v, hq⟩ := hhold
    obtain ⟨rfl, htorn, hload⟩ := h.fresh q pre v hq
    have hcs : InCS S q := Or.inl ⟨_, _, hq⟩
    have hne : S.disk.log ≠ [] := load_ne_nil hload
    obtain ⟨hidle, hidleC⟩ := none_inCS_after h hcs .idle idle_out
    obtain ⟨hdead, hdeadC⟩ := none_inCS_after h hcs .dead dead_out
    cases hg : admissible (credOK S.issued) S.disk.log v r with
    | false =>
      rw [step_submit_refused hq hg]
      exact ⟨h.valid, h.anchor, fun x pre' v' hx => absurd hx (hidle x pre' v'),
        fun x pre' v' k hx => absurd hx (hidleC x pre' v' k),
        fun x pre' v' hx => absurd hx (hidle x pre' v'),
        fun x pre' v' k hx => absurd hx (hidleC x pre' v' k), h.safe, h.mem, h.credNodup,
        h.credPast⟩
    | true =>
      have hnext := load_snoc_some hne hload hg
      have hwrite : ∀ e : Effect, ((S.disk.write r e).torn = false →
          (S.disk.write r e).log = [] ∨
            ∃ w, load (credOK S.issued) (S.disk.write r e).log = some w) ∧
          S.anchor <+: (S.disk.write r e).log ∧ List.Sublist S.acked (S.disk.write r e).log ∧
          ∀ x ∈ S.issued, x.2 <+: (S.disk.write r e).log := by
        intro e
        rcases disk_write_log S.disk r e with hl | hl <;> rw [hl]
        · exact ⟨fun _ => Or.inr ⟨v, hload⟩, h.anchor, h.mem, h.credPast⟩
        · exact ⟨fun _ => Or.inr ⟨_, hnext⟩, h.anchor.trans (List.prefix_append _ _),
            h.mem.trans (List.sublist_append_left _ _),
            fun x hx => (h.credPast x hx).trans (List.prefix_append _ _)⟩
      cases o with
      | ok =>
        obtain ⟨hv, ha, hm, hc⟩ := hwrite .line
        by_cases hr : ∃ k, r = .execute k
        · obtain ⟨k, rfl⟩ := hr
          rw [step_submit_ok_execute hq hg]
          refine ⟨hv, ha, ?_, ?_, ?_, ?_, h.safe, hm, h.credNodup, hc⟩
          · intro x pre' v' hx
            dsimp only at hx ⊢
            by_cases hxq : x = q
            · subst hxq; rw [Sys.set_self] at hx; cases hx
            · rw [Sys.set_ne hxq] at hx; exact absurd (inCS_unique h hcs (Or.inl ⟨_, _, hx⟩)) hxq
          · intro x pre' v' k' hx
            dsimp only at hx ⊢
            by_cases hxq : x = q
            · subst hxq; exact lock_of_inCS h hcs
            · rw [Sys.set_ne hxq] at hx
              exact absurd (inCS_unique h hcs (Or.inr ⟨_, _, _, hx⟩)) hxq
          · intro x pre' v' hx
            dsimp only at hx
            by_cases hxq : x = q
            · subst hxq; rw [Sys.set_self] at hx; cases hx
            · rw [Sys.set_ne hxq] at hx; exact absurd (inCS_unique h hcs (Or.inl ⟨_, _, hx⟩)) hxq
          · intro x pre' v' k' hx
            dsimp only at hx ⊢
            by_cases hxq : x = q
            · subst hxq; rw [Sys.set_self] at hx; cases hx
              refine ⟨by simp [Disk.write], by simpa [Disk.write] using htorn, hload, ?_, h.mem⟩
              simpa [admissible] using hg
            · rw [Sys.set_ne hxq] at hx
              exact absurd (inCS_unique h hcs (Or.inr ⟨_, _, _, hx⟩)) hxq
        · push_neg at hr
          rw [step_submit_ok_other hq hg hr]
          refine ⟨hv, ha, fun x pre' v' hx => absurd hx (hidle x pre' v'),
            fun x pre' v' k hx => absurd hx (hidleC x pre' v' k),
            fun x pre' v' hx => absurd hx (hidle x pre' v'),
            fun x pre' v' k hx => absurd hx (hidleC x pre' v' k), h.safe, ?_, h.credNodup, hc⟩
          simpa [Disk.write] using h.mem.append (List.Sublist.refl [r])
      | fail e =>
        rw [step_submit_fail hq hg]
        obtain ⟨hv, ha, hm, hc⟩ := hwrite e
        exact ⟨hv, ha, fun x pre' v' hx => absurd hx (hidle x pre' v'),
          fun x pre' v' k hx => absurd hx (hidleC x pre' v' k),
          fun x pre' v' hx => absurd hx (hidle x pre' v'),
          fun x pre' v' k hx => absurd hx (hidleC x pre' v' k), h.safe, hm, h.credNodup, hc⟩
      | crash e =>
        rw [step_submit_crash hq hg]
        obtain ⟨hv, ha, hm, hc⟩ := hwrite e
        exact ⟨hv, ha, fun x pre' v' hx => absurd hx (hdead x pre' v'),
          fun x pre' v' k hx => absurd hx (hdeadC x pre' v' k),
          fun x pre' v' hx => absurd hx (hdead x pre' v'),
          fun x pre' v' k hx => absurd hx (hdeadC x pre' v' k), h.safe, hm, h.credNodup, hc⟩
  · push_neg at hhold
    rw [step_submit_not_holding _ hhold]; exact h

theorem inv_effect (h : Inv S) (q : ℕ) : Inv (step .reference S (.effect q)) := by
  by_cases hc : ∃ pre v k, S.procs q = .committed pre v k
  · obtain ⟨pre, v, k, hq⟩ := hc
    obtain ⟨hlog, htorn, hload, hauth, hacked⟩ := h.committed q pre v k hq
    have hcs : InCS S q := Or.inr ⟨_, _, _, hq⟩
    obtain ⟨hidle, hidleC⟩ := none_inCS_after h hcs .idle idle_out
    rw [step_effect_committed _ hq]
    refine ⟨h.valid, h.anchor, fun x pre' v' hx => absurd hx (hidle x pre' v'),
      fun x pre' v' k' hx => absurd hx (hidleC x pre' v' k'),
      fun x pre' v' hx => absurd hx (hidle x pre' v'),
      fun x pre' v' k' hx => absurd hx (hidleC x pre' v' k'), ?_, ?_, h.credNodup, h.credPast⟩
    · intro d hd
      rcases List.mem_append.1 hd with hd | hd
      · exact h.safe d hd
      · rw [List.mem_singleton] at hd; subst hd
        refine ⟨?_, hlog⟩
        show specPermits (credOK S.issued) S.disk.log k = true
        rw [hlog, specPermits_append_execute _ (load_ne_nil hload), ← authorizes_iff_spec hload]
        exact hauth
    · show List.Sublist (S.acked ++ [.execute k]) S.disk.log
      rw [hlog]; exact hacked.append (List.Sublist.refl _)
  · push_neg at hc
    rw [step_effect_not_committed _ hc]; exact h

theorem inv_crash (h : Inv S) (q : ℕ) : Inv (step .reference S (.crash q)) := by
  simp only [step]
  refine ⟨h.valid, h.anchor, ?_, ?_, ?_, ?_, h.safe, h.mem, h.credNodup, h.credPast⟩
  · intro x pre v hx
    by_cases hxq : x = q
    · subst hxq; simp [Sys.set_self] at hx
    · simp only [Sys.set_ne hxq] at hx
      simpa [release, h.owner x pre v hx] using hxq
  · intro x pre v k hx
    by_cases hxq : x = q
    · subst hxq; simp [Sys.set_self] at hx
    · simp only [Sys.set_ne hxq] at hx
      simpa [release, h.ownerC x pre v k hx] using hxq
  · intro x pre v hx
    by_cases hxq : x = q
    · subst hxq; simp [Sys.set_self] at hx
    · simp only [Sys.set_ne hxq] at hx; exact h.fresh x pre v hx
  · intro x pre v k hx
    by_cases hxq : x = q
    · subst hxq; simp [Sys.set_self] at hx
    · simp only [Sys.set_ne hxq] at hx; exact h.committed x pre v k hx

theorem inv_restart (h : Inv S) (q : ℕ) : Inv (step .reference S (.restart q)) := by
  cases hq : S.procs q with
  | dead =>
    simp only [step, hq]
    refine ⟨h.valid, h.anchor, ?_, ?_, ?_, ?_, h.safe, h.mem, h.credNodup, h.credPast⟩
    · intro x pre v hx
      by_cases hxq : x = q
      · subst hxq; simp [Sys.set_self] at hx
      · simp only [Sys.set_ne hxq] at hx; exact h.owner x pre v hx
    · intro x pre v k hx
      by_cases hxq : x = q
      · subst hxq; simp [Sys.set_self] at hx
      · simp only [Sys.set_ne hxq] at hx; exact h.ownerC x pre v k hx
    · intro x pre v hx
      by_cases hxq : x = q
      · subst hxq; simp [Sys.set_self] at hx
      · simp only [Sys.set_ne hxq] at hx; exact h.fresh x pre v hx
    · intro x pre v k hx
      by_cases hxq : x = q
      · subst hxq; simp [Sys.set_self] at hx
      · simp only [Sys.set_ne hxq] at hx; exact h.committed x pre v k hx
  | _ => simp only [step, hq]; exact h

theorem inv_issue (h : Inv S) (c : ℕ) : Inv (step .reference S (.issue c)) := by
  simp only [step]
  cases hc : credAny S.issued c with
  | true => simpa using h
  | false =>
    simp only [Bool.false_eq_true, if_false]
    have hm := @credOK_mono S.issued (c, S.disk.log)
    refine ⟨?_, h.anchor, h.owner, h.ownerC, ?_, ?_, h.safe, h.mem, ?_, ?_⟩
    · intro ht
      rcases h.valid ht with hl | ⟨v, hv⟩
      · exact Or.inl hl
      · exact Or.inr ⟨v, load_mono hm hv⟩
    · intro x pre v hx
      obtain ⟨h1, h2, h3⟩ := h.fresh x pre v hx
      exact ⟨h1, h2, load_mono hm h3⟩
    · intro x pre v k hx
      obtain ⟨h1, h2, h3, h4, h5⟩ := h.committed x pre v k hx
      exact ⟨h1, h2, load_mono hm h3, h4, h5⟩
    · rw [List.map_append, List.nodup_append]
      refine ⟨h.credNodup, List.nodup_singleton _, ?_⟩
      intro a ha b hb hab
      simp only [List.map_cons, List.map_nil, List.mem_singleton] at hb
      subst hb; subst hab
      exact (credAny_false.1 hc) ha
    · intro x hx
      rcases List.mem_append.1 hx with hx | hx
      · exact h.credPast x hx
      · rw [List.mem_singleton] at hx; subst hx; exact List.prefix_refl _

theorem inv_anchorNow (h : Inv S) : Inv (step .reference S .anchorNow) := by
  simp only [step]
  exact ⟨h.valid, List.prefix_refl _, h.owner, h.ownerC, h.fresh, h.committed, h.safe, h.mem,
    h.credNodup, h.credPast⟩

/-- **Inductive step.**  Every in-boundary event of the reference discipline preserves `Inv`. -/
theorem inv_step (h : Inv S) {e : Event} (he : e.inBoundary = true) :
    Inv (step .reference S e) := by
  cases e with
  | provision a b p => exact inv_provision h a b p
  | begin q => exact inv_begin h q
  | submit q r o => exact inv_submit h q r o
  | effect q => exact inv_effect h q
  | crash q => exact inv_crash h q
  | restart q => exact inv_restart h q
  | issue c => exact inv_issue h c
  | anchorNow => exact inv_anchorNow h
  | directEffect _ | rollback _ | tamper _ => simp [Event.inBoundary] at he

theorem inv_foldl : ∀ (es : List Event) (S : Sys), Inv S → (∀ e ∈ es, e.inBoundary = true) →
    Inv (es.foldl (step .reference) S)
  | [], _, h, _ => h
  | e :: es, S, h, hb => by
    simp only [List.foldl_cons]
    exact inv_foldl es _ (inv_step h (hb e (List.mem_cons_self))) fun x hx =>
      hb x (List.mem_cons_of_mem _ hx)

/-- **Every in-boundary run of the reference kernel satisfies `Inv`.** -/
theorem inv_run {es : List Event} (hb : ∀ e ∈ es, e.inBoundary = true) :
    Inv (run .reference es) :=
  inv_foldl es _ inv_init hb

end Preservation

/-! ## Theorems A–C -/

section Memory

/-- **Theorem A (state validity).**  In every in-boundary run of the reference kernel the durable
memory is empty (not yet provisioned), torn by a crash (every process then fails closed), or loads;
every process inside the critical section holds the view folded from the durable log; and a
committed intent extends exactly the memory its decision was made on. -/
theorem state_validity {es : List Event} (hb : ∀ e ∈ es, e.inBoundary = true) :
    let S := run .reference es
    (S.disk.torn = false → S.disk.log = [] ∨ ∃ v, load (credOK S.issued) S.disk.log = some v) ∧
      (∀ q pre v, S.procs q = .holding pre v →
        pre = S.disk.log ∧ load (credOK S.issued) pre = some v) ∧
      ∀ q pre v k, S.procs q = .committed pre v k →
        S.disk.log = pre ++ [.execute k] ∧ load (credOK S.issued) pre = some v := by
  have h := inv_run hb
  exact ⟨h.valid, fun q pre v hq => ⟨(h.fresh q pre v hq).1, (h.fresh q pre v hq).2.2⟩,
    fun q pre v k hq => ⟨(h.committed q pre v k hq).1, (h.committed q pre v k hq).2.2.1⟩⟩

theorem begin_keeps (D : Discipline) (S : Sys) (q : ℕ) :
    (step D S (.begin q)).disk = S.disk ∧ (step D S (.begin q)).effects = S.effects ∧
      (step D S (.begin q)).acked = S.acked := by
  simp only [step]
  split
  · split_ifs <;> (try exact ⟨rfl, rfl, rfl⟩) <;> split <;> exact ⟨rfl, rfl, rfl⟩
  · exact ⟨rfl, rfl, rfl⟩

/-- In-boundary steps of every discipline only append to the durable log: no in-boundary
transition deletes, compacts or rewrites a safety record. -/
theorem log_prefix_step (D : Discipline) (S : Sys) {e : Event} (he : e.inBoundary = true) :
    S.disk.log <+: (step D S e).disk.log := by
  cases e with
  | provision a b p =>
    simp only [step]; split_ifs with hc
    · rw [hc.1]; exact List.nil_prefix
    · exact List.prefix_refl _
  | begin q => rw [(begin_keeps D S q).1]
  | submit q r o =>
    cases hq : S.procs q with
    | holding pre v =>
      cases hg : guard D S.issued pre v r with
      | false => simp [step, hq, hg]
      | true =>
        cases o with
        | ok =>
          cases r <;> simp only [step, hq, hg, ↓reduceIte] <;> split_ifs <;>
            first | exact disk_write_prefix _ _ _ | exact List.prefix_refl _
        | fail e | crash e =>
          simp only [step, hq, hg, ↓reduceIte]
          split_ifs
          · exact disk_write_prefix _ _ _
          · exact List.prefix_refl _
    | idle => simp [step, hq]
    | dead => simp [step, hq]
    | committed pre v k => simp [step, hq]
  | effect q => cases hq : S.procs q <;> simp [step, hq]
  | crash q => simp [step]
  | restart q => cases hq : S.procs q <;> simp [step, hq]
  | issue c => simp only [step]; split_ifs <;> exact List.prefix_refl _
  | anchorNow => simp [step]
  | directEffect _ | rollback _ | tamper _ => simp [Event.inBoundary] at he

/-- **Theorem B (safety-memory preservation).**  In every in-boundary run of the reference kernel,
every record acknowledged to its caller is in the durable log, in acknowledgment order. -/
theorem safety_memory_preserved {es : List Event} (hb : ∀ e ∈ es, e.inBoundary = true) :
    List.Sublist (run .reference es).acked (run .reference es).disk.log :=
  (inv_run hb).mem

/-- **Theorem C (restart consistency).**  When a crashed process restarts, it recovers the view
folded from the durable log; that log contains every acknowledged record in order; and the
recovered latch is exactly the specification's hazard over the durable log. -/
theorem restart_consistency {S : Sys} (h : Inv S) {q : ℕ} (hq : S.procs q = .dead)
    (ht : S.disk.torn = false) (hne : S.disk.log ≠ []) :
    ∃ v, (step .reference S (.restart q)).cache q = some (S.disk.log, v) ∧
      load (credOK S.issued) S.disk.log = some v ∧ List.Sublist S.acked S.disk.log ∧
      ∃ s, specOf (credOK S.issued) S.disk.log = some s ∧ s.hazard = v.locked := by
  obtain ⟨v, hv⟩ := (h.valid ht).resolve_left hne
  refine ⟨v, ?_, hv, h.mem, ?_⟩
  · simp [step, hq, Sys.setCache, Sys.fresh, Disk.view, ht,
      List.isPrefixOf_iff_prefix.2 h.anchor, hv]
  · obtain ⟨s, hs, hr⟩ := load_refines_spec hv
    exact ⟨s, hs, hr.2.2.2.2.2⟩

end Memory

/-! ## Theorem D: replay -/

section Replay

variable {vf : CredOK}

theorem fold_seq : ∀ (pre : List Rec) (v : View) (rs : List Rec) (w : View),
    fold vf pre v rs = some w → (∀ s xs, Rec.observe s xs ∈ pre → s ≤ v.lastSeq) →
    ∀ s xs, Rec.observe s xs ∈ pre ++ rs → s ≤ w.lastSeq
  | _, _, [], _, hf, hp => by simp only [fold, Option.some.injEq] at hf; subst hf; simpa using hp
  | pre, v, r :: rs, w, hf, hp => by
    simp only [fold] at hf
    split_ifs at hf with ha
    have key : ∀ s xs, Rec.observe s xs ∈ pre ++ [r] → s ≤ (next v r).lastSeq := by
      intro s xs hm
      rcases List.mem_append.1 hm with hm | hm
      · have := hp s xs hm
        cases r with
        | observe s' xs' =>
          simp only [admissible, Bool.and_eq_true, decide_eq_true_eq] at ha
          simp only [next]; omega
        | _ => simpa [next] using this
      · rw [List.mem_singleton] at hm; subst hm; simp [next]
    have := fold_seq (pre ++ [r]) (next v r) rs w hf key
    simpa [List.append_assoc] using this

theorem fold_version : ∀ (pre : List Rec) (v : View) (rs : List Rec) (w : View),
    fold vf pre v rs = some w → (∀ p, Rec.policy p ∈ pre → p.version ≤ v.policy.version) →
    ∀ p, Rec.policy p ∈ pre ++ rs → p.version ≤ w.policy.version
  | _, _, [], _, hf, hp => by simp only [fold, Option.some.injEq] at hf; subst hf; simpa using hp
  | pre, v, r :: rs, w, hf, hp => by
    simp only [fold] at hf
    split_ifs at hf with ha
    have key : ∀ p, Rec.policy p ∈ pre ++ [r] → p.version ≤ (next v r).policy.version := by
      intro p hm
      rcases List.mem_append.1 hm with hm | hm
      · have := hp p hm
        cases r with
        | policy p' =>
          simp only [admissible, Policy.successor, Bool.and_eq_true, beq_iff_eq] at ha
          simp only [next]; omega
        | _ => simpa [next] using this
      · rw [List.mem_singleton] at hm; subst hm; simp [next]
    have := fold_version (pre ++ [r]) (next v r) rs w hf key
    simpa [List.append_assoc] using this

theorem fold_reset : ∀ (pre : List Rec) (v : View) (rs : List Rec) (w : View),
    fold vf pre v rs = some w →
    (∀ c, Rec.reset c ∈ pre → ∃ p', p' ++ [Rec.reset c] <+: pre ∧ vf c p' = true) →
    ∀ c, Rec.reset c ∈ pre ++ rs → ∃ p', p' ++ [Rec.reset c] <+: pre ++ rs ∧ vf c p' = true
  | _, _, [], _, _, hp => by simpa using hp
  | pre, v, r :: rs, w, hf, hp => by
    simp only [fold] at hf
    split_ifs at hf with ha
    have key : ∀ c, Rec.reset c ∈ pre ++ [r] →
        ∃ p', p' ++ [Rec.reset c] <+: pre ++ [r] ∧ vf c p' = true := by
      intro c hm
      rcases List.mem_append.1 hm with hm | hm
      · obtain ⟨p', hp', hv⟩ := hp c hm
        exact ⟨p', hp'.trans (List.prefix_append _ _), hv⟩
      · rw [List.mem_singleton] at hm; subst hm
        simp only [admissible, Bool.and_eq_true] at ha
        exact ⟨pre, List.prefix_refl _, ha.2⟩
    have := fold_reset (pre ++ [r]) (next v r) rs w hf key
    simpa [List.append_assoc] using this

theorem load_facts {L : List Rec} {v : View} (h : load vf L = some v) :
    (∀ s xs, Rec.observe s xs ∈ L → s ≤ v.lastSeq) ∧
    (∀ p, Rec.policy p ∈ L → p.version ≤ v.policy.version) ∧
    (∀ c, Rec.reset c ∈ L → ∃ p', p' ++ [Rec.reset c] <+: L ∧ vf c p' = true) := by
  cases L with
  | nil => simp [load] at h
  | cons g rs =>
    simp only [load] at h
    cases hg : genesisView g with
    | none => simp [hg] at h
    | some v0 =>
      simp only [hg, Option.bind_some] at h
      have hgen : ∃ a b p, g = .genesis a b p := by
        cases g <;> simp [genesisView] at hg; exact ⟨_, _, _, rfl⟩
      obtain ⟨a, b, p, rfl⟩ := hgen
      refine ⟨?_, ?_, ?_⟩
      · simpa using fold_seq [_] v0 rs v h (by simp)
      · simpa using fold_version [_] v0 rs v h (by simp)
      · simpa using fold_reset [_] v0 rs v h (by simp)

/-- **Theorem D (replay).**  Let the credentials be issued under distinct names.  A copy of any
record already in a loadable log is refused — a replayed observation is stale, a replayed policy
is not a successor version, a replayed reset carries a credential bound to an older prefix, and a
genesis record is never admissible — with one exception: a replayed action request, which changes
no state and is judged by the guard on the current view like any new request.  In particular no
replay can clear the latch or relax the policy. -/
theorem replay_is_inert {I : Issued} (hI : (I.map Prod.fst).Nodup) {L : List Rec} {v : View}
    (hL : load (credOK I) L = some v) {r : Rec} (hr : r ∈ L)
    (ha : admissible (credOK I) L v r = true) : ∃ k, r = .execute k ∧ next v r = v := by
  obtain ⟨hseq, hver, hres⟩ := load_facts hL
  cases r with
  | genesis => simp [admissible] at ha
  | observe s xs =>
    have := hseq s xs hr
    simp only [admissible, Bool.and_eq_true, decide_eq_true_eq] at ha; omega
  | policy p =>
    have := hver p hr
    simp only [admissible, Policy.successor, Bool.and_eq_true, beq_iff_eq] at ha; omega
  | execute k => exact ⟨k, rfl, rfl⟩
  | reset c =>
    obtain ⟨p', hp', hv'⟩ := hres c hr
    simp only [admissible, Bool.and_eq_true, credOK, decide_eq_true_eq] at ha hv'
    have := List.inj_on_of_nodup_map hI ha.2 hv' rfl
    simp only [Prod.mk.injEq, true_and] at this
    have hlen := hp'.length_le
    rw [← this] at hlen; simp at hlen

/-- **Credential binding.**  In every in-boundary run of the reference kernel, every reset record
of the durable log carries a credential the operator issued for exactly the prefix before it. -/
theorem credential_bound {es : List Event} (hb : ∀ e ∈ es, e.inBoundary = true) (c : ℕ)
    (hc : Rec.reset c ∈ (run .reference es).disk.log)
    (ht : (run .reference es).disk.torn = false) :
    ∃ p', p' ++ [Rec.reset c] <+: (run .reference es).disk.log ∧
      (c, p') ∈ (run .reference es).issued := by
  have h := inv_run hb
  rcases h.valid ht with hl | ⟨v, hv⟩
  · rw [hl] at hc; simp at hc
  · obtain ⟨p', hp', hv'⟩ := (load_facts hv).2.2 c hc
    exact ⟨p', hp', by simpa [credOK] using hv'⟩

end Replay

/-! ## Authorization before execution, blocking, and preservation -/

section Intervention

/-- **Where effects come from (every discipline).**  A step changes the effect history only by
running the effect of a committed intent, or by the out-of-boundary `directEffect`. -/
theorem effect_step_source (D : Discipline) (S : Sys) (e : Event)
    (he : (step D S e).effects ≠ S.effects) :
    (∃ k, e = .directEffect k) ∨
      ∃ q pre v k, e = .effect q ∧ S.procs q = .committed pre v k ∧
        (step D S e).effects = S.effects ++ [⟨k, pre, S.disk.log, S.issued⟩] := by
  cases e with
  | directEffect k => exact Or.inl ⟨k, rfl⟩
  | effect q =>
    right
    cases hq : S.procs q with
    | committed pre v k => exact ⟨q, pre, v, k, rfl, hq, by simp [step, hq]⟩
    | _ => simp [step, hq] at he
  | submit q r o =>
    exfalso; apply he
    cases hq : S.procs q with
    | holding pre v =>
      cases hg : guard D S.issued pre v r with
      | false => simp [step, hq, hg]
      | true => cases o <;> (try cases r) <;> simp [step, hq, hg]
    | _ => simp [step, hq]
  | provision a b p => simp only [step] at he; split_ifs at he <;> simp at he
  | begin q => exact absurd (begin_keeps D S q).2.1 he
  | crash q => simp [step] at he
  | restart q => cases hq : S.procs q <;> simp [step, hq] at he
  | issue c => simp only [step] at he; split_ifs at he <;> simp at he
  | anchorNow => simp [step] at he
  | rollback n => simp [step] at he
  | tamper L => simp [step] at he

/-- **Where committed intents come from (every discipline).**  A process becomes committed to
`execute k` only by a successful `submit` of `execute k` that the writer's guard admitted on its
view. -/
theorem step_enters_committed (D : Discipline) (S : Sys) (e : Event) {q : ℕ} {pre : List Rec}
    {v : View} {k : ℕ} (h' : (step D S e).procs q = .committed pre v k)
    (h : S.procs q ≠ .committed pre v k) :
    e = .submit q (.execute k) .ok ∧ S.procs q = .holding pre v ∧
      guard D S.issued pre v (.execute k) = true := by
  have hset : ∀ (x : ℕ) (ph : Phase), S.set x ph q = .committed pre v k →
      x = q ∧ ph = .committed pre v k := by
    intro x ph hx
    by_cases hxq : q = x
    · subst hxq; rw [Sys.set_self] at hx; exact ⟨rfl, hx⟩
    · rw [Sys.set_ne hxq] at hx; exact absurd hx h
  cases e with
  | submit x r o =>
    cases hx : S.procs x with
    | holding pre' v' =>
      cases hg : guard D S.issued pre' v' r with
      | false =>
        simp only [step, hx, hg, Bool.false_eq_true, ↓reduceIte] at h'
        obtain ⟨-, hc⟩ := hset _ _ h'; cases hc
      | true =>
        cases o with
        | ok =>
          cases r with
          | execute k' =>
            simp only [step, hx, hg, ↓reduceIte] at h'
            obtain ⟨rfl, hc⟩ := hset _ _ h'
            simp only [Phase.committed.injEq] at hc
            obtain ⟨rfl, rfl, rfl⟩ := hc
            exact ⟨rfl, hx, hg⟩
          | _ =>
            simp only [step, hx, hg, ↓reduceIte] at h'
            obtain ⟨-, hc⟩ := hset _ _ h'; cases hc
        | fail _ | crash _ =>
          simp only [step, hx, hg, ↓reduceIte] at h'
          obtain ⟨-, hc⟩ := hset _ _ h'; cases hc
    | _ => simp only [step, hx] at h'; exact absurd h' h
  | begin x =>
    simp only [step] at h'
    split at h'
    · split_ifs at h'
      all_goals first
        | exact absurd h' h
        | (split at h'
           · obtain ⟨-, hc⟩ := hset _ _ h'; cases hc
           · exact absurd h' h)
    · exact absurd h' h
  | effect x =>
    simp only [step] at h'
    split at h'
    · obtain ⟨-, hc⟩ := hset _ _ h'; cases hc
    · exact absurd h' h
  | crash x => simp only [step] at h'; obtain ⟨-, hc⟩ := hset _ _ h'; cases hc
  | restart x =>
    simp only [step] at h'
    split at h'
    · obtain ⟨-, hc⟩ := hset _ _ h'; cases hc
    · exact absurd h' h
  | provision a b p => simp only [step] at h'; split_ifs at h' <;> exact absurd h' h
  | issue c => simp only [step] at h'; split_ifs at h' <;> exact absurd h' h
  | anchorNow | directEffect _ | rollback _ | tamper _ => simp only [step] at h'; exact absurd h' h

/-- The state before event `i` of a trace. -/
abbrev at_ (D : Discipline) (T : List Event) (i : ℕ) : Sys := run D (T.take i)

theorem at_succ (D : Discipline) (T : List Event) {i : ℕ} (hi : i < T.length) :
    at_ D T (i + 1) = step D (at_ D T i) T[i] := by
  simp only [at_, run, List.take_succ, List.getElem?_eq_getElem hi, Option.toList_some,
    List.foldl_append, List.foldl_cons, List.foldl_nil]

theorem inBoundary_take {T : List Event} (hb : ∀ e ∈ T, e.inBoundary = true) (i : ℕ) :
    ∀ e ∈ T.take i, e.inBoundary = true := fun e he => hb e (List.mem_of_mem_take he)

/-- A committed phase at step `m` was entered at an earlier step by a guarded `submit`, and the
process stayed committed since. -/
theorem committed_origin (D : Discipline) (T : List Event) {q : ℕ} {pre : List Rec} {v : View}
    {k : ℕ} : ∀ m, m ≤ T.length → (at_ D T m).procs q = .committed pre v k →
      ∃ i < m, ∃ hi : i < T.length, T[i] = .submit q (.execute k) .ok ∧
        (at_ D T i).procs q = .holding pre v ∧
        guard D (at_ D T i).issued pre v (.execute k) = true ∧
        ∀ m', i < m' → m' ≤ m → (at_ D T m').procs q = .committed pre v k
  | 0, _, h => by simp [at_, run, Sys.init] at h
  | m + 1, hm, h => by
    have hm' : m < T.length := hm
    by_cases hprev : (at_ D T m).procs q = .committed pre v k
    · obtain ⟨i, hi, hiT, hev, hhold, hg, hstay⟩ := committed_origin D T m hm'.le hprev
      refine ⟨i, by omega, hiT, hev, hhold, hg, fun m' h1 h2 => ?_⟩
      rcases Nat.lt_or_ge m' (m + 1) with hlt | hge
      · exact hstay m' h1 (by omega)
      · rw [show m' = m + 1 by omega]; exact h
    · rw [at_succ D T hm'] at h
      obtain ⟨hev, hhold, hg⟩ := step_enters_committed D _ _ h hprev
      refine ⟨m, by omega, hm', hev, hhold, hg, fun m' h1 h2 => ?_⟩
      rw [show m' = m + 1 by omega, at_succ D T hm']; exact h

/-- **Authorization before execution.**  In every in-boundary run of the reference kernel, every
step `j` that performs an effect is the `effect` event of a process `q` that, at a strictly earlier
step `i`:
* submitted `execute k`,
* held the critical section with the view `v` folded from the then-current durable memory `pre`,
* had the guard accept `k` on `v`, so that the declarative specification permitted `k` on `pre`;

and from step `i + 1` to step `j` the process stayed committed to that decision while the durable
memory was exactly `pre` followed by the intent — no other record was written in between, so the
premises of the decision still held when the effect ran. -/
theorem authorized_before_execution {T : List Event} (hb : ∀ e ∈ T, e.inBoundary = true)
    {j : ℕ} (hj : j < T.length)
    (hexec : (at_ .reference T (j + 1)).effects ≠ (at_ .reference T j).effects) :
    ∃ i < j, ∃ q pre v k, ∃ hi : i < T.length,
      T[i] = .submit q (.execute k) .ok ∧
      (at_ .reference T i).procs q = .holding pre v ∧
      pre = (at_ .reference T i).disk.log ∧
      load (credOK (at_ .reference T i).issued) pre = some v ∧
      v.authorizes k = true ∧
      specPermits (credOK (at_ .reference T i).issued) pre k = true ∧
      T[j] = .effect q ∧
      (at_ .reference T (j + 1)).effects =
        (at_ .reference T j).effects ++
          [⟨k, pre, pre ++ [.execute k], (at_ .reference T j).issued⟩] ∧
      ∀ m, i < m → m ≤ j →
        (at_ .reference T m).procs q = .committed pre v k ∧
        (at_ .reference T m).disk.log = pre ++ [.execute k] := by
  rw [at_succ _ _ hj] at hexec
  rcases effect_step_source _ _ _ hexec with ⟨k, hk⟩ | ⟨q, pre, v, k, hev, hq, heff⟩
  · exact absurd (hb _ (List.getElem_mem hj)) (by rw [hk]; simp [Event.inBoundary])
  obtain ⟨i, hij, hiT, hsub, hhold, hg, hstay⟩ := committed_origin _ T j hj.le hq
  have hIi := inv_run (inBoundary_take hb i)
  obtain ⟨hpre, -, hload⟩ := hIi.fresh q pre v hhold
  have hauth : v.authorizes k = true := by
    rw [guard_reference, hpre] at hg; simpa [admissible] using hg
  have hIj := inv_run (inBoundary_take hb j)
  refine ⟨i, hij, q, pre, v, k, hiT, hsub, hhold, hpre, hload, hauth,
    (authorizes_iff_spec hload k).symm.trans hauth, hev, ?_, ?_⟩
  · rw [at_succ _ _ hj, heff, (hIj.committed q pre v k hq).1]
  · intro m h1 h2
    have hc := hstay m h1 h2
    exact ⟨hc, (inv_run (inBoundary_take hb m)).committed q pre v k hc |>.1⟩

/-- **Pre-action intervention (blocking the request).**  In any state satisfying `Inv` (every
in-boundary reference run), an `execute` request that the declarative specification does not
permit on the current durable memory is refused, whatever the write outcome: no record is written,
nothing is acknowledged, no intent is committed and no effect happens. -/
theorem pre_action_intervention {S : Sys} (h : Inv S) (q k : ℕ) (o : WriteOutcome)
    (hnot : specPermits (credOK S.issued) S.disk.log k = false) :
    step .reference S (.submit q (.execute k) o) = S ∨
      step .reference S (.submit q (.execute k) o) =
        { S with procs := S.set q .idle, lock := release q S.lock } := by
  by_cases hhold : ∃ pre v, S.procs q = .holding pre v
  · obtain ⟨pre, v, hq⟩ := hhold
    obtain ⟨rfl, -, hload⟩ := h.fresh q pre v hq
    have hg : admissible (credOK S.issued) S.disk.log v (.execute k) = false := by
      simp only [admissible]; rw [authorizes_iff_spec hload]; exact hnot
    exact Or.inr (step_submit_refused hq hg)
  · push_neg at hhold
    exact Or.inl (step_submit_not_holding _ hhold)

/-- **Blocking the effect.**  In any state satisfying `Inv`, no in-boundary step performs an effect
of a kind that the declarative specification does not permit on the current durable memory. -/
theorem effect_blocked {S : Sys} (h : Inv S) {e : Event} (he : e.inBoundary = true) (k : ℕ)
    (hnot : specPermits (credOK S.issued) S.disk.log k = false) :
    ∀ d ∈ (step .reference S e).effects, d ∉ S.effects → d.kind ≠ k := by
  intro d hd hnew
  by_cases hch : (step .reference S e).effects = S.effects
  · rw [hch] at hd; exact absurd hd hnew
  rcases effect_step_source _ _ _ hch with ⟨k', rfl⟩ | ⟨q, pre, v, k', rfl, hq, heff⟩
  · simp [Event.inBoundary] at he
  rw [heff] at hd
  rcases List.mem_append.1 hd with hd | hd
  · exact absurd hd hnew
  rw [List.mem_singleton] at hd; subst hd
  obtain ⟨hlog, -, hload, hauth, -⟩ := h.committed q pre v k' hq
  intro hk; subst hk
  rw [hlog, specPermits_append_execute _ (load_ne_nil hload), ← authorizes_iff_spec hload,
    hauth] at hnot
  cases hnot

/-- The safety predicate of a system state: every effect was permitted by the specification on the
durable memory when it ran, and no acknowledged record is missing from the durable log. -/
def Safe (S : Sys) : Prop :=
  (∀ d ∈ S.effects, specPermits (credOK d.issued) d.log d.kind = true) ∧
    List.Sublist S.acked S.disk.log

theorem safe_init : Safe Sys.init := ⟨by simp [Sys.init], inv_init.mem⟩

/-- **Inductive safety preservation.**  `Safe` holds initially and after every in-boundary run of
the reference kernel (any number of processes and interleavings, any write failures and crashes,
restarts, operator credentials and anchors). -/
theorem safe_run {es : List Event} (hb : ∀ e ∈ es, e.inBoundary = true) :
    Safe (run .reference es) :=
  ⟨fun d hd => ((inv_run hb).safe d hd).1, (inv_run hb).mem⟩

end Intervention

end RCA
end WaveLockProofs
