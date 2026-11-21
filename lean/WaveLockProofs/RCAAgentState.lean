import Mathlib.Tactic

/-!
# RCA reference safety kernel: typed agent state and persistent safety memory (issue #42)

**Status: `PROPOSED_MODEL`.**  At `rickyjreyes/Wavelock` `c3cb0d1` no RCA agent state, safety
memory, drift measurement or pre-action guard for agents existed (`docs/RCA_FORMALIZATION.md`,
Phase 0).  This module models the *proposed reference mechanism* `wavelock/rca_reference`, which
was written together with it; nothing here verifies preexisting production behaviour.

## Correspondence (`wavelock/rca_reference/memory.py`)

| Lean | Python |
|---|---|
| `Policy`, `Policy.valid`, `Policy.successor` | `Policy`, `Policy.from_json`, `successor` |
| `Rec` | one canonical JSON line (`check_record`); strings are encoded as naturals |
| `View` | `View` without `length`/`head`: the prefix itself plays their role |
| `View.authorizes` | `View.authorizes` (the pre-action guard) |
| `admissible` | `admissible` |
| `next` | `next_view` |
| `load` | `load` (hash chain and encoding checks are abstracted: a log *is* its record list) |
| `CredOK` | `verifier(agent, index, head, credential)`; `(index, head)` names the prefix |

`l1` is `drift.l1` (sum of absolute coordinate differences); it is only applied to vectors of
the baseline's dimension.

## Declarative specification

`specOf` is a second fold over the *same* records that encodes the intended policy semantics
without trusting the kernel's own checks: policies only ever add restrictions (minimum threshold,
union of prohibited kinds), replayed or out-of-order observations carry no information, the
hazard is sticky, and only a reset with a credential issued for exactly its prefix, made while the
last fresh deviation is below the threshold, clears it.  `specPermits L k` is the safety
condition an action `k` must meet after the log `L`.  `load_refines_spec` proves that on every log
the kernel's loader accepts, the operational view and the specification agree, so the guard
`View.authorizes` decides exactly `specPermits`.
-/

namespace WaveLockProofs
namespace RCA

/-! ## Typed state -/

/-- Sum of absolute coordinate differences (`drift.l1`). -/
def l1 : List ℤ → List ℤ → ℕ
  | a :: u, b :: v => (a - b).natAbs + l1 u v
  | _, _ => 0

/-- Policy in force: latch threshold on the anchored deviation, and prohibited action kinds. -/
structure Policy where
  version : ℕ
  threshold : ℕ
  prohibited : List ℕ
  deriving DecidableEq, Repr

/-- `Policy.from_json`: the threshold is at least 1. -/
def Policy.valid (p : Policy) : Bool := decide (1 ≤ p.threshold)

/-- `successor`: next version, threshold not larger, no prohibited kind dropped. -/
def Policy.successor (new old : Policy) : Bool :=
  new.version == old.version + 1 && decide (new.threshold ≤ old.threshold) &&
    old.prohibited.all fun k => decide (k ∈ new.prohibited)

/-- One safety-memory record.  Agent ids, action kinds and credentials are encoded as naturals. -/
inductive Rec where
  | genesis (agent : ℕ) (baseline : List ℤ) (p : Policy)
  | observe (seq : ℕ) (values : List ℤ)
  | policy (p : Policy)
  | execute (k : ℕ)
  | reset (cred : ℕ)
  deriving DecidableEq, Repr

/-- State derived from a memory prefix. -/
structure View where
  agent : ℕ
  baseline : List ℤ
  policy : Policy
  lastSeq : ℕ
  lastDev : ℕ
  locked : Bool
  deriving DecidableEq, Repr

/-- Operator-credential check: credential `c` is valid for exactly the memory prefix `pre`. -/
abbrev CredOK := ℕ → List Rec → Bool

/-- The pre-action guard: not locked, and the action kind is not prohibited. -/
def View.authorizes (v : View) (k : ℕ) : Bool :=
  !v.locked && !decide (k ∈ v.policy.prohibited)

/-! ## Memory semantics -/

/-- Whether record `r` may follow the prefix `pre`, summarised by `v`. -/
def admissible (vf : CredOK) (pre : List Rec) (v : View) : Rec → Bool
  | .genesis .. => false
  | .observe s xs => decide (v.lastSeq < s) && xs.length == v.baseline.length
  | .policy p => p.valid && p.successor v.policy
  | .execute k => v.authorizes k
  | .reset c => v.locked && decide (v.lastDev < v.policy.threshold) && vf c pre

/-- The view after a record (`next_view`); the latch is sticky. -/
def next (v : View) : Rec → View
  | .observe s xs =>
    { v with lastSeq := s, lastDev := l1 xs v.baseline,
             locked := v.locked || decide (v.policy.threshold ≤ l1 xs v.baseline) }
  | .policy p => { v with policy := p, locked := v.locked || decide (p.threshold ≤ v.lastDev) }
  | .reset _ => { v with locked := false }
  | .execute _ => v
  | .genesis .. => v

/-- The view of a genesis record. -/
def genesisView : Rec → Option View
  | .genesis a b p => if p.valid && !b.isEmpty then some ⟨a, b, p, 0, 0, false⟩ else none
  | _ => none

/-- Fold the records after the genesis record, failing closed on an inadmissible one. -/
def fold (vf : CredOK) : List Rec → View → List Rec → Option View
  | _, v, [] => some v
  | pre, v, r :: rs => if admissible vf pre v r then fold vf (pre ++ [r]) (next v r) rs else none

/-- `load`: an empty memory, a non-genesis first record or any inadmissible record fails closed. -/
def load (vf : CredOK) : List Rec → Option View
  | [] => none
  | g :: rs => (genesisView g).bind fun v => fold vf [g] v rs

/-! ## Declarative specification of the intended policy semantics -/

/-- Specification state: accumulated restrictions, freshest sequence number, last fresh
deviation and the sticky hazard. -/
structure Spec where
  baseline : List ℤ
  thr : ℕ
  proh : List ℕ
  maxSeq : ℕ
  lastDev : ℕ
  hazard : Bool
  deriving DecidableEq, Repr

/-- One record under the specification.  Nothing is trusted: weakening policies add nothing,
stale observations are ignored, and an unbound or premature reset does not clear the hazard. -/
def specStep (vf : CredOK) (pre : List Rec) (s : Spec) : Rec → Spec
  | .observe q xs =>
    if s.maxSeq < q ∧ xs.length = s.baseline.length then
      { s with maxSeq := q, lastDev := l1 xs s.baseline,
               hazard := s.hazard || decide (s.thr ≤ l1 xs s.baseline) }
    else s
  | .policy p =>
    { s with thr := min s.thr p.threshold, proh := s.proh ++ p.prohibited,
             hazard := s.hazard || decide (min s.thr p.threshold ≤ s.lastDev) }
  | .reset c => if s.hazard ∧ s.lastDev < s.thr ∧ vf c pre = true then { s with hazard := false } else s
  | .execute _ => s
  | .genesis .. => s

def specInit : Rec → Option Spec
  | .genesis _ b p => if p.valid && !b.isEmpty then some ⟨b, p.threshold, p.prohibited, 0, 0, false⟩
                      else none
  | _ => none

def specFold (vf : CredOK) : List Rec → Spec → List Rec → Spec
  | _, s, [] => s
  | pre, s, r :: rs => specFold vf (pre ++ [r]) (specStep vf pre s r) rs

/-- The specification state of a whole log (any log, admissible or not). -/
def specOf (vf : CredOK) : List Rec → Option Spec
  | [] => none
  | g :: rs => (specInit g).map fun s => specFold vf [g] s rs

/-- **Safety condition of an action**: after log `L`, action kind `k` may execute iff no hazard is
pending and `k` was never prohibited by any policy in `L`. -/
def specPermits (vf : CredOK) (L : List Rec) (k : ℕ) : Bool :=
  match specOf vf L with
  | some s => !s.hazard && !decide (k ∈ s.proh)
  | none => false

theorem specFold_append (vf : CredOK) (pre : List Rec) (s : Spec) (rs : List Rec) (r : Rec) :
    specFold vf pre s (rs ++ [r]) = specStep vf (pre ++ rs) (specFold vf pre s rs) r := by
  induction rs generalizing pre s with
  | nil => simp [specFold]
  | cons x xs ih => simp only [List.cons_append, specFold]; rw [ih]; simp [List.append_assoc]

/-- An `execute` record changes nothing the specification depends on. -/
theorem specOf_append_execute (vf : CredOK) {L : List Rec} (hL : L ≠ []) (k : ℕ) :
    specOf vf (L ++ [.execute k]) = specOf vf L := by
  obtain ⟨g, rs, rfl⟩ := List.exists_cons_of_ne_nil hL
  simp only [List.cons_append, specOf]
  cases specInit g with
  | none => rfl
  | some s => simp [specFold_append, specStep]

theorem specPermits_append_execute (vf : CredOK) {L : List Rec} (hL : L ≠ []) (k a : ℕ) :
    specPermits vf (L ++ [.execute k]) a = specPermits vf L a := by
  simp only [specPermits, specOf_append_execute vf hL]

/-! ## Elementary memory properties -/

section Basic

variable {vf : CredOK}

theorem fold_append (pre : List Rec) (v : View) (rs : List Rec) (r : Rec) :
    fold vf pre v (rs ++ [r]) =
      (fold vf pre v rs).bind fun w =>
        if admissible vf (pre ++ rs) w r then some (next w r) else none := by
  induction rs generalizing pre v with
  | nil => simp [fold]
  | cons x xs ih =>
    simp only [List.cons_append, fold]
    split_ifs with h
    · rw [ih]; simp [List.append_assoc]
    · simp

/-- **Append** (the only memory update): loading `L ++ [r]` is loading `L` and admitting `r`. -/
theorem load_append {L : List Rec} (hL : L ≠ []) (r : Rec) :
    load vf (L ++ [r]) =
      (load vf L).bind fun w => if admissible vf L w r then some (next w r) else none := by
  obtain ⟨g, rs, rfl⟩ := List.exists_cons_of_ne_nil hL
  simp only [List.cons_append, load]
  cases genesisView g with
  | none => simp
  | some v => simp [fold_append]

theorem load_snoc_some {L : List Rec} {r : Rec} {v : View} (hL : L ≠ [])
    (h : load vf L = some v) (ha : admissible vf L v r = true) :
    load vf (L ++ [r]) = some (next v r) := by
  rw [load_append hL, h]; simp [ha]

theorem load_ne_nil {L : List Rec} {v : View} (h : load vf L = some v) : L ≠ [] := by
  rintro rfl; simp [load] at h

/-- Loading is monotone in the set of valid credentials: issuing more credentials never makes an
admitted log inadmissible or changes its view. -/
theorem admissible_mono {vf vf' : CredOK} (hv : ∀ c pre, vf c pre = true → vf' c pre = true)
    {pre : List Rec} {v : View} {r : Rec} (h : admissible vf pre v r = true) :
    admissible vf' pre v r = true := by
  cases r <;> simp_all [admissible]

theorem fold_mono {vf vf' : CredOK} (hv : ∀ c pre, vf c pre = true → vf' c pre = true) :
    ∀ (pre : List Rec) (v : View) (rs : List Rec) (w : View),
      fold vf pre v rs = some w → fold vf' pre v rs = some w
  | _, _, [], _, h => h
  | pre, v, r :: rs, w, h => by
    simp only [fold] at h ⊢
    split_ifs at h with ha
    rw [if_pos (admissible_mono hv ha)]
    exact fold_mono hv _ _ rs w h

theorem load_mono {vf vf' : CredOK} (hv : ∀ c pre, vf c pre = true → vf' c pre = true)
    {L : List Rec} {v : View} (h : load vf L = some v) : load vf' L = some v := by
  cases L with
  | nil => simp [load] at h
  | cons g rs =>
    simp only [load] at h ⊢
    cases hg : genesisView g with
    | none => simp [hg] at h
    | some v0 => simp only [hg, Option.bind_some] at h ⊢; exact fold_mono hv _ _ _ _ h

/-- Every loadable log starts with its genesis record and no other genesis record. -/
theorem admissible_genesis {pre : List Rec} {v : View} {a : ℕ} {b : List ℤ} {p : Policy} :
    admissible vf pre v (.genesis a b p) = false := rfl

/-- The view fields fixed at genesis never change. -/
theorem next_agent (v : View) (r : Rec) : (next v r).agent = v.agent ∧
    (next v r).baseline = v.baseline := by
  cases r <;> simp [next]

/-- `locked` can only be cleared by a reset record. -/
theorem next_locked_of_not_reset {v : View} {r : Rec} (hr : ∀ c, r ≠ .reset c)
    (hl : v.locked = true) : (next v r).locked = true := by
  cases r with
  | reset c => exact absurd rfl (hr c)
  | _ => simp [next, hl]

end Basic

/-! ## The operational view refines the specification -/

section Refinement

variable {vf : CredOK}

/-- Agreement of a specification state and a view. -/
def Rel (s : Spec) (v : View) : Prop :=
  s.baseline = v.baseline ∧ s.thr = v.policy.threshold ∧
    (∀ k, k ∈ s.proh ↔ k ∈ v.policy.prohibited) ∧ s.maxSeq = v.lastSeq ∧
    s.lastDev = v.lastDev ∧ s.hazard = v.locked

theorem rel_step {pre : List Rec} {s : Spec} {v : View} {r : Rec} (h : Rel s v)
    (ha : admissible vf pre v r = true) : Rel (specStep vf pre s r) (next v r) := by
  obtain ⟨hb, ht, hp, hq, hd, hz⟩ := h
  cases r with
  | genesis => simp [admissible] at ha
  | observe q xs =>
    simp only [admissible, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at ha
    have hc : s.maxSeq < q ∧ xs.length = s.baseline.length := by
      rw [hq, hb]; exact ha
    simp only [specStep, if_pos hc, next]
    refine ⟨hb, ht, hp, rfl, by rw [hb], ?_⟩
    rw [hz, hb, ht]
  | policy p =>
    simp only [admissible, Policy.successor, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
      List.all_eq_true] at ha
    obtain ⟨-, ⟨-, hthr⟩, hsub⟩ := ha
    have hmin : min s.thr p.threshold = p.threshold := by rw [ht]; exact min_eq_right hthr
    simp only [specStep, next, hmin]
    refine ⟨hb, rfl, fun k => ?_, hq, hd, by rw [hz, hd]⟩
    simp only [List.mem_append]
    constructor
    · rintro (hk | hk)
      · exact hsub k ((hp k).1 hk)
      · exact hk
    · exact Or.inr
  | execute k => simpa [specStep, next] using ⟨hb, ht, hp, hq, hd, hz⟩
  | reset c =>
    simp only [admissible, Bool.and_eq_true, decide_eq_true_eq] at ha
    obtain ⟨⟨hl, hlt⟩, hc⟩ := ha
    have hcond : s.hazard ∧ s.lastDev < s.thr ∧ vf c pre = true := by
      rw [hz, hd, ht]; exact ⟨hl, hlt, hc⟩
    simp only [specStep, if_pos hcond, next]
    exact ⟨hb, ht, hp, hq, hd, rfl⟩

theorem rel_fold : ∀ (pre : List Rec) (s : Spec) (v : View) (rs : List Rec) (w : View),
    Rel s v → fold vf pre v rs = some w → Rel (specFold vf pre s rs) w
  | _, _, _, [], w, h, hf => by simp only [fold, Option.some.injEq] at hf; subst hf; exact h
  | pre, s, v, r :: rs, w, h, hf => by
    simp only [fold] at hf
    split_ifs at hf with ha
    exact rel_fold _ _ _ rs w (rel_step h ha) hf

/-- **Refinement.**  On every log the loader accepts, the specification state exists and agrees
with the operational view. -/
theorem load_refines_spec {L : List Rec} {v : View} (h : load vf L = some v) :
    ∃ s, specOf vf L = some s ∧ Rel s v := by
  cases L with
  | nil => simp [load] at h
  | cons g rs =>
    simp only [load] at h
    cases g with
    | genesis a b p =>
      simp only [genesisView] at h
      split_ifs at h with hg
      · simp only [Option.bind_some] at h
        refine ⟨specFold vf [.genesis a b p] ⟨b, p.threshold, p.prohibited, 0, 0, false⟩ rs,
          ?_, ?_⟩
        · simp [specOf, specInit, hg]
        · exact rel_fold _ _ _ rs v ⟨rfl, rfl, fun _ => Iff.rfl, rfl, rfl, rfl⟩ h
      · simp at h
    | _ => simp [genesisView] at h

/-- **The guard decides the specification.**  On a loadable log the pre-action guard admits
action `k` exactly when the declarative safety condition permits it. -/
theorem authorizes_iff_spec {L : List Rec} {v : View} (h : load vf L = some v) (k : ℕ) :
    v.authorizes k = specPermits vf L k := by
  obtain ⟨s, hs, _, _, hp, _, _, hz⟩ := load_refines_spec h
  simp only [View.authorizes, specPermits, hs, hz]
  rw [decide_eq_decide.mpr (hp k)]

end Refinement

end RCA
end WaveLockProofs
