import WaveLockProofs.RCAAgentState

/-!
# RCA drift metrics: definitions, properties and their limits (issue #43)

**Status: `PROPOSED_MODEL`.**  No drift metric existed in this repository before this work
(`audit/run_bounty_contract.py`, Critical 7: "No behavior-wide drift oracle, observation contract,
or detection verifier exists").  The metrics below are those of the proposed reference
(`wavelock/rca_reference/drift.py`); they are integer-valued so that Lean and Python compute the
same numbers.  A metric being well defined and having the proved properties does not make it a
validated detector of real behaviour: what it detects is fixed by what the observation channel
reports, which is an assumption (the *observation contract*), not a theorem.

## Notions kept apart

| notion | definition here | kernel use |
|---|---|---|
| state deviation | `l1 o b`: distance of an observation from the committed baseline | latch criterion |
| step deviation | `l1 o' o`: distance between consecutive observations | none (evadable) |
| accumulated drift | `path os`: sum of step deviations (path length) | none |
| prediction | `predictBreach dev rate h θ` | none (a proved-sound building block) |
| behavioural deviation | distance of the *true* state, which the kernel never sees | none |
| threshold | `Policy.threshold`, the latch | `next`, `View.authorizes` |

## Results

* `l1_comm`, `l1_self`, `l1_triangle`, `l1_eq_zero_iff` — `l1` is a metric on vectors of one
  dimension (`LEAN_PROVED`).
* `path_snoc`, `dev_le_path` — accumulated drift is monotone along a trace and bounds how far the
  state can have moved from where it started (`LEAN_PROVED`).
* `anchoredDetects_mono`, `tighten_restricts` — lowering the threshold only adds detections, and an
  admissible (tightening) policy update never authorizes an action that was not authorized
  before (`LEAN_PROVED`); `path_drift_without_deviation` — accumulated drift is not a hazard
  measure (`COUNTEREXAMPLE`; the correct statement is the upper bound `dev_le_path`).
* `step_detector_evaded` — for every step threshold `θ ≥ 2` and every target deviation `D`
  there is a trace whose steps all stay below `θ` but whose final state deviates by `D`: a
  per-step detector cannot bound cumulative drift (`COUNTEREXAMPLE`, counterexample 7).
  `kernel_latches_on_anchored` — the kernel's anchored latch fires on such a trace.
* `predict_sound` — if every future step deviation is at most `rate`, a negative prediction
  guarantees no breach within the horizon (`CONDITIONAL`); `predict_not_complete` (it raises false
  alarms) and `predict_unsound_without_rate_bound` (it misses breaches when the rate assumption
  fails) are `COUNTEREXAMPLE`s.
* `no_false_negative_outside_band`, `no_false_positive_outside_band` — if the measured observation
  is within `ε` (L1) of the true state, the latch decision is correct whenever the true deviation is
  at least `θ + ε` or below `θ - ε` (`CONDITIONAL`); `false_positive_in_band` and
  `false_negative_in_band` show both errors occur inside the band (counterexamples 11, 12).
* `manipulated_reports_hide_any_deviation` — the kernel's decisions are a function of the
  reported records alone (it never sees the true state), so a reporter that lies hides any
  behavioural deviation (counterexample 8): detection needs the observation contract.
-/

namespace WaveLockProofs
namespace RCA

/-! ## `l1` is a metric on vectors of one dimension -/

theorem l1_nil_left (v : List ℤ) : l1 [] v = 0 := by cases v <;> rfl

theorem l1_nil_right (u : List ℤ) : l1 u [] = 0 := by cases u <;> rfl

theorem l1_cons (a b : ℤ) (u v : List ℤ) : l1 (a :: u) (b :: v) = (a - b).natAbs + l1 u v := rfl

theorem l1_self : ∀ u : List ℤ, l1 u u = 0
  | [] => rfl
  | a :: u => by simp [l1_cons, l1_self u]

theorem l1_comm : ∀ u v : List ℤ, l1 u v = l1 v u
  | [], v => by rw [l1_nil_left, l1_nil_right]
  | a :: u, [] => rfl
  | a :: u, b :: v => by
    have : (a - b).natAbs = (b - a).natAbs := by rw [← Int.natAbs_neg, neg_sub]
    rw [l1_cons, l1_cons, l1_comm u v, this]

theorem l1_triangle : ∀ u v w : List ℤ, u.length = v.length → v.length = w.length →
    l1 u w ≤ l1 u v + l1 v w
  | [], _, _, _, _ => by simp [l1_nil_left]
  | _ :: _, [], _, h, _ => by simp at h
  | _ :: _, _ :: _, [], _, h => by simp at h
  | a :: u, b :: v, c :: w, h1, h2 => by
    simp only [List.length_cons, Nat.add_right_cancel_iff] at h1 h2
    rw [l1_cons, l1_cons, l1_cons]
    have := l1_triangle u v w h1 h2
    have hab : (a - c).natAbs ≤ (a - b).natAbs + (b - c).natAbs := by
      have := Int.natAbs_add_le (a - b) (b - c); simpa using this
    omega

theorem l1_eq_zero_iff : ∀ u v : List ℤ, u.length = v.length → (l1 u v = 0 ↔ u = v)
  | [], [], _ => by simp [l1]
  | [], _ :: _, h => by simp at h
  | _ :: _, [], h => by simp at h
  | a :: u, b :: v, h => by
    simp only [List.length_cons, Nat.add_right_cancel_iff] at h
    rw [l1_cons, Nat.add_eq_zero_iff, l1_eq_zero_iff u v h, Int.natAbs_eq_zero, sub_eq_zero,
      List.cons.injEq]

/-! ## Accumulated drift -/

/-- Step deviations of a trace of observations. -/
def steps : List (List ℤ) → List ℕ
  | o :: o' :: os => l1 o' o :: steps (o' :: os)
  | _ => []

/-- Accumulated drift: the length of the observed path. -/
def path (os : List (List ℤ)) : ℕ := (steps os).sum

/-- Same-dimension traces (`check_vector` + the kernel's dimension check). -/
def Dim (n : ℕ) (os : List (List ℤ)) : Prop := ∀ o ∈ os, o.length = n

theorem steps_snoc : ∀ (os : List (List ℤ)) (o x : List ℤ),
    steps (o :: os ++ [x]) = steps (o :: os) ++ [l1 x ((o :: os).getLast (by simp))]
  | [], o, x => by simp [steps]
  | y :: os, o, x => by
    have := steps_snoc os y x
    simp only [List.cons_append] at this ⊢
    simp only [steps, this, List.cons_append, List.getLast_cons_cons]

/-- Accumulated drift only grows. -/
theorem path_snoc (os : List (List ℤ)) (o x : List ℤ) :
    path (o :: os ++ [x]) = path (o :: os) + l1 x ((o :: os).getLast (by simp)) := by
  unfold path; rw [steps_snoc, List.sum_append, List.sum_singleton]

/-- The final deviation is at most the initial deviation plus the accumulated drift. -/
theorem dev_le_path {n : ℕ} (b : List ℤ) (hb : b.length = n) :
    ∀ (os : List (List ℤ)) (o : List ℤ), Dim n (o :: os) →
      l1 ((o :: os).getLast (by simp)) b ≤ l1 o b + path (o :: os)
  | [], o, _ => by simp [path, steps]
  | x :: os, o, hd => by
    have hx : Dim n (x :: os) := fun y hy => hd y (List.mem_cons_of_mem _ hy)
    have ih := dev_le_path b hb os x hx
    have hstep : l1 x b ≤ l1 x o + l1 o b :=
      l1_triangle x o b ((hx x (List.mem_cons_self)).trans (hd o List.mem_cons_self).symm)
        ((hd o List.mem_cons_self).trans hb.symm)
    have hp : path (o :: x :: os) = l1 x o + path (x :: os) := by simp [path, steps]
    simp only [List.getLast_cons_cons] at ih ⊢
    rw [hp]; omega

/-! ## Step detectors are evadable (counterexample 7) -/

/-- A per-step detector fires when one step deviation reaches `θ`. -/
def stepDetects (θ : ℕ) (os : List (List ℤ)) : Bool := (steps os).any fun s => decide (θ ≤ s)

/-- The anchored detector fires when one observation deviates from the baseline by `θ`. -/
def anchoredDetects (θ : ℕ) (b : List ℤ) (os : List (List ℤ)) : Bool :=
  os.any fun o => decide (θ ≤ l1 o b)

/-- The slow ramp `[0], [1], …, [D]`. -/
def ramp : ℕ → List (List ℤ)
  | 0 => [[0]]
  | n + 1 => ramp n ++ [[((n + 1 : ℕ) : ℤ)]]

theorem ramp_cons : ∀ n, ∃ os, ramp n = [0] :: os
  | 0 => ⟨[], rfl⟩
  | n + 1 => by
    obtain ⟨os, h⟩ := ramp_cons n
    exact ⟨os ++ [[((n + 1 : ℕ) : ℤ)]], by simp [ramp, h]⟩

theorem ramp_getLast : ∀ n (h : ramp n ≠ []), (ramp n).getLast h = [(n : ℤ)]
  | 0, _ => rfl
  | n + 1, _ => by simp [ramp]

theorem steps_ramp_all_one : ∀ D, ∀ s ∈ steps (ramp D), s = 1
  | 0 => by simp [ramp, steps]
  | n + 1 => by
    obtain ⟨os, h⟩ := ramp_cons n
    intro s hs
    have e : ramp (n + 1) = [0] :: os ++ [[((n + 1 : ℕ) : ℤ)]] := by simp [ramp, h]
    rw [e, steps_snoc] at hs
    rcases List.mem_append.1 hs with hs | hs
    · exact steps_ramp_all_one n s (by rw [h]; exact hs)
    · rw [List.mem_singleton] at hs; subst hs
      have hl : ([0] :: os).getLast (List.cons_ne_nil _ _) = [(n : ℤ)] := by
        have := ramp_getLast n (by simp [h]); simp only [h] at this; exact this
      rw [hl]; simp [l1]

/-- **Counterexample 7.**  For every step threshold `θ ≥ 2` and every target deviation `D`, the
slow ramp to `D` never triggers the step detector although its final state deviates by `D` from
the baseline `[0]`. -/
theorem step_detector_evaded (θ D : ℕ) (hθ : 2 ≤ θ) :
    stepDetects θ (ramp D) = false ∧ l1 [(D : ℤ)] [0] = D := by
  constructor
  · simp only [stepDetects, List.any_eq_false, decide_eq_true_eq, not_le]
    intro s hs; rw [steps_ramp_all_one D s hs]; omega
  · simp [l1]

/-- The kernel's anchored latch on the same ramp: with threshold `θ ≤ D` it fires. -/
theorem kernel_latches_on_anchored (θ D : ℕ) (hθD : θ ≤ D) :
    anchoredDetects θ [0] (ramp D) = true := by
  simp only [anchoredDetects, List.any_eq_true, decide_eq_true_eq]
  have hne : ramp D ≠ [] := by obtain ⟨os, h⟩ := ramp_cons D; simp [h]
  refine ⟨[(D : ℤ)], ?_, by simpa [l1] using hθD⟩
  rw [← ramp_getLast D hne]; exact List.getLast_mem hne

/-- **Threshold monotonicity.**  Lowering the threshold can only add detections. -/
theorem anchoredDetects_mono {θ θ' : ℕ} (h : θ' ≤ θ) (b : List ℤ) (os : List (List ℤ))
    (hd : anchoredDetects θ b os = true) : anchoredDetects θ' b os = true := by
  simp only [anchoredDetects, List.any_eq_true, decide_eq_true_eq] at hd ⊢
  obtain ⟨o, ho, hθ⟩ := hd
  exact ⟨o, ho, le_trans h hθ⟩

/-- **Accumulated drift is not a hazard measure.**  An oscillating trace accumulates path drift
without bound while never leaving distance 1 of the baseline: thresholding `path` would raise
alarms on a system that never moved (the minimal counterexample to "large accumulated drift means
large deviation"); the correct relation is the upper bound `dev_le_path`. -/
theorem path_drift_without_deviation :
    path [[0], [1], [0], [1], [0], [1], [0]] = 6 ∧
      anchoredDetects 2 [0] [[0], [1], [0], [1], [0], [1], [0]] = false := by
  decide

/-- **Guard monotonicity under tightening.**  If an admissible policy update (a successor policy)
leaves an action authorized, the action was authorized before it: tightening never authorizes
anything new. -/
theorem tighten_restricts (v : View) (p : Policy) (hp : p.successor v.policy = true) (k : ℕ)
    (h : (next v (.policy p)).authorizes k = true) : v.authorizes k = true := by
  simp only [Policy.successor, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
    List.all_eq_true] at hp
  simp only [View.authorizes, next, Bool.and_eq_true, Bool.not_eq_true', Bool.or_eq_false_iff,
    decide_eq_false_iff_not] at h ⊢
  refine ⟨h.1.1, fun hk => h.2 (hp.2 k hk)⟩

/-! ## Prediction -/

/-- Conservative breach predictor (`drift.predict_breach`). -/
def predictBreach (dev rate horizon θ : ℕ) : Bool := decide (θ ≤ dev + rate * horizon)

/-- Under a per-step rate bound, the deviation after `k` steps is at most `dev₀ + rate·k`. -/
theorem dev_le_rate {n : ℕ} (b : List ℤ) (hb : b.length = n) (rate : ℕ) :
    ∀ (os : List (List ℤ)) (o : List ℤ), Dim n (o :: os) → (∀ s ∈ steps (o :: os), s ≤ rate) →
      ∀ k (hk : k < (o :: os).length), l1 ((o :: os)[k]) b ≤ l1 o b + rate * k
  | _, o, _, _, 0, _ => by simp
  | [], _, _, _, k + 1, hk => by simp at hk
  | x :: os, o, hd, hr, k + 1, hk => by
    have hx : Dim n (x :: os) := fun y hy => hd y (List.mem_cons_of_mem _ hy)
    have hr' : ∀ s ∈ steps (x :: os), s ≤ rate := fun s hs => hr s (by simp [steps, hs])
    have h1 : l1 x o ≤ rate := hr _ (by simp [steps])
    have ih := dev_le_rate b hb rate os x hx hr' k (by simpa using hk)
    have htri : l1 x b ≤ l1 x o + l1 o b :=
      l1_triangle x o b ((hx x List.mem_cons_self).trans (hd o List.mem_cons_self).symm)
        ((hd o List.mem_cons_self).trans hb.symm)
    simp only [List.getElem_cons_succ]
    have : rate * (k + 1) = rate * k + rate := by ring
    omega

/-- **Prediction soundness (`CONDITIONAL` on the rate bound).**  If every step deviation of the
coming trace is at most `rate` and the predictor reports no breach within `horizon`, then no
observation within the horizon reaches the threshold. -/
theorem predict_sound {n : ℕ} (b : List ℤ) (hb : b.length = n) (rate horizon θ : ℕ)
    (os : List (List ℤ)) (o : List ℤ) (hd : Dim n (o :: os))
    (hr : ∀ s ∈ steps (o :: os), s ≤ rate)
    (hp : predictBreach (l1 o b) rate horizon θ = false) :
    ∀ k (hk : k < (o :: os).length), k ≤ horizon → l1 ((o :: os)[k]) b < θ := by
  intro k hk hkh
  have h1 := dev_le_rate b hb rate os o hd hr k hk
  simp only [predictBreach, decide_eq_false_iff_not, not_le] at hp
  have : rate * k ≤ rate * horizon := Nat.mul_le_mul_left rate hkh
  omega

/-- The predictor is not complete: it raises an alarm for a trace that never breaches. -/
theorem predict_not_complete :
    predictBreach 0 1 5 3 = true ∧ anchoredDetects 3 [0] [[0], [0], [0], [0], [0], [0]] = false := by
  decide

/-- Without the rate bound the predictor misses a breach one step ahead. -/
theorem predict_unsound_without_rate_bound :
    predictBreach 0 0 1 3 = false ∧ anchoredDetects 3 [0] [[0], [5]] = true := by
  decide

/-! ## False positives and false negatives under bounded measurement noise -/

/-- The measured and true deviations differ by at most the measurement error. -/
theorem measured_dev_bounds {m t b : List ℤ} (hm : m.length = t.length) (ht : t.length = b.length)
    {ε : ℕ} (hε : l1 m t ≤ ε) : l1 t b ≤ l1 m b + ε ∧ l1 m b ≤ l1 t b + ε := by
  constructor
  · have := l1_triangle t m b hm.symm (hm.trans ht)
    rw [l1_comm t m] at this; omega
  · have := l1_triangle m t b hm ht; omega

/-- **No false negative outside the band (`CONDITIONAL` on the noise bound).** -/
theorem no_false_negative_outside_band {m t b : List ℤ} (hm : m.length = t.length)
    (ht : t.length = b.length) {ε θ : ℕ} (hε : l1 m t ≤ ε) (htrue : θ + ε ≤ l1 t b) :
    θ ≤ l1 m b := by
  have := (measured_dev_bounds hm ht hε).1; omega

/-- **No false positive outside the band (`CONDITIONAL` on the noise bound).** -/
theorem no_false_positive_outside_band {m t b : List ℤ} (hm : m.length = t.length)
    (ht : t.length = b.length) {ε θ : ℕ} (hε : l1 m t ≤ ε) (htrue : l1 t b + ε < θ) :
    l1 m b < θ := by
  have := (measured_dev_bounds hm ht hε).2; omega

/-- **Counterexample 11 (false positive).**  True state `[4]` is safe for threshold 5; a
measurement error of 1 reports `[5]` and the latch fires. -/
theorem false_positive_in_band :
    l1 [4] [0] < 5 ∧ l1 [5] [4] ≤ 1 ∧ 5 ≤ l1 [5] [0] := by decide

/-- **Counterexample 12 (false negative).**  True state `[5]` reaches threshold 5; a measurement
error of 1 reports `[4]` and the latch does not fire. -/
theorem false_negative_in_band :
    5 ≤ l1 [5] [0] ∧ l1 [4] [5] ≤ 1 ∧ l1 [4] [0] < 5 := by decide

/-! ## The observation contract (counterexample 8) -/

/-- **Counterexample 8.**  A reporter that always reports the baseline hides any true deviation:
for every threshold and every deviation `D`, the reported observation never latches while the true
state deviates by `D`. -/
theorem manipulated_reports_hide_any_deviation (θ D : ℕ) (hθ : 1 ≤ θ) :
    let truth : List ℤ := [(D : ℤ)]
    let reported : List ℤ := [0]
    D ≤ l1 truth [0] ∧ l1 reported [0] < θ := by
  simp [l1]; omega

end RCA
end WaveLockProofs
