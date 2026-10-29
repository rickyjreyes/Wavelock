import WaveLockProofs.FirstBlockReachability

/-!
# CC-Core-v1-B: exact all-block finite-field absorption operator

This is an executable representation of the *per-block* update of the
256-site wave (gain 1) and accumulator (gain G=7) arrays. It includes
rate injection, two base-p counter digits and the last-block tag.
The first block has q=(k+1)=1 and no last-block tag, matching the
preexisting verified FirstBlockReachability model.

It does not yet verify the Python program's control-flow execution,
the 32-round evolution between block updates or the final digest.
-/

namespace WaveLockProofs

/-- Frozen base-p block counter digit pair, k is zero-based. -/
def ccCounter (k : Nat) : Nat × Nat :=
  counterDigits 2147483647 k

/-- The state update for a single component with the specified rate
gain. Wave gain = 1, accumulator gain = 7. Both capacity-counter
changes and the final-block tag have gain 7 or 1, respectively,
exactly as in the normative reference implementation. -/
def absorbBlockComponent
    (rateGain : CCField)
    (state : Fin 256 → CCField)
    (block : Fin 64 → CCField)
    (k : Nat) (last : Bool) (i : Fin 256) : CCField :=
  if hr : i.val < 64 then
    state i + rateGain * block ⟨i.val, hr⟩
  else if i.val = 64 then
    state i + 7 * ((ccCounter k).1 : CCField)
  else if i.val = 66 then
    state i + 7 * ((ccCounter k).2 : CCField)
  else if i.val = 65 ∧ last = true then
    state i + (1463898162 : CCField)
  else state i

def absorbWaveBlock
    (state : Fin 256 → CCField) (block : Fin 64 → CCField)
    (k : Nat) (last : Bool) : Fin 256 → CCField :=
  absorbBlockComponent 1 state block k last

def absorbAccumulatorBlock
    (state : Fin 256 → CCField) (block : Fin 64 → CCField)
    (k : Nat) (last : Bool) : Fin 256 → CCField :=
  absorbBlockComponent 7 state block k last

/-- Arbitrary-block rate injection, exact in the finite residue ring. -/
theorem absorbWaveBlock_rate
    (state : Fin 256 → CCField) (block : Fin 64 → CCField)
    (k : Nat) (last : Bool) (i : Fin 256) (hi : i.val < 64) :
    absorbWaveBlock state block k last i =
      state i + block ⟨i.val, hi⟩ := by
  simp [absorbWaveBlock, absorbBlockComponent, hi]

/-- The accumulator receives seven times the exact packed byte field
element in every rate cell. -/
theorem absorbAccumulatorBlock_rate
    (state : Fin 256 → CCField) (block : Fin 64 → CCField)
    (k : Nat) (last : Bool) (i : Fin 256) (hi : i.val < 64) :
    absorbAccumulatorBlock state block k last i =
      state i + 7 * block ⟨i.val, hi⟩ := by
  simp [absorbAccumulatorBlock, absorbBlockComponent, hi]

/-- Every coordinate from 67 through 255 is unchanged by absorption,
even in an arbitrarily late block. This says nothing about evolution. -/
theorem absorbBlockComponent_high_unchanged
    (gain : CCField) (state : Fin 256 → CCField)
    (block : Fin 64 → CCField)
    (k : Nat) (last : Bool) (i : Fin 256) (hi : 67 ≤ i.val) :
    absorbBlockComponent gain state block k last i = state i := by
  have hr : ¬ i.val < 64 := by omega
  have h64 : i.val ≠ 64 := by omega
  have h65 : i.val ≠ 65 := by omega
  have h66 : i.val ≠ 66 := by omega
  simp [absorbBlockComponent, hr, h64, h65, h66]

/-- Every block's wave rate encoding is injective for a fixed current
state and block counter. This is before the nonlinear evolution and
the finite-bit digest, NOT a collision-resistance theorem. -/
theorem absorbWaveBlock_rate_injective
    (state : Fin 256 → CCField) (k : Nat) (last : Bool) :
    Function.Injective (fun block : Fin 64 → CCField =>
      absorbWaveBlock state block k last) := by
  intro a b hab
  funext j
  let i : Fin 256 := ⟨j.val, by omega⟩
  have heq := congrFun hab i
  have hleft := absorbWaveBlock_rate state a k last i (by
    dsimp [i]
    exact j.isLt)
  have hright := absorbWaveBlock_rate state b k last i (by
    dsimp [i]
    exact j.isLt)
  change absorbWaveBlock state a k last i = absorbWaveBlock state b k last i at heq
  rw [hleft, hright] at heq
  have hj : (⟨i.val, by
      dsimp [i]
      exact j.isLt⟩ : Fin 64) = j := Fin.ext rfl
  rw [hj] at heq
  exact add_left_cancel heq

/-- For block zero, the exact per-block absorption function coincides
with the earlier verified 256-cell initial wave model. The two
representations agree for every admitted IV array and byte triple. -/
theorem absorbWaveBlock_first_matches_initial
    (iv : Fin 256 → Fin 375)
    (triples : Fin 64 → Fin 256 × Fin 256 × Fin 256) :
    absorbWaveBlock (fun i => ((iv i).val : CCField))
      (fun c => let b := triples c; pack3Field b.1 b.2.1 b.2.2)
      0 false = firstAbsorbedWave iv triples := by
  funext i
  by_cases hr : i.val < 64
  · simp only [absorbWaveBlock, absorbBlockComponent, dif_pos hr,
      firstAbsorbedWave, dif_pos hr]
    unfold firstBlockRateField firstBlockRateValue pack3Field
    push_cast
    ring
  · by_cases h64 : i.val = 64
    · simp [absorbWaveBlock, absorbBlockComponent, firstAbsorbedWave,
        hr, h64, ccCounter, counterDigits]
    · have hcounter : (ccCounter 0).2 = 0 := by
        norm_num [ccCounter, counterDigits]
      simp [absorbWaveBlock, absorbBlockComponent, firstAbsorbedWave,
        hr, h64, ccCounter, counterDigits, hcounter]

end WaveLockProofs
