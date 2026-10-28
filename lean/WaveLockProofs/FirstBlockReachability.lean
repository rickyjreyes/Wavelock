import WaveLockProofs.ExactField

/-!
# Frozen CC-Core-v1-B: full 256-cell first-block wave reachability bound

Every rate cell receives a little-endian 3-byte field element. On block
zero, CAP0 receives G = 7; CAP1 receives no finalization tag because
padded messages have >= 2 blocks; CAP2 receives the high counter digit 0.
The remaining cells keep their IV values. For arbitrary rate bytes and
IV entries in Fin 375 this proves that NO first-block post-absorption
wave coordinate is the Candidate B singular value 195225786.

This is a real ZMod-2147483647 theorem, not a theorem about later
nonlinear rounds, the accumulator, arbitrary internal states, or
cryptographic hardness.
-/

namespace WaveLockProofs

/-- Model the literal post-absorption, pre-wave-round FIRST wave state.
The IV bound contains every documented frozen psi IV coordinate. -/
def firstAbsorbedWave
    (iv : Fin 256 → Fin 375)
    (triples : Fin 64 → Fin 256 × Fin 256 × Fin 256)
    (i : Fin 256) : CCField :=
  if hr : i.val < 64 then
    let b := triples ⟨i.val, hr⟩
    firstBlockRateField (iv i) b.1 b.2.1 b.2.2
  else if i.val = 64 then
    ((iv i).val + 7 : Nat)
  else
    ((iv i).val : Nat)

/-- The complete 256-site wave state avoids the singular field element
immediately following FIRST-block absorption, for every byte triple. -/
theorem firstAbsorbedWave_val_lt_singular
    (iv : Fin 256 → Fin 375)
    (triples : Fin 64 → Fin 256 × Fin 256 × Fin 256)
    (i : Fin 256) :
    (firstAbsorbedWave iv triples i).val < 195225786 := by
  by_cases hr : i.val < 64
  · have hb := firstBlockRate_below_singular
        (iv i) (triples ⟨i.val, hr⟩).1
        (triples ⟨i.val, hr⟩).2.1
        (triples ⟨i.val, hr⟩).2.2
    simpa only [firstAbsorbedWave, dif_pos hr, firstBlockRateField_val] using hb
  · by_cases hc : i.val = 64
    · have hiv : (iv i).val < 375 := (iv i).isLt
      have hbound : (iv i).val + 7 < 2147483647 := by omega
      have hval : (((iv i).val + 7 : Nat) : CCField).val =
        (iv i).val + 7 := ZMod.val_natCast_of_lt hbound
      simp only [firstAbsorbedWave, dif_neg hr, if_pos hc, hval]
      omega
    · have hiv : (iv i).val < 375 := (iv i).isLt
      have hbound : (iv i).val < 2147483647 := by omega
      have hval : (((iv i).val : Nat) : CCField).val =
        (iv i).val := ZMod.val_natCast_of_lt hbound
      simp only [firstAbsorbedWave, dif_neg hr, if_neg hc, hval]
      omega

/-- No cell in the first absorbed wave field lies on the singular
hyperplane j_B(u,v)=0 by reason of v=v_star. This says nothing about
reachability after the next coupled wave update. -/
theorem firstAbsorbedWave_not_singular
    (iv : Fin 256 → Fin 375)
    (triples : Fin 64 → Fin 256 × Fin 256 × Fin 256)
    (i : Fin 256) :
    firstAbsorbedWave iv triples i ≠ (195225786 : CCField) := by
  intro heq
  have hv := congrArg ZMod.val heq
  have hconst : (195225786 : CCField).val = 195225786 :=
    ZMod.val_natCast_of_lt (by norm_num)
  have hlt := firstAbsorbedWave_val_lt_singular iv triples i
  rw [hconst] at hv
  omega

end WaveLockProofs
