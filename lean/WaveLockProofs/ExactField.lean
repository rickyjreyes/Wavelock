import WaveLockProofs.Encoding

/-!
# Exact initial-byte embedding in the frozen CC-Core-v1-B field

The production modulus is 2^31-1. First-block rate-cell arithmetic is
proved in ZMod 2147483647, not merely over unbounded natural numbers.
The bounds are *only* for the first absorption into rate cells before
the first wave update. They do not establish later-round reachability,
full protocol equivalence, or collision/preimage hardness.
-/

namespace WaveLockProofs

/-- The frozen CC-Core-v1-B modular arithmetic representation. -/
abbrev CCField := ZMod 2147483647

/-- Byte triple embedding with actual finite-field reduction semantics. -/
def pack3Field (b0 b1 b2 : Fin 256) : CCField :=
  (pack3 b0 b1 b2 : CCField)

/-- First-block packed-byte values do not wrap modulo the field prime. -/
theorem pack3Field_val (b0 b1 b2 : Fin 256) :
    (pack3Field b0 b1 b2).val = pack3 b0 b1 b2 := by
  apply ZMod.val_natCast_of_lt
  exact lt_trans (pack3_lt_2pow24 b0 b1 b2) (by norm_num)

/-- The first rate-cell byte triple is injective even AFTER casting into
the concrete finite field. This is a restricted encoding property. -/
theorem pack3Field_unique
    (a b c d e f : Fin 256)
    (h : pack3Field a b c = pack3Field d e f) :
    a = d ∧ b = e ∧ c = f := by
  have hv := congrArg ZMod.val h
  have hnat : pack3 a b c = pack3 d e f := by
    simpa only [pack3Field_val] using hv
  exact pack3_unique a b c d e f hnat

/-- The actual first-block absorbed rate-cell value in the finite field. -/
def firstBlockRateField (iv : Fin 375) (b0 b1 b2 : Fin 256) : CCField :=
  (firstBlockRateValue iv b0 b1 b2 : CCField)

/-- Exact field-to-integer agreement before the first wave round. -/
theorem firstBlockRateField_val
    (iv : Fin 375) (b0 b1 b2 : Fin 256) :
    (firstBlockRateField iv b0 b1 b2).val =
      firstBlockRateValue iv b0 b1 b2 := by
  apply ZMod.val_natCast_of_lt
  exact firstBlockRate_no_wrap iv b0 b1 b2

/-- Candidate B's singular value is unreachable at any FIRST-BLOCK
RATE cell after a valid absorbed byte triple and an allowed IV value.
This does not cover capacity cells, subsequent blocks, or later rounds. -/
theorem firstBlockRateField_not_singular
    (iv : Fin 375) (b0 b1 b2 : Fin 256) :
    firstBlockRateField iv b0 b1 b2 ≠ (195225786 : CCField) := by
  intro h
  have hv := congrArg ZMod.val h
  have hconst : (195225786 : CCField).val = 195225786 :=
    ZMod.val_natCast_of_lt (by norm_num)
  have hnat : firstBlockRateValue iv b0 b1 b2 = 195225786 := by
    simpa only [firstBlockRateField_val, hconst] using hv
  exact firstBlockRate_not_singular iv b0 b1 b2 hnat

end WaveLockProofs
