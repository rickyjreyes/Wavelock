import WaveLockProofs.ExactField

/-!
# Frozen CC-Core-v1-B literal constants with checked decimal conversions

These definitions use the executable Python specification's hexadecimal
literals and modular reduction. Several earlier documentation decimal
annotations were arithmetically incorrect, and the earlier round-level
Lean model copied those annotations rather than their literal values.

The exact numeral equalities below are kernel-checked; they are the
source constants used by the corrected per-round and absorption models.
They do NOT establish full Python execution semantics.
-/

namespace WaveLockProofs

/-- Python: RHO0 = 0x57434330. -/
def ccCoreRho0 : Nat := 0x57434330

/-- Python: RHO1 = 2654435761 % P with P = 2^31 - 1. -/
def ccCoreRho1 : Nat := 2654435761 % 2147483647

/-- Python: D_TAG = 0x57434332, the Candidate B final block tag. -/
def ccCoreFinalTag : Nat := 0x57434332

theorem ccCoreRho0_decimal :
    ccCoreRho0 = 1464025904 := by
  decide

theorem ccCoreRho1_decimal :
    ccCoreRho1 = 506952114 := by
  decide

theorem ccCoreFinalTag_decimal :
    ccCoreFinalTag = 1464025906 := by
  decide

/-- The claimed coprimality of the slope with p-1 in an old audit
is false. Their exact common divisor is 186, so no period
argument may assume invertibility modulo p-1. -/
theorem ccCoreRho1_gcd_modulusMinusOne :
    Nat.gcd ccCoreRho1 (2147483647 - 1) = 186 := by
  decide

end WaveLockProofs
