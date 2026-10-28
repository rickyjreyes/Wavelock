import WaveLockProofs.FirstBlockReachability
import WaveLockProofs.SecurityGames

/-!
# Issue #32: concrete Candidate B singularity over ZMod 2147483647

The actual finite residue ring is used without assuming a Field instance
(primality certification is not needed for this polynomial obstruction).
The singular factor makes the one-coordinate Candidate B polynomial
constant in its first input at the specified second argument.

This is NOT proof of valid-message reachability, a final-digest
collision, an efficient attack, or cryptographic hardness.
-/

namespace WaveLockProofs

/-- The exact normative singular residue solves 1+11*v=0 modulo p. -/
theorem candidateB_singular_value_exact :
    (1 : CCField) + 11 * (195225786 : CCField) = 0 := by
  have hnat : (1 + 11 * 195225786 : ℕ) = 2147483647 := by norm_num
  calc
    (1 : CCField) + 11 * (195225786 : CCField) =
        ((1 + 11 * 195225786 : ℕ) : CCField) := by push_cast; ring
    _ = (2147483647 : CCField) := by rw [hnat]
    _ = 0 := ZMod.natCast_self 2147483647

/-- Literal Candidate B injection polynomial on the concrete residue
ring. This does not need the prime-field instance assumed by the
existing generic Field-valued candidateBInjection. -/
def concreteCandidateBInjection (u v : CCField) : CCField :=
  u + 11 * u * v

/-- Exact one-coordinate information erasure on the singular slice. -/
theorem concreteCandidateB_erases_at_singular_value (u : CCField) :
    concreteCandidateBInjection u 195225786 = 0 := by
  calc
    concreteCandidateBInjection u 195225786 =
        u * (1 + 11 * (195225786 : CCField)) := by
          unfold concreteCandidateBInjection
          ring
    _ = 0 := by rw [candidateB_singular_value_exact]; ring

/-- A concrete counterexample to unrestricted fixed-v injectivity of
the internal polynomial, without any claim about message collisions. -/
theorem concreteCandidateB_singular_slice_not_injective :
    ¬ Function.Injective
      (fun u : CCField => concreteCandidateBInjection u 195225786) := by
  intro hinj
  have heq :
      concreteCandidateBInjection (0 : CCField) 195225786 =
      concreteCandidateBInjection (1 : CCField) 195225786 := by
    rw [concreteCandidateB_erases_at_singular_value,
      concreteCandidateB_erases_at_singular_value]
  have h01 : (0 : CCField) = 1 := hinj heq
  exact zero_ne_one h01

end WaveLockProofs
