import WaveLockProofs.FirstBlockReachability
import WaveLockProofs.SecurityGames

/-!
# Issue #32: exact finite-field Candidate B singularity witness

A concrete algebraic counterexample at v_star in the *actual* CC-Core-v1-B
field p=2147483647. The multiplicative injection loses every u value
if its second input reaches v_star.

This is an INTERNAL one-coordinate statement: valid-message reachability
of v_star after any coupled wave round is not proved, nor is a full
digest collision, an efficient attack, or computational hardness.
-/

namespace WaveLockProofs

/-- The normative singular residue solves 1 + 11*v_star = 0 in F_p. -/
theorem candidateB_singular_value_exact :
    (1 : CCField) + 11 * (195225786 : CCField) = 0 := by
  norm_num [CCField]

/-- The exact injection vanishes for every possible first argument at
the normative singular value, including all 2^31-1 field values. -/
theorem candidateB_erases_at_singular_value (u : CCField) :
    candidateBInjection (11 : CCField) u (195225786 : CCField) = 0 :=
  candidateB_singular 11 195225786 candidateB_singular_value_exact u

/-- A concrete counterexample to unrestricted fixed-v injectivity of
the internal Candidate B injection. It is NOT a message-level collision. -/
theorem candidateB_singular_slice_not_injective :
    ¬ Function.Injective
      (fun u : CCField => candidateBInjection 11 u 195225786) := by
  intro hinj
  have heq :
      candidateBInjection (11 : CCField) (0 : CCField) 195225786 =
      candidateBInjection (11 : CCField) (1 : CCField) 195225786 := by
    rw [candidateB_erases_at_singular_value,
      candidateB_erases_at_singular_value]
  have h01 : (0 : CCField) = 1 := hinj heq
  exact zero_ne_one h01

end WaveLockProofs
