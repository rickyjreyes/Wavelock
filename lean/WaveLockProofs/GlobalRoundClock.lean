import WaveLockProofs.Protocol
import WaveLockProofs.BlockAbsorption

/-!
# CC-Core-v1-B: absolute-round clock across consecutive evolutions

The normative coupled evolution uses the same global round index
throughout all padded blocks. Splitting an n+m round segment must
not reset the weights/rho schedule after its first n rounds.

The theorem proves the composition law for the already specified
deterministic runFrom model with the exact absolute time argument.
This is a round-level theorem, not full multi-block absorption parity
or an implementation-level Python correctness theorem.
-/

namespace WaveLockProofs

variable {F : Type*} [Field F]

/-- Two consecutive round segments are exactly one longer segment,
provided the second segment begins at the advanced absolute clock. -/
theorem runFrom_add_rounds (t n m : Nat) (s : CoupledState F) :
    runFrom t (n + m) s =
      runFrom (t + n) m (runFrom t n s) := by
  induction n generalizing t s with
  | zero =>
      simp [runFrom]
  | succ n ih =>
      rw [Nat.succ_add, runFrom_succ, runFrom_succ]
      simpa only [Nat.add_assoc, Nat.succ_eq_add_one, Nat.add_comm, Nat.add_left_comm]
        using ih (t + 1) (candidateBRound t s)

/-- In particular, two consecutive 32-round segments use clock t+32
at the second segment, never the original t. -/
theorem runFrom_two_32_round_segments (t : Nat) (s : CoupledState F) :
    runFrom t 64 s =
      runFrom (t + 32) 32 (runFrom t 32 s) := by
  simpa only [show 32 + 32 = 64 by decide] using
    runFrom_add_rounds t 32 32 s

/-- Run an arbitrarily long finite schedule of round segments. Every
segment starts at the absolute clock reached by preceding segments. -/
def runRoundSegments (t : Nat) :
    List Nat → CoupledState F → CoupledState F
  | [], s => s
  | rounds :: rest, s =>
      runRoundSegments (t + rounds) rest (runFrom t rounds s)

/-- The global round clock is consistent across *any* finite sequence
of segments, including zero-length segments and differing lengths.
This is stronger than a two-block calculation. -/
theorem runRoundSegments_eq_runFrom (t : Nat)
    (segments : List Nat) (s : CoupledState F) :
    runRoundSegments t segments s = runFrom t segments.sum s := by
  induction segments generalizing t s with
  | nil =>
      simp [runRoundSegments, runFrom]
  | cons rounds rest ih =>
      change runRoundSegments (t + rounds) rest
          (runFrom t rounds s) =
        runFrom t (rounds + rest.sum) s
      rw [ih]
      exact (runFrom_add_rounds t rounds rest.sum s).symm

/-- Two arbitrary adjacent executions are the special case of a
two-entry segment list; the time origin of the second is not reset. -/
theorem runRoundSegments_two (t n m : Nat) (s : CoupledState F) :
    runRoundSegments t [n, m] s = runFrom t (n + m) s := by
  simpa only [List.sum_cons, List.sum_nil, add_zero] using
    runRoundSegments_eq_runFrom t [n, m] s

end WaveLockProofs
