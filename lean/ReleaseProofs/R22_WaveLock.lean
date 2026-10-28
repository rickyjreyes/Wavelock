import Mathlib

/-! R22 (DOI 10.5281/zenodo.19122146).
A conditional abstract commitment-binding lemma, plus a concrete
noninjectivity example. This does NOT prove WaveLock one-wayness,
collision resistance, time-locking or implementation correctness. -/
namespace ReleaseProofs.R22

def abstractCommit {State Digest : Type*}
    (step : State → State) (encode : State → Digest) (x : State) : Digest :=
  encode (step x)

theorem commitment_injective_of_injective_components
    {State Digest : Type*} (step : State → State)
    (encode : State → Digest)
    (hs : Function.Injective step) (he : Function.Injective encode) :
    Function.Injective (abstractCommit step encode) := by
  intro x y h
  apply hs
  apply he
  exact h

def toyDissipativeStep (x : ℝ) : ℝ := x - x ^ 3

theorem toyDissipativeStep_collides :
    toyDissipativeStep 0 = toyDissipativeStep 1 ∧ (0 : ℝ) ≠ 1 := by
  simp [toyDissipativeStep]

end ReleaseProofs.R22
