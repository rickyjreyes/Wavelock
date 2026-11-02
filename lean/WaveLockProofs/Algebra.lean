import Mathlib

/-!
# WaveLock polynomial round and Candidate B: exact field algebra

This is a mathematical model of the finite-field equations in
docs/CC_CORE_V1_SPEC.md and docs/CC_CORE_V1_RESTRICTED_BINDING.md.
All theorems are over an arbitrary field; applying them to the normative
F_p requires identifying that field with ZMod 2147483647 and the frozen
coefficients. There is no assertion here about message-level hardness.
-/

namespace WaveLockProofs

variable {F : Type*} [Field F]

/-- CC-Core-v1-B's actual injection polynomial, before modular representation. -/
def candidateBInjection (gamma u v : F) : F := u + gamma * u * v

theorem candidateB_factor (gamma u v : F) :
    candidateBInjection gamma u v = u * (1 + gamma * v) := by
  unfold candidateBInjection
  ring

/-- For fixed next-wave value v away from the singular factor, injection
    is injective in the previous-wave value u. Not a digest theorem. -/
theorem candidateB_injective (gamma v : F) (h : 1 + gamma * v ≠ 0) :
    Function.Injective (fun u : F => candidateBInjection gamma u v) := by
  intro a b hab
  have heq : a * (1 + gamma * v) = b * (1 + gamma * v) := by
    simpa only [candidateB_factor] using hab
  have hzero : (a - b) * (1 + gamma * v) = 0 := by
    calc
      (a - b) * (1 + gamma * v) =
          a * (1 + gamma * v) - b * (1 + gamma * v) := by ring
      _ = 0 := sub_eq_zero.mpr heq
  rcases mul_eq_zero.mp hzero with hz | hz
  · exact sub_eq_zero.mp hz
  · exact False.elim (h hz)

/-- The singular factor erases the injection *at that coordinate only*. -/
theorem candidateB_singular (gamma v : F) (h : 1 + gamma * v = 0)
    (u : F) : candidateBInjection gamma u v = 0 := by
  rw [candidateB_factor, h]
  ring

theorem candidateB_zero_iff (gamma u v : F) :
    candidateBInjection gamma u v = 0 ↔
      u = 0 ∨ 1 + gamma * v = 0 := by
  rw [candidateB_factor]
  exact mul_eq_zero

/-- Exact sign-pair cancellation: the gamma*u*v contribution is even. -/
theorem candidateB_sign_difference (gamma u v : F) :
    candidateBInjection gamma u v -
      candidateBInjection gamma (-u) (-v) = (2 : F) * u := by
  unfold candidateBInjection
  ring

theorem candidateB_sign_separated (gamma u v : F)
    (h2 : (2 : F) ≠ 0) (hu : u ≠ 0) :
    candidateBInjection gamma u v ≠
      candidateBInjection gamma (-u) (-v) := by
  intro heq
  have hz : (2 : F) * u = 0 := by
    calc
      (2 : F) * u =
          candidateBInjection gamma u v -
            candidateBInjection gamma (-u) (-v) :=
              (candidateB_sign_difference gamma u v).symm
      _ = 0 := by rw [heq]; ring
  exact (mul_ne_zero h2 hu) hz

/-- The actual per-cell accumulator polynomial once the diffused C value,
    per-round weight, and round constant have been computed. -/
def accumulatorCell
    (mu quad weight rho gamma cd u v : F) : F :=
  mu * cd + quad * cd ^ 2 +
    weight * candidateBInjection gamma u v + rho

theorem accumulator_sign_difference
    (mu quad weight rho gamma cd u v : F) :
    accumulatorCell mu quad weight rho gamma cd u v -
      accumulatorCell mu quad weight rho gamma cd (-u) (-v) =
        weight * ((2 : F) * u) := by
  unfold accumulatorCell candidateBInjection
  ring

/-- Restricted first-round accumulator separation needs a nonzero weight,
    as well as characteristic != 2 and a nonzero starting amplitude. -/
theorem accumulator_sign_separated
    (mu quad weight rho gamma cd u v : F)
    (hweight : weight ≠ 0) (h2 : (2 : F) ≠ 0) (hu : u ≠ 0) :
    accumulatorCell mu quad weight rho gamma cd u v ≠
      accumulatorCell mu quad weight rho gamma cd (-u) (-v) := by
  intro heq
  have hz : weight * ((2 : F) * u) = 0 := by
    calc
      weight * ((2 : F) * u) =
          accumulatorCell mu quad weight rho gamma cd u v -
            accumulatorCell mu quad weight rho gamma cd (-u) (-v) :=
              (accumulator_sign_difference mu quad weight rho gamma cd u v).symm
      _ = 0 := by rw [heq]; ring
  exact (mul_ne_zero hweight (mul_ne_zero h2 hu)) hz

/-- Frozen Design A polynomial wave round, parameterized by its Laplacian. -/
def waveRound {X : Type*} (lap : (X → F) → X → F)
    (D A B : F) (psi : X → F) : X → F :=
  fun x => psi x + D * lap psi x +
    A * psi x * (B - (psi x) ^ 2)

theorem waveRound_odd {X : Type*}
    (lap : (X → F) → X → F)
    (hlap : ∀ (psi : X → F) (x : X),
      lap (fun y => -psi y) x = -(lap psi x))
    (D A B : F) (psi : X → F) :
    waveRound lap D A B (fun x => -psi x) =
      (fun x => -(waveRound lap D A B psi x)) := by
  funext x
  change (-psi x) + D * lap (fun y => -psi y) x +
    A * (-psi x) * (B - (-psi x) ^ 2) =
      -(psi x + D * lap psi x + A * psi x * (B - (psi x) ^ 2))
  rw [hlap psi x]
  ring

/-- A Laplacian eigenstate of constant squared amplitude can collapse to 0,
    provided the explicit scalar balance equation holds. -/
theorem waveRound_collapse {X : Type*}
    (lap : (X → F) → X → F) (D A B r s : F) (psi : X → F)
    (heigen : ∀ x, lap psi x = -(2 * r) * psi x)
    (hsquare : ∀ x, (psi x) ^ 2 = s ^ 2)
    (hbalance : 1 - 2 * r * D + A * (B - s ^ 2) = 0) :
    waveRound lap D A B psi = (fun _ => (0 : F)) := by
  funext x
  change psi x + D * lap psi x +
    A * psi x * (B - (psi x) ^ 2) = 0
  rw [heigen x, hsquare x]
  calc
    psi x + D * (-(2 * r) * psi x) + A * psi x * (B - s ^ 2) =
        psi x * (1 - 2 * r * D + A * (B - s ^ 2)) := by ring
    _ = 0 := by rw [hbalance]; ring

/-- Distinct opposite-sign collapse states refute global injectivity of the
    internal round. This does not produce two valid *message* preimages. -/
theorem waveRound_not_injective_of_collapse {X : Type*}
    (lap : (X → F) → X → F)
    (hlap : ∀ (psi : X → F) (x : X),
      lap (fun y => -psi y) x = -(lap psi x))
    (D A B : F) (psi : X → F)
    (hcollapse : waveRound lap D A B psi = (fun _ => (0 : F)))
    (h2 : (2 : F) ≠ 0)
    (hnonzero : ∃ x, psi x ≠ 0) :
    ¬ Function.Injective (waveRound lap D A B) := by
  intro hinj
  obtain ⟨x, hx⟩ := hnonzero
  have hneg : waveRound lap D A B (fun y => -psi y) =
      (fun _ => (0 : F)) := by
    calc
      waveRound lap D A B (fun y => -psi y) =
          (fun y => -(waveRound lap D A B psi y)) :=
            waveRound_odd lap hlap D A B psi
      _ = (fun _ => (0 : F)) := by
        funext y
        have hy := congrFun hcollapse y
        simp only [hy, neg_zero]
  have heq : psi = (fun y => -psi y) := by
    apply hinj
    rw [hcollapse, hneg]
  have hxEq : psi x = -(psi x) := congrFun heq x
  have hz : (2 : F) * psi x = 0 := by
    calc
      (2 : F) * psi x = psi x - (-psi x) := by ring
      _ = 0 := sub_eq_zero.mpr hxEq
  exact (mul_ne_zero h2 hx) hz

/-- The same 16 by 16 periodic neighbor operator as the Python reference,
    represented with modular row and column coordinates. -/
abbrev TorusSite := ZMod 16 × ZMod 16

def torusLap (psi : TorusSite → F) (x : TorusSite) : F :=
  psi (x.1 + 1, x.2) +
    psi (x.1 - 1, x.2) +
    psi (x.1, x.2 + 1) +
    psi (x.1, x.2 - 1) -
    4 * psi x

theorem torusLap_odd :
    ∀ (psi : TorusSite → F) (x : TorusSite),
      torusLap (fun y => -psi y) x = -(torusLap psi x) := by
  intro psi x
  dsimp [torusLap]
  ring

theorem torusWave_odd (D A B : F) (psi : TorusSite → F) :
    waveRound torusLap D A B (fun x => -psi x) =
      (fun x => -(waveRound torusLap D A B psi x)) :=
  waveRound_odd torusLap torusLap_odd D A B psi

theorem torusSignPair_injection_difference (gamma D A B : F)
    (psi : TorusSite → F) (x : TorusSite) :
    candidateBInjection gamma (psi x) (waveRound torusLap D A B psi x) -
      candidateBInjection gamma (-psi x)
        (waveRound torusLap D A B (fun y => -psi y) x) =
          (2 : F) * psi x := by
  rw [congrFun (torusWave_odd D A B psi) x]
  exact candidateB_sign_difference gamma (psi x)
    (waveRound torusLap D A B psi x)

end WaveLockProofs
