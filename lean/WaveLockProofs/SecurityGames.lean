import WaveLockProofs.Protocol

/-!
# Exact security games, composition limits, and counterexamples

This module distinguishes equality properties from *computational* claims.
No hardness bound is assumed or concluded. "Finite output" does not imply an
efficient collision-finder, and "collision-free in a tested family" does not
imply global collision resistance.
-/

namespace WaveLockProofs

/-- A pair of distinct inputs with the same observable digest. -/
def Collision {Msg Digest : Type*} (H : Msg → Digest) (a b : Msg) : Prop :=
  a ≠ b ∧ H a = H b

/-- A second preimage relative to a specified challenge message. -/
def SecondPreimage {Msg Digest : Type*}
    (H : Msg → Digest) (target candidate : Msg) : Prop :=
  target ≠ candidate ∧ H target = H candidate

/-- A preimage relative to a specified challenge output. -/
def Inverts {Msg Digest : Type*}
    (H : Msg → Digest) (output : Digest) (candidate : Msg) : Prop :=
  H candidate = output

/-- An adversary may use the digest as input. No oracle cost model is hidden. -/
def SecondPreimageGame {Msg Digest : Type*}
    (H : Msg → Digest) (A : Digest → Msg) (challenge : Msg) : Prop :=
  SecondPreimage H challenge (A (H challenge))

def InversionGame {Msg Digest : Type*}
    (H : Msg → Digest) (A : Digest → Msg) (output : Digest) : Prop :=
  Inverts H output (A output)

theorem collision_is_second_preimage {Msg Digest : Type*}
    (H : Msg → Digest) (a b : Msg) (h : Collision H a b) :
    SecondPreimage H a b := h

theorem no_second_preimage_if_injective {Msg Digest : Type*}
    (H : Msg → Digest) (hH : Function.Injective H)
    (a b : Msg) : ¬ SecondPreimage H a b := by
  intro h
  exact h.1 (hH h.2)

/-- Internal state merging implies digest merging only when the two messages
    actually reach that same state. -/
theorem collision_if_same_absorbed_state
    {Msg State Mid Digest : Type*}
    (absorb : Msg → State) (evolve : State → Mid)
    (squeeze : Mid → Digest) (a b : Msg)
    (hne : a ≠ b) (heq : absorb a = absorb b) :
    Collision (fun m => squeeze (evolve (absorb m))) a b := by
  constructor
  · exact hne
  · exact congrArg (fun state : State => squeeze (evolve state)) heq

theorem collision_if_same_evolved_state
    {Msg State Mid Digest : Type*}
    (absorb : Msg → State) (evolve : State → Mid)
    (squeeze : Mid → Digest) (a b : Msg)
    (hne : a ≠ b) (heq : evolve (absorb a) = evolve (absorb b)) :
    Collision (fun m => squeeze (evolve (absorb m))) a b := by
  constructor
  · exact hne
  · exact congrArg squeeze heq

/-- Sufficient but strong *information-theoretic* conditions. For an
    unbounded byte-message domain with fixed 256-bit output these cannot
    all hold; this is not a candidate security proof. -/
theorem composition_injective_of_layers
    {Msg State Mid Digest : Type*}
    (absorb : Msg → State) (evolve : State → Mid)
    (squeeze : Mid → Digest)
    (habsorb : Function.Injective absorb)
    (hevolve : Function.Injective evolve)
    (hsqueeze : Function.Injective squeeze) :
    Function.Injective (fun m => squeeze (evolve (absorb m))) := by
  intro a b h
  exact habsorb (hevolve (hsqueeze h))

/-- First-round separation does not survive arbitrary postprocessing.
    Identity is injective on two inputs; the constant output map is not. -/
theorem intermediate_injective_does_not_imply_digest_injective :
    ∃ (j : Fin 2 → Fin 2) (q : Fin 2 → Unit),
      Function.Injective j ∧
        ¬ Function.Injective (fun x => q (j x)) := by
  refine ⟨(fun x => x), (fun _ => ()), ?_, ?_⟩
  · intro x y h
    exact h
  · intro h
    have heq : (0 : Fin 2) = 1 := h (by rfl)
    have hne : (0 : Fin 2) ≠ 1 := by decide
    exact hne heq

/-- Pigeonhole theorem for one more input than output: a global
    injectivity claim is mathematically impossible in this setting. It
    supplies NO efficient collision algorithm. -/
theorem finite_output_collision (n : Nat) (H : Fin (n + 1) → Fin n) :
    ∃ a b, Collision H a b := by
  classical
  by_contra hc
  have hinj : Function.Injective H := by
    intro a b heq
    by_contra hne
    exact hc ⟨a, b, hne, heq⟩
  have hle : n + 1 ≤ n := by
    simpa only [Fintype.card_fin] using
      (Fintype.card_le_of_injective H hinj)
  omega

/-- A break of a finite *internal* transition does not automatically
    furnish a valid message-level collision. This assertion is merely the
    logical distinction between an arbitrary-state map and its restricted
    message-domain image. -/
theorem reachable_collision_requires_two_messages
    {Msg State Digest : Type*} (absorb : Msg → State)
    (digest : State → Digest) (a b : Msg)
    (h : Collision (fun m => digest (absorb m)) a b) :
    a ≠ b ∧ digest (absorb a) = digest (absorb b) := h

end WaveLockProofs
