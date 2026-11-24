import WaveLockProofs.OTSVerification

/-!
# One-time forgery event reduction

For a fixed authenticated key and one signing query, acceptance of a distinct
normalized byte message implies a digest collision or an opening of a half not
released by that query. This file proves event containment and a finite uniform
coin-space count bound. It does not assume that the actual seed-derived slices
are independent, does not assign a security advantage to SHAKE, and does not
identify the Python opaque slice checks with the typed opening predicate.
-/

namespace WaveLockProofs.OTSSecurityReduction
open scoped BigOperators

/-- Message-level forgery, not strong signature unforgeability. The public key
is fixed; openings may be arbitrary byte strings, not just honest slice sizes. -/
def Forgery {n : Nat} {Msg Slice Commit : Type}
    (D : Msg → Fin n → Bool) (H : Fin n → Bool → Slice → Commit)
    (secret : Fin n → Bool → Slice) (signed forged : Msg)
    (response : Fin n → Slice) : Prop :=
  forged ≠ signed ∧
    ∀ i, H i (D forged i) (response i) = H i (D forged i) (secret i (D forged i))

/-- The guessed half was not disclosed, and the returned bytes open its
published commitment. An alternative preimage suffices. -/
def Unopened {n : Nat} {Msg Slice Commit : Type}
    (D : Msg → Fin n → Bool) (H : Fin n → Bool → Slice → Commit)
    (secret : Fin n → Bool → Slice) (signed forged : Msg)
    (response : Fin n → Slice) (j : Fin n × Bool) : Prop :=
  D signed j.1 ≠ j.2 ∧ D forged j.1 = j.2 ∧
    H j.1 j.2 (response j.1) = H j.1 j.2 (secret j.1 j.2)

/-- No probabilistic assumption is required for this extraction disjunction. -/
theorem forgery_event_cover {n : Nat} {Msg Slice Commit : Type}
    (D : Msg → Fin n → Bool) (H : Fin n → Bool → Slice → Commit)
    (secret : Fin n → Bool → Slice) (signed forged : Msg)
    (response : Fin n → Slice)
    (h : Forgery D H secret signed forged response) :
    Collision D forged signed ∨ ∃ j, Unopened D H secret signed forged response j := by
  classical
  by_cases hd : D forged = D signed
  · exact Or.inl ⟨h.1, hd⟩
  · have hi : ∃ i, D forged i ≠ D signed i := by
      by_contra hn
      apply hd
      funext i
      by_contra hne
      exact hn ⟨i, hne⟩
    obtain ⟨i, hi⟩ := hi
    exact Or.inr ⟨(i, D forged i), Ne.symm hi, rfl, h.2 i⟩

/-- Exact numerator of an event probability on a finite uniform sample space.
Nonuniform experiments must be represented by an appropriate sampling map
from the experiment's random coins, not by counting distinct transcripts. -/
noncomputable def eventCount {Ω : Type} [Fintype Ω] (E : Ω → Prop) : Nat := by
  classical
  exact ∑ w : Ω, if E w then 1 else 0

/-- Finite union bound for a distinguished event and indexed extraction events.
No disjointness or independence is needed. -/
theorem event_count_cover {Ω I : Type} [Fintype Ω] [Fintype I]
    (A C : Ω → Prop) (P : I → Ω → Prop)
    (cover : ∀ w, A w → C w ∨ ∃ i, P i w) :
    eventCount A ≤ eventCount C + ∑ i : I, eventCount (P i) := by
  classical
  unfold eventCount
  calc
    (∑ w : Ω, if A w then 1 else 0) ≤
        ∑ w : Ω, ((if C w then 1 else 0) + ∑ i : I, if P i w then 1 else 0) := by
      apply Finset.sum_le_sum
      intro w _
      by_cases ha : A w
      · rcases cover w ha with hc | ⟨i, hi⟩
        · simp [ha, hc]
        · have hsingle : (if P i w then (1 : Nat) else 0) ≤
              ∑ j : I, if P j w then 1 else 0 :=
            Finset.single_le_sum (f := fun j : I => if P j w then (1 : Nat) else 0)
              (fun _ _ => Nat.zero_le _) (Finset.mem_univ i)
          rw [if_pos hi] at hsingle
          rw [if_pos ha]
          omega
      · simp [ha]
    _ = _ := by rw [Finset.sum_add_distrib, Finset.sum_comm]

/-- The event-level reduction over arbitrary finite random coins, including
adversary randomness and correlated key material. No independence is needed
until one converts the unopened events into standard inversion games. -/
theorem finite_forgery_count {n : Nat} {Ω Msg Slice Commit : Type} [Fintype Ω]
    (D : Msg → Fin n → Bool) (H : Fin n → Bool → Slice → Commit)
    (secret : Ω → Fin n → Bool → Slice)
    (signed forged : Ω → Msg) (response : Ω → Fin n → Slice) :
    eventCount (fun w => Forgery D H (secret w) (signed w) (forged w) (response w)) ≤
      eventCount (fun w => Collision D (forged w) (signed w)) +
        ∑ j : Fin n × Bool,
          eventCount (fun w => Unopened D H (secret w) (signed w) (forged w) (response w) j) := by
  apply event_count_cover
  intro w h
  exact forgery_event_cover D H (secret w) (signed w) (forged w) (response w) h

/-- A bound for each tagged unopened event gives the exact 2n loss factor.
The hypotheses are event bounds on the SAME coin space, not claims about
standard preimage games; their computational simulation is a separate obligation. -/
theorem finite_uniform_bound {n : Nat} {Ω Msg Slice Commit : Type} [Fintype Ω]
    (D : Msg → Fin n → Bool) (H : Fin n → Bool → Slice → Commit)
    (secret : Ω → Fin n → Bool → Slice)
    (signed forged : Ω → Msg) (response : Ω → Fin n → Slice) (B : Nat)
    (hB : ∀ j : Fin n × Bool,
      eventCount (fun w => Unopened D H (secret w) (signed w) (forged w) (response w) j) ≤ B) :
    eventCount (fun w => Forgery D H (secret w) (signed w) (forged w) (response w)) ≤
      eventCount (fun w => Collision D (forged w) (signed w)) + (n * 2) * B := by
  classical
  have hs : (∑ j : Fin n × Bool,
      eventCount (fun w => Unopened D H (secret w) (signed w) (forged w) (response w) j)) ≤
        ∑ _j : Fin n × Bool, B := by
    exact Finset.sum_le_sum (fun j _ => hB j)
  have hc : (∑ _j : Fin n × Bool, B) = (n * 2) * B := by simp
  rw [hc] at hs
  exact (finite_forgery_count D H secret signed forged response).trans (Nat.add_le_add_left hs _)

end WaveLockProofs.OTSSecurityReduction
