import WaveLockProofs.BytePadding
import WaveLockProofs.BlockAbsorption
import WaveLockProofs.Protocol

/-!
# Complete multi-block absorption, squeeze-input structure, conformance set, and the
# impossibility of universal collision-freeness

**Multi-block absorption** (`reference.absorb`). For `k = 0 … nb-1`: inject block `k` (rate gain
1 for `ψ` and `G = 7` for `C`; counter digits with gain 7 in `CAP0` and `CAP2`; tag `D_TAG` in
`CAP1` iff `k = nb-1`), then apply `T = 32` coupled rounds from the global round index. The
round index is never reset. Here the round map is a parameter `evolveT ri s`.
`concreteEvolveT` instantiates it with `Protocol.runFrom ri 32` on the flat↔torus layout
`x = 16·i + j`.

Proved:
* `absorb_clock`: after absorbing, the global round index is exactly `32·nb`.
* `absorb_unfold_succ`: the exact one-block transition with the correct `last` flag.
* `tag_only_on_last`: the final tag is added at block `k` iff `k = nb-1`, and never at block 0.

**Squeeze structure** (`reference.squeeze`). Each squeeze round outputs the 64 bits
`[C[t] > C[t+128]]`, `t < 64`, comparing canonical representatives in `[0, p)`.
* `squeezeRound_ignores`: one squeeze round does not read `ψ` at all, or any accumulator cell
  outside `{0..63} ∪ {128..191}`. Two states differing only there give identical bits.

**Counterexample.** `no_universal_collision_freedom`: *any* function from admissible messages to
32-byte digests (in particular `cc_hash`) has two distinct admissible messages with equal
digests. So "collision-free" can only ever be a *computational* hardness assumption. It is not
proved, and it is not provable as an unconditional universal property.

**Conformance set** (`conformance_*`): byte-exact padding and packing values and arithmetic
constants, checked against `conformance_vectors.json`. That file is regenerated from the Python
reference by `scripts/gen_lean_conformance.py`, and `tests/test_lean_conformance.py` checks it.
This set covers only the listed vectors. It is **not** a proof that the full Python
`cc_hash` matches this model.
-/

namespace WaveLockProofs

/-- `2^31 - 1` is prime (kernel-checked by `norm_num`), so `CCField` is a field. -/
instance ccPrime : Fact (Nat.Prime 2147483647) := ⟨by norm_num⟩

abbrev CCState := (Fin 256 → CCField) × (Fin 256 → CCField)

/-- `reference._inject_block`. -/
def injectBlock (s : CCState) (blk : Fin 64 → CCField) (k : Nat) (last : Bool) : CCState :=
  (absorbWaveBlock s.1 blk k last, absorbAccumulatorBlock s.2 blk k last)

/-- One absorb step for block `k`, starting from global round `ri`. -/
def absorbStep (evolveT : Nat → CCState → CCState) (P : List Byte) (nb k : Nat)
    (st : CCState × Nat) : CCState × Nat :=
  (evolveT st.2 (injectBlock st.1 (packBlockField P k) k (k == nb - 1)), st.2 + 32)

/-- Process the remaining `n` blocks `nb-n, …, nb-1`. -/
def absorbAux (evolveT : Nat → CCState → CCState) (P : List Byte) (nb : Nat) :
    Nat → CCState × Nat → CCState × Nat
  | 0, st => st
  | n + 1, st => absorbAux evolveT P nb n (absorbStep evolveT P nb (nb - (n + 1)) st)

/-- `reference.absorb` from the IV state. -/
def absorb (evolveT : Nat → CCState → CCState) (iv : CCState) (m : List Byte) : CCState × Nat :=
  let P := pad m
  let nb := P.length / 192
  absorbAux evolveT P nb nb (iv, 0)

theorem absorbAux_clock (evolveT : Nat → CCState → CCState) (P : List Byte) (nb : Nat) :
    ∀ (n : Nat) (st : CCState × Nat), (absorbAux evolveT P nb n st).2 = st.2 + 32 * n
  | 0, st => by simp [absorbAux]
  | n + 1, st => by
    rw [absorbAux, absorbAux_clock evolveT P nb n]
    simp [absorbStep]; ring

/-- The global round index after absorption is `32·nb`, with `nb = paddedBlockCount |m|`. -/
theorem absorb_clock (evolveT : Nat → CCState → CCState) (iv : CCState) (m : List Byte) :
    (absorb evolveT iv m).2 = 32 * paddedBlockCount m.length := by
  simp [absorb, absorbAux_clock, pad_blocks_eq_paddedBlockCount]

theorem absorb_unfold_succ (evolveT : Nat → CCState → CCState) (P : List Byte) (nb n : Nat)
    (st : CCState × Nat) :
    absorbAux evolveT P nb (n + 1) st =
      absorbAux evolveT P nb n
        (evolveT st.2 (injectBlock st.1 (packBlockField P (nb - (n + 1))) (nb - (n + 1))
          (nb - (n + 1) == nb - 1)), st.2 + 32) := rfl

/-- The tag flag is `true` exactly at the last block, and never at block `0` of a padded
message. -/
theorem tag_only_on_last (m : List Byte) (k : Nat) :
    (k == (pad m).length / 192 - 1) = true ↔ k = paddedBlockCount m.length - 1 := by
  rw [pad_blocks_eq_paddedBlockCount]; simp

theorem tag_never_first (m : List Byte) :
    (0 == (pad m).length / 192 - 1) = false := by
  rw [pad_blocks_eq_paddedBlockCount]
  have := paddedBlockCount_at_least_two m.length
  simp; omega

/-! ## Concrete round map -/

/-- Flat index `16 i + j` ↔ torus site `(i, j)`. -/
def flatIdx (x : TorusSite) : Fin 256 :=
  ⟨x.1.val * 16 + x.2.val, by
    have h1 := x.1.isLt; have h2 := x.2.isLt
    simp only [ZMod.val] at *; omega⟩

def toTorus (s : Fin 256 → CCField) : TorusSite → CCField := fun x => s (flatIdx x)

def fromTorus (f : TorusSite → CCField) : Fin 256 → CCField :=
  fun i => f ((i.val / 16 : ℕ), (i.val % 16 : ℕ))

/-- The round-level model of `Protocol` lifted to the flat layout (`T = 32` rounds from global
round `ri`). The round equations themselves are those of `Protocol.candidateBRound`. -/
noncomputable def concreteEvolveT (ri : Nat) (s : CCState) : CCState :=
  let r := runFrom (F := CCField) ri 32 (toTorus s.1, toTorus s.2)
  (fromTorus r.1, fromTorus r.2)

/-! ## Squeeze -/

/-- One squeeze round: bit `t` is `C[t] > C[t+128]` on canonical representatives. -/
def squeezeRound (C : Fin 256 → CCField) (t : Fin 64) : Bool :=
  decide ((C ⟨t.val + 128, by omega⟩).val < (C ⟨t.val, by omega⟩).val)

def squeezeRead (i : Fin 256) : Prop := i.val < 64 ∨ (128 ≤ i.val ∧ i.val < 192)

/-- A squeeze round ignores every unread accumulator cell, and `ψ` is not an argument. -/
theorem squeezeRound_ignores (C C' : Fin 256 → CCField)
    (h : ∀ i, squeezeRead i → C i = C' i) : squeezeRound C = squeezeRound C' := by
  funext t
  have h1 := h ⟨t.val, by omega⟩ (Or.inl t.isLt)
  have h2 := h ⟨t.val + 128, by omega⟩ (Or.inr ⟨by simp, by simp; omega⟩)
  simp [squeezeRound, h1, h2]

/-- Explicit witness: changing an unread cell (here cell 64) leaves the bits unchanged. -/
theorem squeezeRound_not_injective :
    ∃ C C' : Fin 256 → CCField, C ≠ C' ∧ squeezeRound C = squeezeRound C' := by
  refine ⟨fun _ => 0, fun i => if i.val = 64 then 1 else 0, ?_, ?_⟩
  · intro h
    have := congrFun h ⟨64, by norm_num⟩
    simp only [if_true] at this
    exact zero_ne_one this
  · apply squeezeRound_ignores
    intro i hi
    rcases hi with hi | ⟨hi1, hi2⟩ <;> simp <;> omega

/-! ## Impossibility of universal collision-freeness -/

/-- **Counterexample to universal collision-freeness.** Every function from admissible messages
to 32-byte digests has a collision among 33-byte messages. -/
theorem no_universal_collision_freedom (H : List Byte → Fin 32 → Byte) :
    ∃ m₁ m₂ : List Byte, admissible m₁ ∧ admissible m₂ ∧ m₁ ≠ m₂ ∧ H m₁ = H m₂ := by
  have hcard : Fintype.card (Fin 32 → Byte) < Fintype.card (Fin 33 → Byte) := by
    simp only [Fintype.card_fun, Fintype.card_fin]
    norm_num
  obtain ⟨a, b, hab, he⟩ :=
    Fintype.exists_ne_map_eq_of_card_lt (fun v : Fin 33 → Byte => H (List.ofFn v)) hcard
  refine ⟨List.ofFn a, List.ofFn b, ?_, ?_, ?_, he⟩
  · simp [admissible]
  · simp [admissible]
  · intro h; exact hab (List.ofFn_injective h)

/-! ## Conformance set (values from `conformance_vectors.json`) -/

theorem conformance_pad_lengths :
    (pad []).length = 384 ∧
    (pad [97, 98, 99]).length = 384 ∧
    (pad (List.replicate 191 0)).length = 384 ∧
    (pad (List.replicate 192 255)).length = 576 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;>
    simp only [pad_length, List.length_replicate, List.length_cons, List.length_nil,
      padZeros] <;> norm_num

theorem conformance_length_field :
    be8 (8 * 0) = [0, 0, 0, 0, 0, 0, 0, 0] ∧
    be8 (8 * 3) = [0, 0, 0, 0, 0, 0, 0, 24] ∧
    be8 (8 * 191) = [0, 0, 0, 0, 0, 0, 5, 248] ∧
    be8 (8 * 192) = [0, 0, 0, 0, 0, 0, 6, 0] := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide

theorem conformance_abc_block0 :
    packBlockNat (pad [97, 98, 99]) 0 0 = 6513249 ∧
    packBlockNat (pad [97, 98, 99]) 0 1 = 1 ∧
    packBlockNat (pad [97, 98, 99]) 0 2 = 0 ∧
    packBlockNat (pad []) 0 0 = 1 := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> decide

theorem conformance_constants :
    (candidateBWeight 0 ((0 : ZMod 16), (0 : ZMod 16)) : CCField) = 150538 ∧
    (candidateBWeight 31 ((15 : ZMod 16), (15 : ZMod 16)) : CCField) = 505727969 ∧
    (candidateBRoundConstant 0 : CCField) = 1464025904 ∧
    (candidateBRoundConstant 100 : CCField) = 619629776 ∧
    ccCounter 0 = (1, 0) ∧ ccCounter 2147483646 = (0, 1) ∧
    (candidateBInjection 11 (5 : CCField) 7) = 390 := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp only [candidateBWeight, candidateBRoundConstant, ccCoreRho0, ccCoreRho1, ccCounter,
      counterDigits, candidateBInjection] <;> decide

end WaveLockProofs
