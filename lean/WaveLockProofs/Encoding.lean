import Mathlib

/-!
# CC-Core-v1-B byte packing and restricted first-block reachability

These are exact *integer* facts for the input packing and initial rate-cell
bounds in docs/CC_CORE_V1_NORMATIVE_PROTOCOL.md. The byte representation is
Fin 256, the canonical three-byte packing is little-endian. None of these
theorems treats modular addition after later nonlinear rounds.
-/

namespace WaveLockProofs

/-- Three bytes packed into a single rate-cell integer (little-endian). -/
def pack3 (b0 b1 b2 : Fin 256) : Nat :=
  b0.val + 256 * b1.val + 65536 * b2.val

/-- A packed rate cell is strictly smaller than 2^24. -/
theorem pack3_lt_2pow24 (b0 b1 b2 : Fin 256) :
    pack3 b0 b1 b2 < 16777216 := by
  have h0 := b0.isLt
  have h1 := b1.isLt
  have h2 := b2.isLt
  dsimp [pack3]
  omega

/-- The three-byte packing is injective; no two byte triples alias as Nat. -/
theorem pack3_unique
    (a b c d e f : Fin 256) (h : pack3 a b c = pack3 d e f) :
    a = d ∧ b = e ∧ c = f := by
  have ha := a.isLt
  have hb := b.isLt
  have hc := c.isLt
  have hd := d.isLt
  have he := e.isLt
  have hf := f.isLt
  dsimp [pack3] at h
  have hv : a.val = d.val ∧ b.val = e.val ∧ c.val = f.val := by
    omega
  exact ⟨Fin.ext hv.1, Fin.ext hv.2.1, Fin.ext hv.2.2⟩

/-- Rate-cell initial IV bytes satisfy a documented bound <= 374. -/
def firstBlockRateValue (iv : Fin 375) (b0 b1 b2 : Fin 256) : Nat :=
  iv.val + pack3 b0 b1 b2

/-- The first absorbed rate-cell integer never reaches Candidate B's
    singular value, and modular reduction does not occur there. -/
theorem firstBlockRate_below_singular
    (iv : Fin 375) (b0 b1 b2 : Fin 256) :
    firstBlockRateValue iv b0 b1 b2 < 195225786 := by
  have hi := iv.isLt
  have hp := pack3_lt_2pow24 b0 b1 b2
  dsimp [firstBlockRateValue]
  omega

theorem firstBlockRate_no_wrap
    (iv : Fin 375) (b0 b1 b2 : Fin 256) :
    firstBlockRateValue iv b0 b1 b2 < 2147483647 := by
  have h := firstBlockRate_below_singular iv b0 b1 b2
  omega

theorem firstBlockRate_not_singular
    (iv : Fin 375) (b0 b1 b2 : Fin 256) :
    firstBlockRateValue iv b0 b1 b2 ≠ 195225786 := by
  have h := firstBlockRate_below_singular iv b0 b1 b2
  omega

/-- Exact base-p two-digit encoding before the field casts. -/
def counterDigits (p k : Nat) : Nat × Nat :=
  ((k + 1) % p, (k + 1) / p)

theorem counterDigits_injective (p : Nat) :
    Function.Injective (counterDigits p) := by
  intro a b heq
  have hmod : (a + 1) % p = (b + 1) % p :=
    congrArg Prod.fst heq
  have hdiv : (a + 1) / p = (b + 1) / p :=
    congrArg Prod.snd heq
  have hs : a + 1 = b + 1 := by
    calc
      a + 1 = (a + 1) % p + p * ((a + 1) / p) :=
        (Nat.mod_add_div (a + 1) p).symm
      _ = (b + 1) % p + p * ((b + 1) / p) := by rw [hmod, hdiv]
      _ = b + 1 := Nat.mod_add_div (b + 1) p
  omega

/-- Message padding appends a 0x01 marker, fills to the next 192-byte
    boundary, and then appends an entire length block. Its block count
    is therefore at least TWO, including for the empty message. -/
def paddedBlockCount (messageBytes : Nat) : Nat :=
  (messageBytes + 192) / 192 + 1

theorem paddedBlockCount_at_least_two (messageBytes : Nat) :
    2 ≤ paddedBlockCount messageBytes := by
  dsimp [paddedBlockCount]
  omega

/-- Consequently the first block is never the final block, so CAP1's
    finalization tag is NOT injected into the first pre-wave-round state. -/
theorem firstBlock_not_last (messageBytes : Nat) :
    0 ≠ paddedBlockCount messageBytes - 1 := by
  have h := paddedBlockCount_at_least_two messageBytes
  omega

end WaveLockProofs
