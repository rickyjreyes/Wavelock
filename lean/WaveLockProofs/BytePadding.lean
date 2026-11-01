import WaveLockProofs.ExactField

/-!
# Byte-exact CC-Core-v1 padding, block packing, and encoding injectivity

This file gives an exact model of `wavelock/curvature_capacity_v1/reference.py`:

```
pad(m)        = m ‖ 0x01 ‖ 0x00^z ‖ (8·|m| as 8 bytes big-endian) ‖ 0x00^184,
                z = (192 - (|m|+1) mod 192) mod 192
pack_block(P, k)[c] = P[192k+3c] + 2^8·P[192k+3c+1] + 2^16·P[192k+3c+2],   c < 64
```

Python rejects messages with `8·|m| > 2^64 - 1` (`MAX_INPUT_BITS`). That is the
hypothesis `admissible`.

Proved:
* `pad_length`, `pad_length_mod`, `pad_blocks_eq_paddedBlockCount`: the exact padded length,
  matching the existing `paddedBlockCount` (so there are at least two blocks).
* `be8_decode`: the 8-byte big-endian length field decodes back to `n mod 2^64`.
* `pad_injective`: padding is injective on admissible messages.
* `bytes_eq_of_packed_eq`: equal packed 24-bit rate words at every block and cell force equal
  padded byte strings of equal 192-multiple length.
* `encoding_injective`: **end-to-end encoding injectivity.** The block count together with
  every rate-cell value *as an element of `ZMod (2^31-1)`* determines the admissible message.
  So the sponge input is uniquely decodable before any wave round.

Out of scope: injectivity of absorption *after* wave rounds, and any collision or preimage
resistance of `cc_hash`.
-/

namespace WaveLockProofs

open List

abbrev Byte := Fin 256

/-- Python: `8·len(m) ≤ MAX_INPUT_BITS = 2^64 - 1`. -/
def admissible (m : List Byte) : Prop := 8 * m.length < 2 ^ 64

/-- Zero-fill length after the `0x01` marker. -/
def padZeros (len : Nat) : Nat := (192 - (len + 1) % 192) % 192

/-- `n.to_bytes(8, "big")` for `n < 2^64` (reduced mod `2^64` in general). -/
def be8 (n : Nat) : List Byte :=
  [7, 6, 5, 4, 3, 2, 1, 0].map fun i => ⟨n / 256 ^ i % 256, Nat.mod_lt _ (by norm_num)⟩

/-- Big-endian decoding. -/
def beDecode (l : List Byte) : Nat := l.foldl (fun acc b => 256 * acc + b.val) 0

theorem be8_length (n : Nat) : (be8 n).length = 8 := rfl

theorem be8_decode (n : Nat) : beDecode (be8 n) = n % 2 ^ 64 := by
  simp only [be8, beDecode, List.map, List.foldl]
  norm_num
  omega

/-- The 192-byte length block. -/
def lengthBlock (len : Nat) : List Byte := be8 (8 * len) ++ replicate 184 0

theorem lengthBlock_length (len : Nat) : (lengthBlock len).length = 192 := by
  rw [lengthBlock, List.length_append, be8_length, List.length_replicate]

/-- `reference.pad`. -/
def pad (m : List Byte) : List Byte :=
  m ++ [1] ++ replicate (padZeros m.length) 0 ++ lengthBlock m.length

theorem pad_length (m : List Byte) :
    (pad m).length = m.length + 1 + padZeros m.length + 192 := by
  simp only [pad, List.length_append, List.length_singleton, List.length_replicate,
    lengthBlock_length]

theorem pad_length_mod (m : List Byte) : (pad m).length % 192 = 0 := by
  rw [pad_length]; unfold padZeros; omega

/-- The block count is `paddedBlockCount`, which is always at least two. -/
theorem pad_blocks_eq_paddedBlockCount (m : List Byte) :
    (pad m).length / 192 = paddedBlockCount m.length := by
  rw [pad_length]; unfold padZeros paddedBlockCount; omega

theorem pad_take (m : List Byte) : (pad m).take m.length = m := by
  simp [pad, List.append_assoc]

theorem pad_lengthBlock (m : List Byte) :
    (pad m).drop ((pad m).length - 192) = lengthBlock m.length := by
  have h : (pad m).length - 192 = (m ++ [1] ++ replicate (padZeros m.length) 0).length := by
    rw [pad_length]; simp; omega
  rw [h]
  simp [pad]

/-- **Padding is injective on admissible messages.** -/
theorem pad_injective {m₁ m₂ : List Byte} (h₁ : admissible m₁) (h₂ : admissible m₂)
    (h : pad m₁ = pad m₂) : m₁ = m₂ := by
  have hL := pad_lengthBlock m₁
  rw [h, pad_lengthBlock] at hL
  have h8 := congrArg (fun l => beDecode (l.take 8)) hL
  simp only [lengthBlock, List.take_left' (be8_length _), be8_decode] at h8
  unfold admissible at h₁ h₂
  have hlen : m₁.length = m₂.length := by
    rw [Nat.mod_eq_of_lt h₁, Nat.mod_eq_of_lt h₂] at h8; omega
  have t1 := pad_take m₁
  rw [h, hlen, pad_take] at t1
  exact t1.symm

/-- `reference.pack_block`, with out-of-range reads defaulting to `0` (never used on padded
inputs). -/
def packBlockNat (P : List Byte) (k : Nat) (c : Fin 64) : Nat :=
  pack3 (P.getD (192 * k + 3 * c) 0) (P.getD (192 * k + 3 * c + 1) 0)
    (P.getD (192 * k + 3 * c + 2) 0)

/-- Field-valued rate block, as absorbed. -/
def packBlockField (P : List Byte) (k : Nat) (c : Fin 64) : CCField :=
  pack3Field (P.getD (192 * k + 3 * c) 0) (P.getD (192 * k + 3 * c + 1) 0)
    (P.getD (192 * k + 3 * c + 2) 0)

/-- Equal packed words at every block and cell force equal byte strings. -/
theorem bytes_eq_of_packed_eq {P Q : List Byte} (hl : P.length = Q.length)
    (h : ∀ k c, packBlockNat P k c = packBlockNat Q k c) : P = Q := by
  apply List.ext_getElem hl
  intro i hiP hiQ
  set k := i / 192
  have hc : (i % 192) / 3 < 64 := by omega
  set c : Fin 64 := ⟨(i % 192) / 3, hc⟩
  have hi : i = 192 * k + 3 * c.val + i % 3 := by simp [k, c]; omega
  obtain ⟨e0, e1, e2⟩ := pack3_unique _ _ _ _ _ _ (h k c)
  have key : P.getD i 0 = Q.getD i 0 := by
    rcases (by omega : i % 3 = 0 ∨ i % 3 = 1 ∨ i % 3 = 2) with r | r | r
    · rw [hi, r]; simpa using e0
    · rw [hi, r]; simpa using e1
    · rw [hi, r]; simpa using e2
  rwa [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hiP,
    List.getElem?_eq_getElem hiQ, Option.getD_some, Option.getD_some] at key

/-- **End-to-end encoding injectivity.** Two admissible messages with the same number of padded
blocks and the same field-valued rate block at every index are equal. -/
theorem encoding_injective {m₁ m₂ : List Byte} (h₁ : admissible m₁) (h₂ : admissible m₂)
    (hnb : (pad m₁).length / 192 = (pad m₂).length / 192)
    (hblk : ∀ k c, packBlockField (pad m₁) k c = packBlockField (pad m₂) k c) :
    m₁ = m₂ := by
  apply pad_injective h₁ h₂
  have hl : (pad m₁).length = (pad m₂).length := by
    have a := pad_length_mod m₁; have b := pad_length_mod m₂; omega
  apply bytes_eq_of_packed_eq hl
  intro k c
  obtain ⟨e0, e1, e2⟩ := pack3Field_unique _ _ _ _ _ _ (hblk k c)
  simp only [packBlockNat, e0, e1, e2]

end WaveLockProofs
