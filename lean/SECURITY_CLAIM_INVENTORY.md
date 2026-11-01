# WaveLock CC-Core-v1-B: security-claim inventory (Task 8)

The table separates three kinds of claim: **(M)** mathematical correctness, which is
kernel-checked; **(A)** cryptographic assumptions, which are stated and not proved; and **(I)**
implementation security, which is out of scope for Lean.

Build: `cd lean && lake exe cache get && lake build` (Lean v4.23.0, Mathlib v4.23.0). There is
no `sorry`, `admit` or local axiom. `#print axioms` on the theorems below lists only `propext`,
`Classical.choice` and `Quot.sound`.

## (M) Proved, exact

| Area | Theorem(s) | File |
|---|---|---|
| Byte padding length / block count ≥ 2 | `pad_length`, `pad_length_mod`, `pad_blocks_eq_paddedBlockCount`, `paddedBlockCount_at_least_two` | BytePadding, Encoding |
| Length field (8-byte big-endian) decodes exactly | `be8_decode` | BytePadding |
| Padding injective on admissible inputs (`8·|m| < 2^64`) | `pad_injective` | BytePadding |
| Packed rate words determine padded bytes | `bytes_eq_of_packed_eq`, `pack3_unique` | BytePadding, Encoding |
| **Sponge input uniquely decodable** (block count + field-valued rate blocks ⇒ message) | `encoding_injective` | BytePadding |
| Block counter digits injective | `counterDigits_injective` | Encoding |
| Single-block injection (rate gains 1/7, counters, tag) | `absorbWaveBlock_rate`, `absorbAccumulatorBlock_rate`, … | BlockAbsorption (PR #39) |
| Multi-block absorption, global clock `32·nb` | `absorb_clock`, `absorbAux_clock`, `absorb_unfold_succ` | MultiBlock |
| Final tag only on last block, never first | `tag_only_on_last`, `tag_never_first` | MultiBlock |
| Round-level coupled update | `candidateBRound`, `runFrom_*` | Protocol (PR #39) |
| Frozen constants (hex ↔ decimal), `gcd(ρ₁, p−1) = 186` | `ccCoreRho*_decimal`, `ccCoreRho1_gcd_modulusMinusOne` | NormativeConstants (PR #39) |
| `2^31−1` prime, so `CCField` is a field | `ccPrime` | MultiBlock |
| Conformance set vs Python reference | `conformance_pad_lengths`, `conformance_length_field`, `conformance_abc_block0`, `conformance_constants` + `tests/test_lean_conformance.py` | MultiBlock |

## (M) Proved counterexamples / limits

| Proposed universal property | Status | Witness |
|---|---|---|
| `cc_hash` (any 256-bit digest map) is collision-free | **False**, unconditionally | `no_universal_collision_freedom` (33-byte inputs) |
| Injection `j_B(u,v)` injective in `u` for every `v` | **False** at `v = V* = 195225786` | `concreteCandidateB_singular_slice_not_injective` (PR #39) |
| Squeeze output depends on the whole state | **False** per round: reads only `C[0..63] ∪ C[128..191]`, never `ψ` | `squeezeRound_ignores`, `squeezeRound_not_injective` |
| Injective intermediate layers ⇒ injective digest | **False** in general | `intermediate_injective_does_not_imply_digest_injective` (PR #39) |
| `ρ₁` invertible mod `p−1` | **False** (gcd 186) | `ccCoreRho1_gcd_modulusMinusOne` |

## (A) Assumptions: not proved, and must not be presented as theorems

* Computational collision resistance, second-preimage resistance and one-wayness of `cc_hash`.
* Any post-quantum security claim.
* Any reduction from the security games (`SecondPreimageGame`, `InversionGame` in
  SecurityGames) to a named hard problem. **None has been constructed.** The games and the
  experiments are defined, but no reduction exists in this repository.

## (I) Implementation: outside Lean

* Byte-for-byte agreement of the full `cc_hash` (all 32·nb rounds + squeeze) between Python and
  this model is verified **only** on the listed conformance vectors and the round-level
  equations. A full execution-semantics proof is open.
* Side channels, constant-time behaviour, NumPy/CPU dispatch reproducibility, and key or bounty
  operations.

## Open obligations
1. Prove `concreteEvolveT` equals `coupled_evolve_T` on the flat layout for all states, i.e.
   the Laplacian indexing and the wave-update constants against `evolve.py`.
2. Formal squeeze to 256 bits (4 squeeze rounds with evolution in between) and bit packing.
3. Restricted binding statements beyond the first block (#32), and the replay interaction with
   the accumulator across blocks.
4. Any conditional reduction, with the hardness assumption stated as a hypothesis.
