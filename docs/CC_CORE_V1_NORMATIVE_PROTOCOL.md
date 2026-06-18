# CC-Core-v1 / Candidate B — Normative Protocol (frozen for Phase CC-3)

**Version identifier:** `CC-Core-v1-B` (`VERSION = "WaveLock-CC-Core-v1"`).
**Status:** EXPERIMENTAL. No security claim. This document fixes the *exact*
deterministic protocol against which "reachability" is defined in Phase CC-3.
It describes the existing implementation (`wavelock/curvature_capacity_v1/`); it
adds no new behavior and changes no pinned vector.

---

## 1. Field, lattice, constants

- Modulus `p = 2³¹ − 1 = 2147483647`.
- Lattice: 16×16 torus, `N_CELLS = 256`, two fields ψ (wave) and C (accumulator).
- `T = 32` coupled rounds per block.
- `GAMMA = 11`, `D_C = 3`, `A_C = 2`, `MU = 5`, `D = 5`, `A = 3`, `B = 1431655765`.
- `ETA = 0`, `ZETA = 0` (Candidate B linear injection `j_B = u(1+γv)`).
- `RHO0 = 0x57434330`, `RHO1 = 2654435761 mod p`, `WA,WB,WC = 40503,50021,60013`.
- `G = 7`, `D_TAG = 0x57434332` ("WCC2"), `CAP0,CAP1,CAP2 = 64,65,66`, `RATE = 64`.
- Singular value `v_star = −GAMMA⁻¹ mod p = 195225786`.

## 2. Message byte domain and length

- A valid message `m` is a finite byte string, `m ∈ {0,…,255}*`.
- Minimum length: 0 bytes (the empty message is valid).
- Maximum length: `(2⁶⁴ − 1)` bits = `2⁶¹ − 1` bytes (length field is 64-bit).
- No other byte restriction; all 256 byte values are valid at every position.

## 3. Padding (exact)

```
pad(m):
  out = m
  out += 0x01                                   # domain bit
  out += 0x00 * ((−len(out)) mod 192)           # zero-fill to block boundary
  lb  = 192 zero bytes; lb[0:8] = (8·len(m)) big-endian   # length block
  out += lb
```
`BYTES_PER_BLOCK = 192`. After padding, `len(pad(m))` is a positive multiple of
192. The number of blocks is `nb = len(pad(m)) / 192 ≥ 1` (the empty message pads
to exactly 2 blocks: one `0x01`-prefixed zero block? No — `len("")=0`, append
0x01 → 1 byte, zero-fill to 192 → 192 bytes, append 192-byte length block → 384
bytes → `nb = 2`).

## 4. Block packing (byte → field element)

For block `k`, rate cell `c ∈ {0,…,63}` reads 3 consecutive bytes at offset
`192k + 3c`:
```
elem_k[c] = byte0 + (byte1 << 8) + (byte2 << 16)        ∈ [0, 2²⁴) = [0, 16777216)
```
Only the 64 rate cells receive a packed element; **each packed element lies in
the strict subinterval `[0, 2²⁴)` of F_p.**

## 5. Initial states (IVs; no hash)

```
ψ_IV[x] = (1 + x + TAG_PSI[x mod |TAG_PSI|]) mod p,   TAG_PSI = b"WaveLock-CC-Core-v0:psi"
C_IV[x] = (1 + x + TAG_C  [x mod |TAG_C|])   mod p,   TAG_C   = b"WaveLock-CC-Core-v0:acc"
```
Concretely `ψ_IV[x] ∈ [54, 374]` (small constants). These IVs are shared with
Candidate A deliberately (controlled comparison); message-level separation comes
from `D_TAG`.

## 6. Block absorption rule (exact, mutates the running state)

For block `k` (0-based), with `q = k+1`, `q0 = q mod p`, `q1 = q // p`:
```
for c in 0..63:  ψ[c] += elem_k[c]            (mod p)
                 C[c] += G · elem_k[c]         (mod p)
ψ[CAP0] += q0·G ;  ψ[CAP2] += q1·G            (mod p)
C[CAP0] += q0·G ;  C[CAP2] += q1·G            (mod p)
if k == nb−1:  ψ[CAP1] += D_TAG ;  C[CAP1] += D_TAG   (mod p)
```
**Coordinates `67..255` of ψ are never written by absorption** (they evolve only
through the wave round). `CAP1 = 65` receives `D_TAG = 1463898162` on the last
block only.

## 7. Coupled evolution per block

After absorbing block `k`, run `T = 32` coupled rounds; the global round index
`ri` is **not reset** between blocks (it continues 0,1,2,… across the whole
message), so `rho_t` and `W_t(x)` differ at every absolute round.

```
coupled_round(ψ, C, t):
    ψ' = F(ψ)                                  # frozen Design A wave round
    C'[x] = MU·cd[x] + A_C·cd[x]² + W_t(x)·j_B(ψ[x], ψ'[x]) + rho_t   (mod p)
            cd = C + D_C·Lap(C);  j_B(u,v) = u + GAMMA·u·v
```

## 8. The deterministic trajectory map

For a one-block message the trajectory is
```
m ↦ ψ_0(m) = (absorb block 0 into ψ_IV)
   ↦ ψ_1(m) = F(ψ_0(m))
   ↦ … ↦ ψ_T(m) = Fᵀ(ψ_0(m)).
```
For multi-block messages, after `ψ_T` of block 0 the next block is absorbed
(Section 6) into the *current* (ψ, C) and another `T` rounds run; the round index
continues. **`ψ_0(m)` denotes the post-absorption, pre-first-wave-round state of
block 0.**

## 9. Finalization & squeeze

After the last block's `T` rounds, the 256-bit digest is read from C via the 64
disjoint comparison pairs `(t, t+128)`, re-evolving the coupled system between
output reads. MSB-first packing. (Unchanged from the spec; not relevant to ψ
reachability.)

## 10. Validity and verifier behavior

- **Valid inputs are byte strings only.** Arbitrary internal lattice states
  (ψ, C) are **NOT** valid protocol inputs; they may be studied mathematically
  but are not producible by `cc_hash`.
- A verifier recomputes `cc_hash_B(m)` under this spec and accepts iff the
  version identifier matches `"WaveLock-CC-Core-v1"` and the digest matches.
- Invalid encodings: none at the byte level (all byte strings are valid); a
  transcript with a mismatched version identifier is rejected.

## 11. Reachability stratification (used throughout Phase CC-3)

1. **Arbitrary mathematical states** — any (ψ) ∈ F_p²⁵⁶. (Superset; not protocol.)
2. **Valid-message reachable** — `R = ⋃_{t≥0} F^t(R_0)` where `R_0` is the set of
   post-absorption states of valid messages.
3. **Reachable after one valid block** — `R_0` and its forward orbit within the
   first 32 rounds.
4. **Reachable after multiple blocks** — the full `R`.

**Key structural fact (proved in `CC_CORE_V1_VSTAR_REACHABILITY.md`):** every
coordinate of `ψ_0(m)` lies in `[0, 16777397] ∪ {fixed CAP/IV constants}`, all
strictly below `v_star = 195225786`. Hence no coordinate of `ψ_0(m)` can equal
`v_star`: the singular hyperplane is unreachable *before the first wave round* for
every valid message. Reachability at rounds ≥ 1 is studied empirically and with
solvers (Parts IV–VIII).
