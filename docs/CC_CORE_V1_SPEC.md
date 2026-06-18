# WaveLock CC-Core-v1 / Candidate B — Specification

**Version identifier:** `CC-Core-v1-B`
**Status:** EXPERIMENTAL research candidate. No security claim.
**Package:** `wavelock/curvature_capacity_v1/`
**Relationship:** Linear-injection revision of Candidate A (`CC-Core-v0-A`,
`wavelock/curvature_capacity/`). Candidate A is frozen (see
`docs/CC_CORE_V0_BASELINE.md`) and is never overwritten.

---

## 1. The single algebraic change

```
Candidate A (CC-Core-v0-A):  j_A(u, v) = u + GAMMA·u·v + ETA·u² + ZETA·v
Candidate B (CC-Core-v1-B):  j_B(u, v) = u + GAMMA·u·v          = u·(1 + GAMMA·v)
```

Candidate B sets **ETA = 0 and ZETA = 0**, leaving a single multiplicative factor
`(1 + GAMMA·v)`. This removes Candidate A's proved generic 2-to-1 relation (the
symmetric `ETA·u²` term) but introduces a **singular multiplicative hyperplane**
at `v = V_STAR` (§7), which the algebra doc (`CC_CORE_V1_ALGEBRA.md`) and the
singular-hyperplane audit (Part IV) analyze before any preference claim.

Everything else — modulus, lattice, wave round, neighbor coupling, weights, round
constants, initialization, finalization, squeeze — is **identical to Candidate A**,
deliberately, so an A/B comparison from the same initial wave state isolates the
single change.

---

## 2. Field, lattice, dimensions

| Item | Value |
|---|---|
| Modulus p | 2³¹ − 1 = 2147483647 |
| Lattice | 16 × 16 torus, N_CELLS = 256 |
| Fields | ψ (wave) and C (accumulator), each 256 cells in F_p |
| Rounds per block | T = 32 |

## 3. Constants

Identical to Candidate A except `ETA=0`, `ZETA=0`, `D_TAG`, `VERSION`:

| Constant | Value |
|---|---|
| D, A, B | 5, 3, 1431655765 (wave; frozen Design A) |
| D_C | 3 |
| GAMMA (γ) | 11 |
| ETA (η) | **0** (was 13 in A) |
| ZETA (ζ) | **0** (was 17 in A) |
| A_C | 2 |
| MU (μ) | 5 |
| RHO0 | 0x57434330 = 1463898160 |
| RHO1 | 2654435761 mod p = 507468114 |
| WA, WB, WC | 40503, 50021, 60013 |
| G | 7 |
| D_TAG | **0x57434332 ("WCC2")** = 1463898162 (was WCC1 in A) |
| CAP0, CAP1, CAP2 | 64, 65, 66 |
| RATE / BYTES_PER_BLOCK | 64 / 192 |
| VERSION | **"WaveLock-CC-Core-v1"** |

## 4. Wave round (frozen Design A, unchanged)

```
F(ψ)[x] = ψ[x] + D·Lap(ψ)[x] + A·ψ[x]·(B − ψ[x]²)   (mod p)
```
Byte-for-byte identical to `wavelock.pde_hash` (asserted by parity test).

## 5. Neighbor coupling

The accumulator uses the same toroidal 5-point Laplacian as the wave field:
```
Lap(C)[x] = C[x−N] + C[x+N] + C[x−1] + C[x+1] + (p−4)·C[x]   (mod p)
```

## 6. Accumulator transition Φ_t^(B)

Per cell x, with u = ψ_t[x], v = ψ_{t+1}[x]:

```
j_B(u, v) = u + GAMMA·u·v                                  (mod p)
Cd[x]     = C[x] + D_C·Lap(C)[x]                           (mod p)
W_t(x)    = 1 + (t+1)·WA + (x+1)·WB + (t+1)·(x+1)·WC        (mod p)
rho_t     = RHO0 + RHO1·t                                  (mod p)
C_{t+1}[x] = MU·Cd[x] + A_C·Cd[x]² + W_t(x)·j_B(u,v) + rho_t  (mod p)
```

Coupled round: `(ψ_{t+1}, C_{t+1}) = (F(ψ_t), Φ_t^(B)(C_t, ψ_t, ψ_{t+1}))`.

## 7. Time-domain separation

- `rho_t = RHO0 + RHO1·t` makes the update round-index-dependent (order sensitivity).
- `W_t(x)` depends on both `t` and `x` (breaks lattice translation/sign symmetry
  AND makes the injection round-dependent).
- The squeeze re-evolves the coupled system between output reads.

## 8. Initialization

- ψ IV: base-256 injection of `b"WaveLock-CC-Core-v0:psi"` (shared with A; isolates
  the injection in trajectory comparison).
- C IV: base-256 injection of `b"WaveLock-CC-Core-v0:acc"` (shared with A).
- No hash is used.

## 9. Message / block domain separation

- Block size 192 bytes; rate = 64 cells × 3 bytes.
- Per block: additive rate write into ψ; G·(block) into C; injective base-P block
  counter into CAP0/CAP2 of both fields.
- Finalization: last block adds `D_TAG = 0x57434332` into CAP1 of both fields.
  **This distinct D_TAG domain-separates Candidate B's message hashing from
  Candidate A.**

## 10. Finalization & squeeze

256 output bits via disjoint comparison pairs `(t, t+128)` for t in 0..63 read
from C, re-evolving the coupled system between reads (squeeze is part of the
dynamics). MSB-first bit packing.

## 11. The singular hyperplane (new structural risk)

For fixed v, j_B(·, v) is **linear** in u with slope `(1 + GAMMA·v)`:
- If `(1 + GAMMA·v) ≠ 0 mod p`: j_B is **injective in u** (slope nonzero).
- If `v = V_STAR := −GAMMA⁻¹ mod p = 195225786`: slope is 0 and
  `j_B(u, V_STAR) = 0` for **every** u — the injection erases all u-information
  at that coordinate.

This is the new risk Candidate B trades for removing the 2-to-1 relation. It is
analyzed in `docs/CC_CORE_V1_ALGEBRA.md` and attacked in Part IV
(`candidate_b_singular_hyperplane.json`). **No preference claim is made until that
audit completes.**

## 12. Pinned test vectors (Candidate B)

`cc_hash` (reference == optimized):

| message | digest (hex) |
|---|---|
| `b""` | `eb6402bb517d4d2ef409b6a1a16093cdae06462a548d240fe6783f0aec2216bd` |
| `b"\x00"` | `0ef50d4fecc6b7696a8bc0d06bcd73a1341bf9c854b19269b5a3c69994b29b84` |
| `b"\x01"` | `93815bbb3df1c3827c92e30daa96b97bffe18cc2b8e7a63f53f391de3da0110a` |
| `b"\xff"` | `3d0a0267f4223179c0a18f9a0a8dd5fa40d0583344d3a5ebdbe2dc529c1509c2` |
| `b"abc"` | `223797841b2e201aa2c3bfa40623306fbee14e420d5741b5d1a04309a4936922` |
| `b"WaveLock"` | `9e14e1cf32e2285be0f1d402e697d79171aa91138c8700897b72e330f4a708b9` |
| `b"\x00"*192` | `116d4e011f2aaad4bf31561db7fd45635d3b4ba4667345a2a91ec1ce46cd8264` |
| `b"\x00"*193` | `31785af8b1f9edcbb603109d2937753fd6946ce83e57424d5d1328d17899c9ff` |

Zero-state trajectory digest `trajectory_digest(zeros(16,16))`:
`c4ed8b688e14f2127c8e03ea62eee0ecd10cc15ed5dd695afc8a193efc9198d3`
(identical to Candidate A: when ψ=0, j_A = j_B = 0, so the trajectories coincide).

## 13. Explicit non-claims

- No claim that Candidate B is collision-resistant, one-way, or "provably secure".
- No claim that Candidate B dominates Candidate A until Parts IV–XV complete.
- No claim that the singular hyperplane is unreachable until Part IV measures it.
