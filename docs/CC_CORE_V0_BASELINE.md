# CC-Core-v0 / Candidate A — Frozen Baseline Declaration

**Version identifier:** `CC-Core-v0-A`
**Status:** FROZEN experimental baseline. Retained for historical comparison.
**Branch:** `research/curvature-capacity-wavelock` · PR #19 (draft, do not merge).

This document freezes Candidate A so Phase CC-2 (Candidate B / `CC-Core-v1-B`)
can be compared against an immutable reference. **Nothing in this document may
be changed by later phases.** A regression test
(`curvature_audit/test_candidate_a_frozen.py`) fails if any equation, constant,
or pinned output below changes.

---

## 1. Exact equations

The wave field ψ evolves by the **unmodified Design A round** F (frozen primitive
`wavelock/pde_hash`), duplicated verbatim in `wavelock/curvature_capacity`:

```
F(ψ)[x] = ψ[x] + D·Lap(ψ)[x] + A·ψ[x]·(B − ψ[x]²)   (mod p)
Lap(ψ)[x] = ψ[x−N] + ψ[x+N] + ψ[x−1] + ψ[x+1] + (p−4)·ψ[x]   (toroidal, mod p)
```

The path-binding accumulator C evolves by Φ_t (Candidate A), per cell x with
u = ψ_t[x], v = ψ_{t+1}[x]:

```
j_A(u, v) = u + GAMMA·u·v + ETA·u² + ZETA·v               (mod p)   [injection]
Cd[x]     = C[x] + D_C·Lap(C)[x]                           (mod p)   [self-diffusion]
W_t(x)    = 1 + (t+1)·WA + (x+1)·WB + (t+1)·(x+1)·WC       (mod p)   [position weight]
rho_t     = RHO0 + RHO1·t                                  (mod p)   [round constant]
C_{t+1}[x] = MU·Cd[x] + A_C·Cd[x]² + W_t(x)·j_A(u,v) + rho_t  (mod p)
```

Coupled round: `(ψ_{t+1}, C_{t+1}) = (F(ψ_t), Φ_t(C_t, ψ_t, ψ_{t+1}))`.

---

## 2. Exact constants

| Constant | Value |
|---|---|
| p | 2³¹ − 1 = 2147483647 |
| N | 16 (N×N = 256 cells) |
| D | 5 |
| A | 3 |
| B | 1431655765 |
| T | 32 rounds per block transform |
| D_C | 3 |
| GAMMA (γ) | 11 |
| ETA (η) | 13 |
| ZETA (ζ) | 17 |
| A_C | 2 |
| MU (μ) | 5 |
| RHO0 | 0x57434330 = 1463898160 |
| RHO1 | 2654435761 mod p = 507468114 |
| WA, WB, WC | 40503, 50021, 60013 |
| G | 7 |
| D_TAG | 0x57434331 = 1463898161 |
| CAP0, CAP1, CAP2 | 64, 65, 66 |
| RATE | 64 |
| BYTES_PER_BLOCK | 192 |
| IV_TAG_PSI | `b"WaveLock-CC-Core-v0:psi"` |
| IV_TAG_C | `b"WaveLock-CC-Core-v0:acc"` |

---

## 3. Proved 2-to-1 injection relation (Candidate A's known structural flaw)

For fixed v, j_A(·, v) is a degree-2 polynomial in u with symmetric leading term
η·u². Therefore:

```
j_A(u, v) = j_A(u', v)   ⟺   u = u'   OR   u + u' = −(1 + γ·v)·η⁻¹   (mod p)
```

**Proof:** j_A(u,v) − j_A(u',v) = (u − u')·(1 + γ·v + η·(u + u')). Over the field
F_p this product is zero iff one factor is zero. ∎

This is a **provable, generic 2-to-1 map in u** (every j_A value has at most two
u-preimages for fixed v). It holds for **every** (u, v) pair (confirmed 500/500
in `accumulator_algebraic_attacks.py`). It is the structural weakness that
Phase CC-2 Candidate B is designed to remove.

**No digest collision** was produced by exploiting this relation (200 trials,
`eta_pairing_attack`): the wave round F is not free, so changing u changes
v = F(ψ)[x], breaking the cancellation.

---

## 4. Full 47-state Phase 8J separation result

The complete Phase 8J zero-preimage family has **47 states**:
- r = 1: 8 states; r = 2: 36 states; r = 4: 2 states; plus the zero fixed point.
- All 46 nonzero states map to the all-zero state under one frozen Design A round.

Under Candidate A's trajectory commitment, **all 47 produce distinct digests**:
- distinct digests: 47 / 47
- minimum pairwise Hamming distance: **98** bits
- (artifacts: `phase8j_full_collision_family.json`, `full_family_path_binding.json`)

---

## 5. Reduced-model collision results (Candidate A)

Full joint (ψ, C) enumeration at toy scale:
- p = 3, N = 2 (6561 joint states): **5562 coupled collisions** — NOT injective.
- p = 5, N = 2 (390,625 joint states): **375,073 coupled collisions** — NOT injective.

N = 2 torus degeneracy (up==down, left==right) dominates; **not extrapolated** to
N = 16. (artifact: `reduced_exhaustive_cc1.json`)

---

## 6. Shortcut-audit results (Candidate A)

Six candidate shortcuts audited (skip-wave, backward inversion, MITM, cycle
detection, low-degree solve, 2-to-1 exploit). **None demonstrated.** All absences
are bounded (not proofs). (artifact: `shortcut_computation.json`,
doc: `CC_CORE_SHORTCUT_COMPUTATION_AUDIT.md`)

---

## 7. Allowed claims (Candidate A)

- The wave round inside Candidate A is byte-identical to frozen Design A (tested).
- Reference and optimized implementations agree byte-for-byte (tested).
- All 47 Phase 8J zero-collapse states map to distinct Candidate A trajectory
  digests (min Hamming 98).
- Forward evaluation is polynomial: O(blocks·T·N²).
- j_A is generically 2-to-1 in u (proved).
- Curvature magnitude is convention-dependent (diagnostic only).

## 8. Forbidden claims (Candidate A)

- "provably secure" / "collision-resistant" / "one-way" (no proof, no lower bound).
- "P vs NP proves security."
- "heat makes inversion impossible."
- "exponential curvature."
- "the accumulator makes the map injective" (toy collisions exist).
- any 256-bit security claim.

---

## 9. Pinned digests and test vectors (Candidate A — frozen)

`cc_hash` (reference == optimized, both pinned):

| message | digest (hex) |
|---|---|
| `b""` | `99e7beade48a10b0e4badf5dcecfa617e3a361b789c60e5afaee8c02ab55d6d2` |
| `b"\x00"` | `c44b0b91973fe2ec9af5dae3692c9e83a14fc707f0359b2e6800d429bc63e266` |
| `b"\x01"` | `8fdc46738f2d7e7167b9af90c5b992431dbdf484a11d56c43236919973061a17` |
| `b"\xff"` | `783e26c6db282e939b65894e54afc9d51d000c3f41e0dc39f83c7b886f1cc973` |
| `b"abc"` | `385600d91057f24522d1210cb4bb7c8983339cfdcb0b466e41bde2d2c93044ef` |
| `b"WaveLock"` | `afbcb55d3a54af05a370703780273e375b08269e4071920c92f3e35a3978c44e` |
| `b"\x00"*192` | `faa462019807c38ccefac61c50e79a1c6e1dfc46825d510d3526871d3a523f5f` |
| `b"\x00"*193` | `34d8c71f620edae8f9f2d4039376be2e29ffc59435a66ea984a55bc197d6e8d8` |

Zero-state trajectory digest (`trajectory_digest(zeros(16,16))`):
`c4ed8b688e14f2127c8e03ea62eee0ecd10cc15ed5dd695afc8a193efc9198d3`

Design A reference digests (frozen primitive, also pinned here for cross-check):

| message | Design A digest (hex) |
|---|---|
| `b""` | `d12c29be1429775e6dcc9ff3e29d9bca96865c0179a99b9bcee58581bf118820` |
| `b"abc"` | `e6231beb61a76e304a5292473a955a970b74b25f55027ca6f0cc34a1cd21985d` |
| `b"WaveLock"` | `5109e4c0d3effe338c4b1b35555aac8db35f2754753afea961cd768a04937cb2` |

---

**This baseline is immutable. Candidate B is a separate package (`CC-Core-v1-B`)
and never overwrites Candidate A.**
