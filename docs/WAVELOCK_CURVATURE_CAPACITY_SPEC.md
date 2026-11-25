# WaveLock Curvature-Capacity Core (CC-Core-v0) — Formal Specification

**Status:** EXPERIMENTAL research candidate. **No security claim.**
**Branch:** `research/curvature-capacity-wavelock` · **Date:** 2026-06-18
**Implementation:** `wavelock/curvature_capacity/`
**Audit:** `curvature_audit/`, `docs/WAVELOCK_CURVATURE_CAPACITY_RESULTS.md`

This specifies a PDE-native, **hash-free** candidate whose digest commits to the
ordered wave *trajectory* (the "wake" / curvature signature), not only to the
terminal state. It is built **on top of the frozen Design A** wave round and
does not modify it. No SHA / SHAKE / BLAKE / MD5 / AES / ChaCha / SipHash / HMAC
/ HKDF / Argon2 / library digest or XOF appears in absorption, evolution, the
accumulator, the constants, or the squeeze. A forbidden-import test enforces
this (`curvature_audit/test_forbidden_imports.py`).

The motivating design intent (from the task): security should *not* be claimed
from large state space, path count, visual complexity, chaos, avalanche, or
trajectory length. The intended (and, per the Results, **unproved**) mechanism
is curvature-resolution cost exceeding bounded solver capacity. This spec
defines the object precisely so the claim can be tested and — as it turns out —
*not* established.

---

## 1. Shared field and the frozen wave round F (Design A, unmodified)

Field F_p, p = 2³¹ − 1; lattice N × N with N = 16 (256 cells); flat index
idx = i·N + j. The **wave field** ψ ∈ F_p^{N×N} evolves by the exact Design A
round (finite-field Allen–Cahn discretization):

```
t      = ψ²                              mod p
react  = a·(ψ·((b + (p − t)) mod p))     mod p
lap    = (ψ[i+1,j]+ψ[i−1,j]+ψ[i,j+1]+ψ[i,j−1] + (p−4)·ψ) mod p
ψ'     = (ψ + (D·lap mod p) + react)     mod p
```

with D = 5, a = 3, b = 1431655765, T = 32 (identical to
`wavelock/pde_hash/spec.py`). The wave field is **autonomous**: ψ_{t+1} = F(ψ_t)
does not depend on the accumulator. This is deliberate — it makes the Design A
eigenmode collisions (`F(s·σ)=0`) apply verbatim to ψ, so the central question
("does the trajectory commitment separate Design A's terminal-state
collisions?") is crisp. The parity test
`curvature_audit/test_parity.py::test_wave_round_matches_design_a` asserts the
wave round is byte-identical to Design A.

---

## 2. The accumulator field C and the coupled round (Candidate A)

A second field C ∈ F_p^{N×N} (the **path accumulator**) is co-evolved. At global
round index t (the index increments across *all* rounds of *all* blocks and into
the squeeze, never resetting), with u = ψ_t[x], v = ψ_{t+1}[x] = F(ψ_t)[x]:

```
injection   j(x)  = ( u + γ·u·v + η·u² + ζ·v )                  mod p
self-diff   Cd    = ( C + D_C·Laplacian(C) )                    mod p
update      C'[x] = ( μ·Cd[x] + A_C·Cd[x]² + W_t(x)·j(x) + ρ_t ) mod p
```

Constants (new, domain-separated, small fixed integers — none from a hash):
`D_C = 3, γ = 11, η = 13, ζ = 17, A_C = 2, μ = 5`.

- **round constant** ρ_t = (RHO0 + RHO1·t) mod p, RHO0 = 0x57434330 ("WCC0"),
  RHO1 = 2654435761 mod p.
- **position weights** W_t(x) = (1 + (t+1)·WA + (x+1)·WB + (t+1)·(x+1)·WC) mod p,
  WA = 40503, WB = 50021, WC = 60013.

Each requirement of task Part VI and the design rationale:

| Requirement | Mechanism |
|---|---|
| order sensitivity | ρ_t and W_t depend on the global round index t; reordering rounds/blocks changes them. |
| every round influences C | C'[x] depends on C_t[x] (carry) and on (ψ_t, ψ_{t+1}); the **digest reads C**, so no round is discarded. |
| resist additive cancellation | A_C·Cd² and the `W_t(x)` multipliers are nonlinear / position-varying; constant or sign-symmetric injections do not cancel. |
| resist sign symmetry (the Design A killer) | the **odd-in-u** term `u` (and `ζ·v`) make C₁ differ for ±s·σ: when v = 0 (eigenmode → 0), j(x) = u = ±s·σ(x), whose sign survives. |
| not a passive checksum | D_C·Laplacian(C) couples neighbours; over T rounds C mixes globally. |
| order asymmetry of a transition | η·u² (earlier state) vs ζ·v (later state) is **not** symmetric under u↔v, so transition direction is bound. |

**Known weakness (documented, not hidden).** The term η·u² makes j quadratic in
u; for fixed v the per-cell injection is therefore ≤ 2-to-1 in u. Toy
enumeration (`reduced_models.py`) shows the coupled round is **not** globally
injective; residual collisions are dominated by N=2 neighbour degeneracy (the
same pathology Design A flagged) and are not extrapolated to N=16. The coupled
round's global injectivity is **UNRESOLVED**, exactly as Design A's was.

### 2.1 Other candidates (compared, not selected as primary)

- **Candidate B (irreversible trajectory injection):** C_{t+1} = F_t(C_t +
  J_t(ψ_t)) with a time-dependent J_t. Subsumed by A (A's injection is a J_t with
  the carry inside).
- **Candidate C (two-way coupling):** ψ_{t+1} = F_t(ψ_t, C_t). Available behind a
  flag; **not** the analyzed primary, because feeding C back into ψ would
  invalidate the clean "Design A eigenmodes apply to ψ" test. Noted for future
  work.
- **Candidate D (delayed state):** ψ_{t+1} = F_t(ψ_t, ψ_{t−1}, C_t). Not
  implemented in v0; recorded as future work.

Candidate A is the primary because it isolates the single new mechanism (the
path accumulator) against an unchanged, well-characterized wave field.

---

## 3. Trajectory as the cryptographic object

For a message m the construction defines the ordered trajectory

```
Γ(m) = ( (ψ_0,C_0), (ψ_1,C_1), …, (ψ_End, C_End) )
```

across all blocks and the squeeze. **The digest is a function of the C-path**,
so two messages with Γ(m₁) ≠ Γ(m₂) may still differ in digest even if their
terminal wave states coincide. This is the explicit repair of Design A's
terminal-state-only commitment.

### 3.1 Initial states (IVs), no hash

Two domain-separated IVs by direct base-256 injection (spec/state.py):

```
IV_field[i,j] = (1 + idx + tag[idx mod len(tag)]) mod p
  ψ-IV tag: b"WaveLock-CC-Core-v0:psi"
  C-IV tag: b"WaveLock-CC-Core-v0:acc"     (distinct ⇒ domain separation)
```

### 3.2 Message encoding, padding, absorption

Identical structure to Design A: 3 bytes per field element (injective), `10*`
padding plus a dedicated trailing 192-byte length block (first 8 bytes = bit
length, big-endian). Block = 64 field elements = 192 bytes.

Per block k (mutating both fields):

```
for c in 0..63:  ψ[c] += block[c];  C[c] += G·block[c]         (mod p)   # G=7
(q0,q1) = encode_block_counter(k)                                         # base-p, injective
ψ[CAP0]+=q0·G; ψ[CAP2]+=q1·G;  C[CAP0]+=q0·G; C[CAP2]+=q1·G    (mod p)
if last block: ψ[CAP1]+=D_TAG;  C[CAP1]+=D_TAG                 (mod p)    # D_TAG="WCC1"
(ψ,C) ← coupled_evolve_T(ψ,C)    # T coupled rounds, global round index advances
```

CAP0=64, CAP1=65, CAP2=66 (capacity cells, disjoint from rate cells 0..63). The
message is injected into ψ additively and into C with a fixed multiplier G so the
two fields never receive the identical perturbation.

### 3.3 Squeeze (digest extraction from the accumulator)

```
out_bits = []
while len(out_bits) < 256:
    for t in 0..63:  out_bits.append( 1 if C[t] > C[t+128] else 0 )   # disjoint pairs
    if len(out_bits) < 256:  (ψ,C) ← coupled_evolve_T(ψ,C)            # re-evolve both
pack MSB-first into 32 bytes
```

The comparison reads the **accumulator** C (not ψ); the re-evolution between
squeeze rounds advances the *coupled* system, so the squeeze is part of the
dynamics. Exactly 4 squeeze rounds, 3 intermediate coupled T-evolutions.

A key non-degeneracy property (verified in
`eigenmode_attacks.coupled_nondegeneracy`): even when ψ reaches the zero fixed
point, ρ_t and the accumulator's own dynamics keep C evolving, so the digest
does **not** collapse to all-zero bytes (unlike Design A's zero state).

---

## 4. Curvature and signature functionals (measurable; diagnostics only)

Defined on **lifted integer representatives** (convention CENTERED:
r ↦ r−p if r > p/2 else r). See `curvature_audit/curvature_metrics.py`.

| Functional | Definition | Efficient? | Crypto-relevant? |
|---|---|---|---|
| spatial curvature K2(ψ) | Σ_x (Δψ(x))² | yes | **no** (see below) |
| gradient energy G(ψ) | Σ_edges (ψ(x)−ψ(y))² | yes | no |
| temporal curvature Kτ | Σ_x (ψ_{t+1}−2ψ_t+ψ_{t−1})² | yes | no |
| trajectory separation D_Γ(m,m′) | Σ_t w_t·‖ψ_t(m)−ψ_t(m′)‖² | yes | no (saturates at avalanche) |
| spectral effective rank | exp(H(normalized 2-D power spectrum)) | yes (float) | no |
| digest byte entropy | Shannon entropy of digest bytes | yes | output statistic only |

**Documented non-invariance / counterexample.** Curvature *magnitude* is
**convention-dependent**: a structured low-magnitude state has K2 = 80 under the
centered lift and K2 ≈ 9.2 × 10¹⁹ under the naive [0,p) lift — ~18 orders of
magnitude apart (`curvature_metrics.lifting_sensitivity`). Therefore **no
cryptographic claim may rest on raw curvature magnitude.** These functionals
describe the forward trajectory; none is shown to be a hardness measure. This is
the operational meaning of the task's "do not conflate large numerical curvature
with computational hardness."

---

## 5. Domain separation summary

Provided without any hash by: (a) distinct ψ/C IV tags; (b) the per-block,
injective base-p counter injected into both fields; (c) the trailing length
block + the "WCC1" finalization tag injected into both fields; (d) the
round-index-dependent ρ_t and W_t schedules. No keyed mode is defined in v0.

---

## 6. Pinned test vectors (CC-Core-v0)

Emitted by `wavelock/curvature_capacity/optimized.py`, reproduced byte-for-byte
by `reference.py` (parity test). Any constant change is a version bump.

```
cc_hash("")            = 99e7beade48a10b0e4badf5dcecfa617e3a361b789c60e5afaee8c02ab55d6d2
cc_hash(0x00)          = c44b0b91973fe2ec9af5dae3692c9e83a14fc707f0359b2e6800d429bc63e266
cc_hash(0x01)          = 8fdc46738f2d7e7167b9af90c5b992431dbdf484a11d56c43236919973061a17
cc_hash(0xff)          = 783e26c6db282e939b65894e54afc9d51d000c3f41e0dc39f83c7b886f1cc973
cc_hash("abc")         = 385600d91057f24522d1210cb4bb7c8983339cfdcb0b466e41bde2d2c93044ef
cc_hash("WaveLock")    = afbcb55d3a54af05a370703780273e375b08269e4071920c92f3e35a3978c44e
cc_hash(0x00*192)      = faa462019807c38ccefac61c50e79a1c6e1dfc46825d510d3526871d3a523f5f
cc_hash(0x00*193)      = 34d8c71f620edae8f9f2d4039376be2e29ffc59435a66ea984a55bc197d6e8d8
```

---

## 7. Explicit non-claims

CC-Core-v0 is an experimental candidate with **no security proof**. The
trajectory-commitment mechanism is **not** proven to deliver collision /
preimage resistance. The coupled round is **not** proven injective. No
exponential lower bound on attack cost exists. See the Results document for the
adversarial audit and the precise allowed / forbidden statements.
