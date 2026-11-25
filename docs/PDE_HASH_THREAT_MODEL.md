# WaveLock-PDE-256 — Threat Model

**Status:** experimental. Defines what "secure" would *mean* for the candidate,
what the attacker may do, and the concrete falsification targets the
`pde_audit/` suite must attempt. This document deliberately assumes the
construction is **insecure until evidence says otherwise**.

**Branch:** `research/hash-free-pde-core`. **Date:** 2026-06-17.

---

## 1. What is being tested

`H_PDE : {0,1}* → {0,1}^256`, `H_PDE(m) = Q(Φ_P^T(A(m)))`, instantiated as
WaveLock-PDE-256-v0 (`docs/PDE_HASH_SPEC.md`, Design A) and, separately, as the
exact fixed-point curvature-feedback map (Design B). The question is narrow:

> Does the PDE evolution *itself* — with no SHA/SHAKE/BLAKE anywhere in
> absorption, initialization, evolution, round constants, or squeeze — provide
> the one-way / collision-resistant behavior expected of a cryptographic hash?

Conventional hashes appear in this work **only** as external statistical
reference points (Phase 9) and for repository/test bookkeeping — never inside
the primitive or inside any test that runs on the primitive's output.

## 2. Security goals (the properties an attacker tries to violate)

| Goal | Definition | Generic bound for 256-bit output |
|---|---|---|
| **Preimage resistance** | Given `y`, find `m` with `H_PDE(m)=y` | ~2²⁵⁶ work |
| **Second-preimage** | Given `m`, find `m'≠m` with equal digest | ~2²⁵⁶ |
| **Collision resistance** | Find any `m≠m'` with equal digest | ~2¹²⁸ (birthday) |
| **Indistinguishability** | Outputs look uniform over `{0,1}^256` | no PPT distinguisher with non-negligible advantage |
| **Determinism / portability** | Same input → identical bytes everywhere | exact (a hard requirement, not a security goal) |

A candidate that cannot beat a *generic* attack in some regime is **not viable**
for that regime. Beating generic attacks empirically is **necessary but not
sufficient**; it is never a proof.

## 3. Attacker model

The attacker is computationally bounded but otherwise unrestricted and has:

- **Full white-box knowledge** of the spec, all constants `(N,p,T,D,a,b,g,d)`,
  the IV, the absorption schedule, and the squeeze map. There is no secret key;
  this is an unkeyed hash. Security must not rely on obscurity of any constant.
- The ability to query `H_PDE` and all intermediate stages (`absorb`, `evolve`,
  `squeeze`) on chosen inputs, and to run reduced-round variants `Φ_P^t` for any
  `t ≤ T`.
- Access to the exact reference and optimized implementations.

The attacker wins by exhibiting any of the breaks in §4 at sub-generic cost, or
any structural leak in §5.

## 4. Attack surface — what `pde_audit/` must attempt (Phase 8)

### 4.1 Differential / avalanche
- Single-bit input flips → output Hamming-distance **distribution** (mean, min,
  max, outliers — never the mean alone), per-output-bit flip frequency,
  diffusion vs. round count, worst-case input bit and input family, and
  higher-order (multi-bit) differentials.
- **Falsifies** the candidate if avalanche is far from 128/256 bits, if some
  output bit is stuck/biased, or if a low-weight input differential yields a
  low-weight, predictable output differential (a differential trail).

### 4.2 Collisions
- Exact-output collisions; reduced-round collisions; truncated-output
  collisions; differential trails; symmetry-generated collisions (lattice
  translations/reflections); padding-equivalent messages; boundary-equivalent
  states; multicollisions; fixed points and short cycles of `Φ`.
- **Falsifies** if any collision class is found below birthday cost, or if the
  field dynamics collapse many messages into a small set of terminal states.

### 4.3 Preimage / state recovery
- Gradient-free inversion; SAT/SMT modeling of reduced-round `Φ` over 𝔽_p;
  mixed-integer / algebraic (Gröbner-style) formulations exploiting that `Φ` is
  polynomial of low degree over 𝔽_p; genetic search; simulated annealing;
  meet-in-the-middle on the sponge-like construction; local-linear / Jacobian
  approximations;
  surrogate models; full state reconstruction from output; chosen-prefix
  attacks; recovery of internal state from the squeeze.
- **Special concern for Design A:** each round is a degree-3 polynomial map over
  𝔽_p. The composition `Φ_P^T` has degree growing as `3^T`, but algebraic
  attacks (Gröbner, linearization, interpolation) may still bite at low `T`.
  The audit must locate the round count at which algebraic inversion becomes
  intractable and report it as an *observed* threshold, not a bound.

### 4.4 Structural distinguishers (Phase 8)
- Does the output reveal message length, Hamming weight, repeated blocks, byte
  frequency, spatial translations, rotations/reflections, affine relations,
  low-frequency modes, or input prefixes/suffixes?
- Train simple classifiers (logistic regression, random forest) to separate
  PDE outputs from uniform 256-bit strings, with disjoint train/test sets and
  reported confidence intervals.
- **Falsifies** if any classifier achieves significant advantage, or any input
  feature is recoverable from the output.

### 4.5 Algebraic / dynamical analysis
- Lyapunov-style sensitivity, attractor count, basin structure, cycle lengths,
  effective rank / dimension of the round map, local Jacobian spectra over 𝔽_p
  where meaningful, contraction vs. expansion, sensitivity to the squeeze cell
  selection, and whether distinct messages collapse into shared terminal states.
- **Discipline:** chaotic sensitivity (Design B) or large cycle counts are
  **not** evidence of one-wayness on their own and will not be reported as such.

## 5. Structural leaks specific to this construction

| Risk | Mechanism | Test |
|---|---|---|
| Translation collisions | toroidal lattice is shift-symmetric; a shifted absorption pattern could map to a shifted state with equal squeeze | symmetry-collision search |
| Zero/repeat cancellation | additive absorption could let a repeated block cancel | counter-injection (§3.5 spec) + collision search |
| Squeeze bias | comparison bits may be unbalanced if state values cluster | monobit / correlation on output bits |
| Low `T` linearity | few rounds → near-affine map → algebraic inversion | reduced-round SAT/Gröbner |
| Contraction (Design B) | curvature feedback + damping may collapse to fixed points → trivial collisions and information loss masquerading as one-wayness | cycle/fixed-point + collision analysis |

## 6. Baselines (Phase 9)

The candidate is compared against deliberately weak non-cryptographic baselines
(direct truncation, XOR-folding, linear cellular automata, simple integer
mixing, a reduced-round toy sponge) and, **as external reference only**, against
conventional hashes. Passing a few avalanche/randomness tests at parity with
SHA-256 will **not** be reported as equivalence to SHA-256.

## 7. Out of scope for v0

- Keyed modes / MAC / PRF security.
- Sub-byte input granularity.
- Side-channel / timing resistance of the reference code.
- Connection to CurvaChain or any downstream system (forbidden until the
  standalone primitive is tested).

## 8. Verdict rubric (Phase 10)

The final `docs/PDE_HASH_RESULTS.md` will assign exactly one verdict per design:

- **Broken** — a practical structural attack succeeds (§4 at sub-generic cost).
- **Not viable in current form** — determinism, bias, collisions, contraction,
  or performance prevent serious use.
- **Unresolved experimental candidate** — no practical break found, but no proof
  and insufficient evidence.
- **Promising research candidate** — broad adversarial testing supports it,
  still with no production-security claim.

`observed attack cost ≠ lower bound ≠ proof of one-wayness` is the governing
disclaimer for every positive statement.
