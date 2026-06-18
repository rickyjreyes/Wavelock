# WaveLock-PDE-256-v0 — Phase 8 Adversarial Cryptanalysis Results

**Status:** experimental. This document reports an empirical adversarial audit of
the Design A candidate. It makes **no** formal security claim.

> **Observed attack cost is not a lower bound and is not a proof of one-wayness,
> collision resistance, or cryptographic security.**

---

## 1. Construction tested

WaveLock-PDE-256-v0 (Design A): a **finite-field polynomial dynamical system
derived from the algebraic form of the Allen–Cahn reaction–diffusion equation**,
`ψ' = ψ + D·∇²ψ + a·ψ·(b − ψ²) (mod p)`, on a 16×16 toroidal lattice over
𝔽_p (p = 2³¹ − 1), iterated as a sponge-like absorb/evolve/squeeze construction
with a comparison-based 256-bit squeeze. Full specification:
`docs/PDE_HASH_SPEC.md`. The T-round transform `evolve_T` is **not** known to be
bijective.

## 2. Exact commit and parameter set

- Branch: `research/hash-free-pde-core`; reviewed base `346c4aa`. Corrections +
  Phase 8 added in subsequent commits (see `git log`).
- Parameters (v0): `p = 2³¹−1, N = 16, D = 5, a = 3, b = 1431655765, G = 7,
  T = 32`; rate = cells 0..63, capacity 64..255; cap0=64, cap1=65, cap2=66.
- All audits use the parameterized `pde_audit/_harness.PDEVariant`, verified to
  reproduce the normative `optimized.pde_hash` byte-for-byte at v0 parameters.

## 3. Methods

All analyses run on **raw PDE state or the raw 256-bit output**; no conventional
hash is ever applied to candidate output before analysis. Conventional hashes
(SHA-256, BLAKE2b) appear only as external reference columns in `baselines.py`.
Every experiment uses a fixed deterministic seed and writes a machine-readable
artifact under `pde_audit/artifacts/`. Environment: Python 3.11.15, NumPy 2.4.6,
scikit-learn 1.9.0, z3 4.16.0, Linux x86_64, 4 vCPU.

## 4. Sample sizes (headline)

| Phase | Sample |
|---|---|
| 8A avalanche | 10 T-values × 13 base msgs × 256 input bits (+2/4-bit higher order) |
| 8B state-map | 63 full-state Jacobians; 8 Brent starts (budget 2·10⁴); 2·10⁴ duplicate search; 10 exhaustive toy systems |
| 8C squeeze | 8 000 messages × 256 output bits (+ tie/round/pair-table) |
| 8D distinguishers | 7 tasks, ~3 000 samples/task, dedup + fresh-seed reproduction |
| 8E symmetry | 8 equivariance checks; 400 swap/repeat; 300 geometric; 6 000-digest corpus |
| 8F collision | pools 30k(T32)/100k(T8)/220k(T4); truncations 8..40 bit |
| 8G preimage | brute force to budget 3·10⁵; local search 2·10⁴ evals; 2nd-preimage 2·10⁵ |
| 8H algebraic | N=1 functional degree; 600-sample affine test; 6 z3 toy inversions |
| 8I parameter | 19 predeclared regimes |
| Baselines | 5 weak controls + 2 external hash refs |

## 5. Attacks attempted (all categories present)

Diffusion/avalanche by round (8A); state injectivity / Jacobian rank / fixed
points / cycles / collapse, full + exhaustive toy (8B); squeeze bias / ties /
correlation / round-dependence / pair-table sensitivity (8C); structural
distinguishers + ML classifiers vs uniform and for input properties (8D);
lattice-symmetry and structured collisions (8E); truncated collision scaling,
multicollision, chosen-prefix (8F); brute-force / hill-climb / simulated-
annealing / genetic preimage, second-preimage, state-recovery underdetermination
(8G); functional vs formal degree, affine relations, SMT reduced-round inversion
(8H); parameter sweep (8I); weak baselines + external references (Part III).

## 6–9. Findings (positive / negative / weaknesses / failed attacks)

### 8A — Avalanche / diffusion (artifact: `phase8a_avalanche.json`)
- Diffusion saturates by **T ≈ 8**. At normative **T = 32**: single-bit-flip
  output Hamming distance mean **127.9** (ideal 128), min 100, max 158; per-
  output-bit flip probability in **[0.475, 0.525]**. Two- and four-bit
  differentials likewise centered at 128.
- **No avalanche weakness.** Reduced rounds T ≤ 4 are clearly under-diffused
  (T=1 mean HD 19.9; T=2 67.1; T=4 113.3) — relevant to reduced-round attacks.

> **Correction (Phase 8J, see §8J and `PDE_HASH_STRUCTURED_COLLISION_NOTE.md`):**
> the wording below originally treated sampled full-rank Jacobians as if they
> bore on injectivity. They do **not**. The state transformation is now known to
> be **constructively non-injective**: an explicit family of ≥47 structured
> states maps to the all-zero state in one round. A full-rank Jacobian shows
> only *local* non-singularity at the sampled point. Random sampling here could
> never reach the structured family (≈46 of ~10²⁴⁰⁰ states).

### 8B — State map (artifact: `phase8b_state_map.json`)
- **Full system (N=16):** one-round modular Jacobian is **full rank (256/256)
  at every sampled state** → the one-round map is *locally non-singular* at
  every sampled point. **This does NOT establish global injectivity** — Phase 8J
  exhibits distinct states with identical full-rank Jacobians that both map to
  zero. No short cycles found (Brent, budget 2·10⁴, 8 starts); no
  duplicate next-states in 2·10⁴ random states (but see 8J: the colliding states
  are a measure-≈0 structured set that random sampling cannot hit).
- **Structural facts:** the **all-zero state is a fixed point** of `evolve_T`
  (constant states `v` with `v=0` or `v²=b` are fixed; here `b` is a
  non-residue so only `v=0`). The IV ≠ 0 and absorption perturbs every block,
  so reaching all-zero from a real message is not demonstrated, but the fixed
  point exists.
- **Exhaustive toy systems:** the one-round map is **non-injective with heavy
  collapse** (e.g. N=2,p=5: image fraction 0.13, max preimage multiplicity 81;
  N=2,p=11,T=4: max multiplicity 291, image fraction 0.19). This is driven
  largely by **N=1/N=2 neighbor degeneracy** (duplicated toroidal neighbors) and
  the inherent non-injectivity of the cubic reaction over tiny fields. **Per the
  mandate, this is NOT extrapolated to the full system**, where the Jacobian is
  full-rank; it flags collapse as a mechanism to keep testing at larger scale.

### 8C — Squeeze (artifact: `phase8c_squeeze.json`)
- Per-bit P(1) ∈ **[0.486, 0.517]**; max |monobit z| = **2.97**, **0** of 256
  bits with |z|>3 at n=8000 (within multiple-comparison expectation). **Zero
  ties** across all pairs. Max pairwise |corr| = **0.047**; round-block
  correlations small. Output Hamming weight **128.0 ± 8.0** (ideal 128 ± 8).
  Byte χ² = 245 (255 dof). Alternative pair tables show similar low bias.
- **No squeeze bias found** at this sample size.

### 8D — Distinguishers (artifact: `phase8d_distinguishers.json`)
- **Methodological correction recorded:** an initial *zero-vs-random* classifier
  scored **AUC 0.999**. This was traced to **train/test leakage** from a low-
  diversity input class (only ~300 distinct all-zero messages → identical
  digests in both splits → memorization, not generalization; zero-vs-*uniform*
  was also 0.99 and feature importances were diffuse). After **de-duplicating
  identical feature rows**, it fell to **0.516**.
- With the fix, **every task is AUC < 0.55** (PDE-vs-uniform 0.494; zero-vs-
  random 0.516; low/high-entropy 0.542; length 0.488; input-HW 0.516;
  repeated-block 0.499; shared-prefix 0.509), all with label-permutation
  controls ≈ 0.5. **No distinguisher found.**

### 8E — Symmetry (artifact: `phase8e_symmetry.json`)
- `evolve_T` is **equivariant** under all 8 toroidal symmetries (translations,
  90° rotations, reflections, transpose) — a real structural property of the
  core transform. It is **broken end-to-end** by the position-dependent IV, the
  fixed-cell counter/finalization injections, and the fixed squeeze table:
  **0** block-swap full/state collisions, **0** geometric-shift state matches,
  output HD ≈ 128, truncated collisions at birthday rate. **No usable symmetry
  collision.**

### 8F — Collision scaling (artifact: `phase8f_collision_scaling.json`)
- Truncated first-collision evaluations match the birthday law: at T=32,
  observed/expected ratios across 8..28-bit truncations are **0.75–1.29**.
  Reduced-round pools (T=4, T=8) extend the curve to larger truncations at the
  same generic rate. A **3-way multicollision** at 16 bits was found in 1816
  evals; a **chosen-prefix** 24-bit collision (T=8) in 3528 evals — both generic.
- **No structural collision weakness:** truncated collisions behave like a
  random function.

### 8H — Algebraic (artifact: `phase8h_algebraic.json`)
- **Functional vs formal degree:** for the N=1 toy the composed map's functional
  degree over 𝔽_p stays ~3 while the **formal degree grows as 3ᵀ** (e.g. 6561 at
  T=8) — formal degree does **not** imply hardness, *and* tiny systems are
  algebraically simple.
- Full N=16 one-round map has **no affine relation** (sample column rank
  513/513) → genuinely nonlinear.
- **SMT (z3) reduced-round inversion:** solves only the tiniest toys (N=2,p=5,
  T=1–2; N=2,p=7,T=1). At **N=2,p=7,T=2 and p≥11 it returns `unknown`** (no
  preimage extracted). *Limitation:* z3's nonlinear-integer reasoning is weak and
  **no Gröbner-basis tool was installed**; full N=16 SMT inversion was not
  attempted. This is a limitation, **not** a "pass."

<!-- 8G, 8I, Part III filled below after their runs complete -->

### 8G — Preimage / second-preimage (artifact: `phase8g_preimage.json`)
- **Brute-force truncated preimage at normative T=32 is ~generic:** n=8 → 120
  evals (2⁸=256), n=12 → 12 642 (2¹²=4096), n=16 → 47 381 (2¹⁶=65 536). Within
  the tested range (≤16 bits) cost tracks 2ⁿ.
- **Reduced rounds are weak:** at T=4, 12-bit preimages are found in ~3 evals
  and 16-bit in 313 (≪ 2ⁿ) — residual output bias from under-diffusion (cf. 8A).
  T=8 shows mild residual bias at small truncations (n=12 → 428). This is a
  reduced-round result; it does not affect T=32 in the tested range.
- **Local search does not beat blind sampling:** against a 32-bit target (T=8),
  best Hamming distances were hill-climb 6, simulated-annealing 6, genetic 5,
  **random baseline 5** — the truncated objective has no exploitable gradient.
- **Second-preimage** (16-bit, T=8) found in 145 760 evals (~2× generic);
  20-bit not found within 2·10⁵ budget.
- **State recovery is hugely underdetermined:** 0 of 4 000 random states match a
  fixed pattern of the 64 squeeze-comparison bits (the squeeze exposes 64 bits
  of a ~7 936-bit state).
- **Meet-in-the-middle** was not implemented: the sponge-like construction has
  no clean known-plaintext state split (the message writes only 64 rate cells
  while the hard step is inverting `evolve_T`), so a standard MITM is not
  formulable here; documented as such rather than claimed passed.

### 8I — Parameter study (artifact: `phase8i_parameter_sweep.json`)
- Predeclared 19-regime grid (one axis at a time around v0). **Every regime with
  T ≥ 8** shows avalanche HD ≈ 127–129, low bias, **Jacobian full-rank fraction
  1.00**, no short cycles, and 16-bit collision ratios scattered around 1
  (generic). **T = 4 is under-diffused** (HD 113.9, higher bias, collision ratio
  0.26). Behaviour is robust across the tested `D ∈ {1,2,3,5,7,11}`,
  `a ∈ {1,2,3,5,7}`, `b ∈ {1, p/4, p/3, p/2, 2p/3}`.
- **No regime collapsed or cycled.** v0 sits in a stable region with margin
  above the T≈8 diffusion threshold. (Per the mandate, v0 is unchanged; any
  future tuning would be a new version with new vectors.)

### Part III — Baselines (artifact: `partIII_baselines.json`)
| function | avalanche HD | max bit bias | hashes/s |
|---|---|---|---|
| truncation | 0.8 | 0.090 | 3.9M |
| xor_fold | 1.0 | 0.097 | 59k |
| linear_ca | 2.0 | 0.057 | 949 |
| mod_linear | 15.5 | **0.500** | 3.1k |
| pde_T1 (reduced) | 20.0 | 0.494 | 2.3k |
| **pde_T32 (candidate)** | **128.0 ± 8.0** | **0.029** | **97** |
| sha256 (ext ref) | 128.1 ± 8.0 | 0.024 | 1.57M |
| blake2b (ext ref) | 127.1 ± 8.0 | 0.035 | 1.39M |

The weak controls all fail avalanche/bias badly. The candidate matches SHA-256/
BLAKE2b on these *statistical surrogates* — **which is NOT evidence of equivalent
security** — but is **~16 000× slower** than SHA-256 in this unoptimized NumPy
implementation.

### 8J — Structured Laplacian-eigenmode collisions (artifact: `phase8j_eigenmode_collisions.json`; note: `PDE_HASH_STRUCTURED_COLLISION_NOTE.md`)
- **A constructive full-parameter collision exists in the internal state
  transformation.** Sign fields that are toroidal-Laplacian eigenvectors,
  `ψ = s·σ` with `s² ≡ b − (2rD−1)/a (mod p)`, satisfy `F(s·σ)=0` exactly. For
  the normative constants, **r∈{1,2,4}** give quadratic-residue amplitudes
  (r∈{0,3} do not). Verified exactly (element-by-element, all implementations,
  both ±s): **checkerboard (r=4)**, **row/column stripes (r=2)**, **period-4
  columns (r=1)** all map to the all-zero state after one round and after
  `evolve_T`.
- **Constructive lower bound:** ≥ **46 distinct nonzero states** (r=1: 8, r=2:
  36, r=4: 2; 6 symmetry orbits) plus the zero state → **≥ 47 explicit one-round
  preimages of zero**. The full ±1-eigenvector family over 𝔽_p is conjectured
  larger but **not claimed exponential** without a constructive argument.
- **Jacobian:** full rank (256/256) at *every* collision state, including the
  two distinct checkerboard states (±s) which share an identical Jacobian and
  both map to zero — **a direct refutation of "full-rank ⇒ injective"**.
- **The state transformation is therefore NOT globally injective** and is not
  globally one-way at these structured points. It must not be called
  collision-resistant or globally one-way.
- **Message-level reachability: not lifted.** Absorption controls only 64/256
  cells; the **189 capacity cells `67..255` are never injected** and hold fixed
  IV values (`< ~300`) that cannot equal `±s` (`~10⁸`). So no structured
  collision state is reachable as the pre-first-round state for any message
  (exact structural reason), and a 50 000-message search found no message whose
  pre-squeeze state is zero or structured. **No end-to-end message collision or
  message preimage of zero was demonstrated.**

## 10. Limitations

- **Scale of testing.** Collisions were measured only on **truncated** outputs
  (≤28 bits at T=32); preimages only to ≤16-bit truncation at T=32. Full
  128-/256-bit behaviour is **extrapolated, not observed**.
- **Algebraic tooling.** Only z3 (weak at nonlinear integer arithmetic) was
  available; **no Gröbner-basis / dedicated F_p solver**. Algebraic inversion
  was demonstrated to fail beyond the tiniest toys, but a stronger solver could
  reach further; full N=16 algebraic inversion was not attempted.
- **Toy degeneracy.** Toy collapse evidence is dominated by N=1/N=2 neighbor
  degeneracy and is **not extrapolated** to N=16.
- **Local/heuristic search** explored a tiny fraction of the message space.
- **Statistical power.** Distinguisher/squeeze tests used thousands of samples;
  rare biases below that resolution would be missed.
- **Performance.** The reference/optimized implementations are unoptimized
  Python/NumPy; throughput (97 h/s) is far below conventional hashes.
- **No proof.** Nothing here bounds any attack from below.

## 11. Runtime and hardware

Linux x86_64, 4 vCPU, Python 3.11.15 / NumPy 2.4.6 / scikit-learn 1.9.0 /
z3 4.16.0. Approx. wall-clock: 8A 190 s, 8B 15 s, 8C 84 s, 8D 194 s, 8E 93 s,
8F 722 s, 8G 2 311 s, 8H 25 s, 8I 264 s, baselines 48 s. Cross-impl 10k parity
(post-corrections) 610 s.

## 12. Reproducibility commands

```bash
python -m pde_audit.avalanche          # 8A   (seed 80010)
python -m pde_audit.state_map          # 8B   (seed 80020)
python -m pde_audit.squeeze_analysis   # 8C   (seed 80030)
python -m pde_audit.distinguishers     # 8D   (seed 80040)
python -m pde_audit.symmetry_attacks   # 8E   (seed 80050)
python -m pde_audit.collision_scaling  # 8F   (seed 80060)
python -m pde_audit.preimage_attacks   # 8G   (seed 80070)
python -m pde_audit.algebraic_analysis # 8H   (seed 80080)
python -m pde_audit.parameter_sweep    # 8I   (seed 80100)
python -m pde_audit.baselines          # III  (seed 80090)
python -m pde_audit.run_phase8         # all + artifacts/INDEX.json
python -m pytest pde_audit/ -c pde_audit/pytest.ini -m "not slow"
```

## 13. Raw artifact index

`pde_audit/artifacts/`: `phase8a_avalanche.json`, `phase8b_state_map.json`,
`phase8c_squeeze.json`, `phase8d_distinguishers.json`, `phase8e_symmetry.json`,
`phase8f_collision_scaling.json`, `phase8g_preimage.json`,
`phase8h_algebraic.json`, `phase8i_parameter_sweep.json`,
`phase8j_eigenmode_collisions.json`, `partIII_baselines.json`, `INDEX.json`.
Deterministic vectors: `pde_audit/vectors.json`.

## 14. Verdict

### Message digest: Unresolved experimental candidate
### State transformation: constructively non-injective (NOT globally injective / one-way)

These are **two layers** and the rubric is applied to each.

**Internal state transformation — corrected finding (Phase 8J).** A constructive
full-parameter collision was found: an explicit family of ≥47 structured
Laplacian-eigenstate states maps exactly to the all-zero state after one round,
and therefore after `evolve_T`. The state transformation is **not globally
injective** and must not be called collision-resistant or globally one-way. (The
earlier statement that "no structural attack succeeded" applied to random/
statistical testing of the digest; it did **not** cover this mathematically
targeted family, which random sampling could never reach. Sampled full-rank
Jacobians showed only local non-singularity, not global injectivity.)

**Complete message digest — Unresolved experimental candidate.** No end-to-end
collision between distinct valid messages, and no message-level preimage of the
zero state, has been demonstrated. The structured internal collisions are **not
reachable from message absorption**: only 64 of 256 cells are message-
controlled, and the 189 capacity cells `67..255` are pinned to fixed IV values
that no eigenmode state can match (exact structural reason; a 50 000-message
search also found nothing). Combined with the earlier results — full avalanche
(HD 127.9), unbiased squeeze, no distinguisher (after correcting a train/test
leakage artifact), generic truncated collision/preimage cost, no affine
relations — and with the standing caveats (shallow test scale, no Gröbner tool,
reduced-round bias at T≤8, ~16 000× slower than SHA-256, **no proof**), the
digest sits at **Unresolved experimental candidate**, now carrying the explicit
companion finding: *the internal state transformation is constructively
non-injective with simple algebraic preimages of zero.*

> **Observed attack cost is not a lower bound and is not a proof of one-wayness,
> collision resistance, or cryptographic security.**

## 15. On Design B

Design A reached an honest *unresolved* state with no break, so proceeding to
**Design B (the exact fixed-point translation of the original curvature-feedback
dynamics) is warranted — but with two revisions informed by Phase 8**, not
unchanged: (1) adopt the **exact-integer / finite-field discretization
discipline** that made Design A bit-reproducible and algebraically analyzable,
rather than float chaos; (2) **instrument the same audit suite from the start**,
with special attention to the contraction / fixed-point and singularity risks
that the curvature feedback (division by ψ, `log ψ²`) is expected to face — the
all-zero fixed point and toy-collapse mechanisms seen here make state-collapse
the first thing to measure. Design B is **not** started in this task.

