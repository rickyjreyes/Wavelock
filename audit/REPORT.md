# Hostile Cryptographic Audit — WaveLock

**Scope:** WaveLock's standalone one-way / commitment behavior only.
`C = SHA256(Serialize(ψ*))`, where `ψ*` is a terminal field produced by evolving
a seed-derived field `ψ0` under WaveLock's nonlinear curvature/PDE operator.
Out of scope: CurvaChain, OTS signatures, replay/ledger, Merkle.

**Target code (consensus reference):**
`wavelock/chain/Wavelock_numpy.py` (evolution `_evolve` L189-200, serialization
`_serialize_commitment` L106-124), seed derivation `wavelock/chain/xof_init.py`.
Parameters (hardcoded): `alpha=1.5, beta=0.0026, theta=1e-5, epsilon=delta=1e-12,
dt=0.1, steps=50, damping=2e-5`; lattice `side=2**max(1,n//2)` (CLI/demos `n=4` → 4×4).

**Hardware/software:** Intel Xeon @2.80GHz, 4 vCPU, 15 GiB RAM; Linux x86-64;
Python 3.11.15; NumPy 2.4.6; OpenBLAS 0.3.31. Single-core, NumPy-only, no GPU.

**Harness fidelity:** `audit/_wl.py` reproduces the project's published
commitment **byte-for-byte** (`n=4, seed=42 → C=5287fcb0…6c38`) and our
standalone `evolve()` equals the repo `_evolve()`. All evidence below binds to
real WaveLock output. Reproduce everything with `bash audit/run_all.sh`.

---

## TL;DR / Final verdict

> **WaveLock does NOT behave like a high-entropy one-way commitment function on
> its own merits. Every preimage/collision/indistinguishability property it
> appears to have is inherited entirely from SHA-256. The nonlinear PDE
> contributes no cryptographic hardness and actively *harms* the construction:
> its chaotic floating-point sensitivity destroys cross-platform reproducibility
> (same input → different commitment across conforming implementations), and its
> cheap-to-evaluate map over low-entropy integer seeds makes brute-force seed
> recovery trivial.**

The single most important methodological point: **`C = SHA256(...)` will pass
every avalanche / monobit / NIST-style test no matter how degenerate `ψ*` is.**
Any "WaveLock looks random" result measured on `C` is non-evidence — it measures
SHA-256. Honest tests must run on `ψ*` and `Serialize(ψ*)`, and there WaveLock
fails badly (Finding H-2).

---

## Findings summary

| ID | Class | Finding | Severity |
|----|-------|---------|----------|
| **C-1** | Floating-point determinism (#6) | Mathematically-equivalent float reassociation (the difference between BLAS/SIMD/compiler/NumPy-version/GPU builds) changes `ψ*` and therefore the commitment for the **same seed**. Vendor already guards GPU as "non-consensus". | **Critical** |
| **C-2** | Seed-space weakness (#7) | Standalone WaveLock commits to a caller integer seed; the shipped CLI default is `42`. Given only `C`, the seed is recovered by brute force: default in **43 tries**, a 20-bit secret in **146 s**, full 32-bit in ~10 days single-core (hours parallel/GPU). Legacy path is capped at 32-bit. | **Critical** (low-entropy deployment) |
| **H-1** | Serialization ambiguity (#5) | `Serialize(ψ*)` is non-canonical: `+0.0` vs `-0.0`, distinct NaN payloads, and the sanctioned LE (WLv2) vs BE (WLv3) bodies all map one logical field to multiple commitments. | **High** |
| **H-2** | Distinguishability (#4) | `Serialize(ψ*)` byte χ² = **5.5×10⁶** (crit 310.5); `ψ*` skew 195, excess kurtosis 5.7×10⁴, neighbor corr 0.57. Trivially distinguishable from random — only SHA-256 hides it. | **High** |
| N-1 | 2nd-preimage / collision (#1) | **No** collision found in 1,000,000 seeds; no fixed-point collapse; cost is SHA-bound (~2¹²⁸). | Negative (resistance is SHA-256's) |
| N-2 | Attractor collapse / entropy (#3) | **No** collapse: 1e6/1e6 commitments distinct, all fields finite. Entropy preserved (≥20 bits over the sweep). | Negative |
| N-3 | Jacobian / inversion (#8) | Map is chaotic (Lyapunov up to 0.80/step) **and** ill-conditioned (cond up to 4.5×10¹¹); Newton inversion of `ψ*→ψ0` fails. | Negative (no shortcut) |
| N-4 | Neural/surrogate (#9) | `ψ*→ψ0` surrogate R²≈0.03, does not beat the predict-0.5 baseline; `C→ψ0` is a SHA preimage (not attempted). | Negative |
| N-5 | Parameter regimes (#10) | Default sits in a chaotic/expansive regime; no tested setting yields collapse or NaN. Defaults are "safe" only in the trivial sense that the danger is reproducibility, not collapse. | Informational |

Net: the PDE adds **two exploitable weaknesses** (C-1, C-2) and **two
quality/encoding defects** (H-1, H-2) while adding **zero** one-wayness,
collision, or preimage hardness beyond SHA-256.

---

## C-1 — Cross-platform nondeterminism (Critical)

**Claim broken:** "same input must produce identical bytes and identical
commitment on supported platforms."

**Evidence (`audit/a6_determinism.py`, `audit/a2_jacobian.py`):**
- The kernel is a positive-Lyapunov, expansive map. Measured per-step Lyapunov
  exponents 0.04–0.80 nats/step; over 50 steps the amplification factor
  `e^(L·50)` reaches **1.84×10¹⁷** (seed 1), with `σ_max(J)` up to **3.7×10¹⁰**.
  A last-ULP input difference (~1e-16) is therefore amplified to O(1–10) in `ψ*`.
- **T4 reassociation:** running the *identical* kernel with a mathematically
  equivalent but differently-ordered floating-point expression (the exact class
  of difference introduced by a different BLAS/SIMD width/compiler/NumPy
  version/GPU) changed `ψ*` by `3.55×10⁻¹⁵` → **different commitment** for the
  same seed (`a6_determinism.json: T4_reassociation.commitment_changed=true`).
- **T3:** a `1e-12` perturbation to one cell already flips `C`.
- **Vendor admission:** `wavelock/chain/WaveLock.py` raises
  *"GPU backend is non-consensus … Only NumPy reference backend may emit
  consensus commitments."* — i.e. the authors already know the kernel is not
  reproducible across backends and have restricted "consensus" to a single
  implementation.

**Caveat (honesty):** on this single host, in-process repeats and 1/2/4/8-thread
runs produced identical commitments (`T1`, `T2`) — expected, because the kernel
uses no BLAS (only elementwise ops + `np.roll`/`np.gradient`/`np.sum`). The break
is demonstrated by reassociation emulation and corroborated by the vendor's GPU
guard, rather than by two physically distinct CPUs/GPUs (only one host was
available). The chaos math (a2) shows any real cross-build float divergence is
amplified more than enough to flip `C`.

**Repro:** `python audit/a6_determinism.py` ; `python audit/a2_jacobian.py`.
**Complexity vs brute force:** n/a (correctness/availability failure).
**Fix:** Do not hash raw float64 of a chaotic iterate. Either (a) drop the PDE
and commit directly to the seed/material with SHA-256/SHAKE; or (b) quantize
`ψ*` to a coarse, reproducibility-safe integer grid with a documented rounding
mode *and* prove the quantization is stable under the map's amplification (it is
not, at these parameters); or (c) commit to `ψ0`/seed bytes, never to `ψ*`.

---

## C-2 — Seed-space brute-force input recovery (Critical for low-entropy seeds)

**Claim broken:** "given `C`, infeasible to recover the original input."

**Evidence (`audit/a4_seedspace.py`, `a4_seedspace.json`):**
- Published `C=5287fcb0…6c38` (the CLI/demos default). Brute force over
  `0..100000` recovered **seed = 42 after 43 candidates**.
- A "random-looking" 20-bit secret seed `703710` was recovered from its `C`
  (`68bbdbf1…47fb`) in **145.95 s** at **4821 seeds/s** single-core, pure NumPy.
- Extrapolation at the measured rate: exhaust 2³² in **~10.3 days single-core**,
  i.e. hours across a handful of cores/a GPU (the 4×4 kernel is microseconds and
  embarrassingly parallel). The legacy `np.random.seed` path is **capped at
  32-bit**, so it is fully within reach.

**Why it matters:** `WaveLock(seed)` is public and cheap; if a deployment commits
to a low-entropy integer seed (and the shipped CLI defaults to `42`), the
commitment leaks the seed. This is recovery of the *actual input*, not a
birthday/preimage attack.

**Mitigation present elsewhere:** the OTS path
(`wavelock/crypto/wavelock_ots.py`) seeds with `os.urandom` (256-bit), which is
not brute-forceable. The weakness is the **standalone WaveLock API accepting
low-entropy integer seeds** plus the CLI default.

**Repro:** `python audit/a4_seedspace.py`.
**Complexity vs brute force:** this *is* brute force, but the effective keyspace
is the (tiny) seed space, not 2²⁵⁶ — for default/32-bit seeds that is seconds to
days, not infeasible.
**Fix:** Require ≥128-bit (preferably 256-bit) seed entropy in the standalone
API; reject short integer seeds; remove the `seed=42` CLI default; document that
the commitment's security ceiling is the seed entropy, not SHA-256's 256 bits.

---

## H-1 — Serialization non-canonicality (High)

**Claim attacked:** stable, canonical `Serialize(ψ*)`.

**Evidence (`audit/a5_serialization.py`, `a5_serialization.json`):**
- **S1 signed zero:** a field with `-0.0` in one cell vs `+0.0` is numerically
  identical (`-0.0 == 0.0`) yet serializes to different bytes → different
  commitments (`ed40c1…` vs `1733e7…`).
- **S2 NaN payloads:** two distinct NaN bit-patterns are both "NaN" but commit
  differently; a NaN cell also makes `verify()` self-inconsistent (`NaN != NaN`).
  (The kernel emits no NaN for finite seeds — a1 — so this is conditional.)
- **S3 dual schema / endianness:** the *same* `ψ*` has two project-sanctioned
  serializations — WLv2 little-endian `tobytes` vs WLv3 big-endian `>f8` — giving
  two different commitments. Commitment identity depends on the schema label,
  not just on `ψ*`.
- **S4 energy block:** the packed `struct.pack("<4d", …)` energies are a pure
  function of `ψ*` (zero added second-preimage strength) but add four more
  float64 channels that must match bit-exactly across implementations —
  amplifying C-1.

**Repro:** `python audit/a5_serialization.py`.
**Fix:** Canonicalize before hashing: normalize `-0.0→+0.0`, reject/forbid
NaN/Inf, pin one endianness and one schema in the hashed bytes, and drop the
redundant derived-energy block from the commitment input.

---

## H-2 — Distinguishability of ψ* / Serialize(ψ*) (High)

**Claim broken (for the pre-hash object):** outputs indistinguishable from random.

**Evidence (`audit/a7_distinguishability.py`, `a7_distinguishability.json`):**
- **Control on `C` (SHA output):** monobit ones-fraction 0.50031, byte
  χ²(df255)=258.8 < 310.5 critical, 1-bit-seed avalanche 0.494 ≈ 0.5. `C` looks
  perfectly random — **this only demonstrates SHA-256**.
- **Real test on `Serialize(ψ*)`:** byte χ²(df255) = **5,506,757** (crit 310.5) —
  off the charts non-uniform (float64 sign/exponent bytes are highly biased).
- **`ψ*` value distribution:** mean 17.4, std 834, **skew 195**, **excess
  kurtosis 57,036**, ~some negative — heavy-tailed and structured, nothing like
  the input `U[0,1)` or any clean reference.
- **Spatial structure:** nearest-neighbor Pearson correlation **0.57** (the
  smoothing stencil leaves strong local correlation).

**Interpretation:** WaveLock's nonlinearity produces a *highly distinguishable*
field. The commitment is indistinguishable from random solely because SHA-256 is
applied last. If any protocol ever exposed `ψ*`, used a truncated/linear
finalizer, or relied on the PDE for entropy, it would be trivially broken.

**Repro:** `python audit/a7_distinguishability.py 20000`.
**Fix:** Recognize that the PDE provides no diffusion/confusion of cryptographic
value; rely on SHA-256/SHAKE alone over canonical seed bytes.

---

## Negative results (reported honestly)

These attacks did **not** break WaveLock — but only because SHA-256 carries the
construction, not because the PDE adds strength.

- **N-1 second preimage / collision (`a3`, `a1`):** 0 exact `ψ*` collisions and
  0 commitment collisions across **1,000,000** seeds; `ψ*(A)` and `ψ*(B)` do not
  converge under up to 5000 extra steps (L2 distance stays ≈ 5, never → 0), so
  there is no seed-independent attractor to mass-produce collisions. Generic
  collision cost remains ~2¹²⁸ (SHA-256).
- **N-2 attractor collapse / entropy (`a1`):** all 1e6 seeds finite (0 NaN/Inf;
  15 with |ψ*|>1e6), 1e6/1e6 distinct commitments, 232,928 distinct
  integer-rounded fields (mild near-duplication, not collision). Output entropy
  is preserved (≥ log₂(1e6) ≈ 20 bits over the sweep). *Caveat:* the per-cell
  histogram entropy (~2 bits/cell) is a coarse lower bound (bin width ≈ 4.9 over
  a heavy-tailed range) and should not be over-read.
- **N-3 Jacobian / inversion (`a2`):** the 16×16 Jacobian is simultaneously
  expansive (`σ_max` up to 3.7×10¹⁰) and near-singular in directions
  (`σ_min` down to 1.9×10⁻¹⁰), condition numbers up to 4.5×10¹¹; Newton/LM
  inversion of `ψ*→ψ0` does not converge (residual ~2011, far from the true
  `ψ0`). No linear inversion shortcut.
- **N-4 neural/surrogate (`a8`):** ridge/MLP for `ψ*→ψ0` reach R² ≈ 0.0001/0.031
  and barely match the trivial predict-0.5 baseline (MAE 0.2498 vs 0.2499);
  forward `ψ0→ψ*` is also unlearnable (R² ≈ 0.006). `C→ψ0` is a SHA preimage and
  is out of scope.
- **N-5 parameters (`a9`):** the default `(steps=50, dt=0.1, alpha=1.5,
  damping=2e-5, n=4)` is chaotic/expansive with no collapse or NaN; raising
  `dt`/`alpha` increases chaos, and even `damping=0.5` does not force collapse.

---

## Answer to the assignment's final question

*"Does WaveLock behave like a high-entropy one-way commitment function, or does
nonlinear PDE contraction create exploitable collapse, distinguishability,
nondeterminism, or inversion shortcuts?"*

**Neither "contraction-collapse" nor an "inversion shortcut from C" was found —
but WaveLock still fails as a self-contained one-way commitment.** Concretely:

1. **One-wayness given only `C` is real but is 100% SHA-256.** Remove SHA-256 and
   the PDE output is trivially distinguishable (H-2) and structurally analyzable.
   The PDE adds no preimage/collision hardness (N-1…N-4).
2. **The PDE introduces nonlinear *expansion/chaos*, not contraction**, which
   breaks the determinism the commitment scheme depends on (**C-1**, Critical)
   and forces the authors to ban GPUs from "consensus".
3. **The seed space — not SHA-256 — is the real security parameter**, and for the
   shipped integer-seed / `seed=42` usage it is brute-forceable (**C-2**,
   Critical).
4. The serialization is non-canonical (**H-1**), compounding C-1.

**Recommendation:** treat WaveLock's PDE as cryptographically inert (at best a
costly, distinguishable, non-portable pre-processor). Commit to high-entropy
seed material directly with SHA-256/SHAKE over a canonical encoding; if the PDE
must remain for product reasons, (a) hash `ψ0`/seed rather than the chaotic
`ψ*`, (b) enforce ≥128-bit seeds and remove the `seed=42` default, and (c)
canonicalize bytes (sign-zero, NaN/Inf, endianness, single schema) before
hashing.

---

## Reproduction index

| Script | Attack class | Artifact |
|--------|--------------|----------|
| `audit/_wl.py` | harness fidelity | (stdout) |
| `audit/a1_attractor_entropy.py [N]` | 3 (+1,8) | `artifacts/a1_entropy.json` |
| `audit/a2_jacobian.py` | 8 (+2) | `artifacts/a2_jacobian.json` |
| `audit/a3_second_preimage.py` | 1 | `artifacts/a3_second_preimage.json` |
| `audit/a4_seedspace.py` | 7 (+2) | `artifacts/a4_seedspace.json` |
| `audit/a5_serialization.py` | 5 | `artifacts/a5_serialization.json` |
| `audit/a6_determinism.py` | 6 | `artifacts/a6_determinism.json` |
| `audit/a7_distinguishability.py [M]` | 4 | `artifacts/a7_distinguishability.json` |
| `audit/a8_surrogate.py [NTRAIN]` | 9 | `artifacts/a8_surrogate.json` |
| `audit/a9_parameters.py` | 10 | `artifacts/a9_parameters.json` |
| `audit/run_all.sh` | all | all of the above |

All results in this report were produced on the hardware/software listed at the
top; sweep sizes: a1 N=1,000,000, a7 M=20,000, a8 NTRAIN=40,000, a4 up to 2²⁰.
