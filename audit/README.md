# WaveLock Audit — Reading Guide & Scope Clarification

This folder contains a **hostile, adversarial cryptographic audit** of WaveLock
and the documentation that interprets it. The raw audit lives in
[`REPORT.md`](./REPORT.md) and is **preserved verbatim** — nothing in this guide
edits, softens, or hides it. The purpose of this README is to explain *what the
audit means* and to convert its findings into a clean research / bounty boundary.

> **Bottom line up front:** The audit clarified the boundary. WaveLock should
> currently be tested as **machine-to-machine (M2M) commitment / attestation /
> replay / drift-detection infrastructure**, not as direct payload encryption or
> a formal SHA-256 replacement.

Companion documents:

- [`BOUNTY_SCOPE.md`](./BOUNTY_SCOPE.md) — what is in and out of bounty scope.
- [`HARDENING_RECOMMENDATIONS.md`](./HARDENING_RECOMMENDATIONS.md) — recommended
  future changes (outside this folder, **not** implemented here).
- [`PASS_FAIL.md`](./PASS_FAIL.md) — compact pass/fail/constraint matrix.

---

## What the audit tested

The audit targeted WaveLock's **standalone one-way / commitment behavior only**:

```
C = SHA256(Serialize(ψ*))
```

where `ψ*` is the terminal field produced by evolving a seed-derived field `ψ0`
under WaveLock's nonlinear curvature/PDE operator. Concretely:

- **Consensus reference backend:** `wavelock/chain/Wavelock_numpy.py`
  (evolution and `_serialize_commitment`), seed derivation in
  `wavelock/chain/xof_init.py`.
- **Parameters (hardcoded):** `alpha=1.5, beta=0.0026, theta=1e-5,
  epsilon=delta=1e-12, dt=0.1, steps=50, damping=2e-5`; lattice
  `side=2**max(1, n//2)` (CLI/demos use `n=4` → a 4×4 lattice).
- **Environment:** single host, Python 3.11, NumPy 2.4, OpenBLAS, single-core,
  NumPy-only, no GPU.
- **Harness fidelity:** `audit/_wl.py` reproduces the project's published
  commitment **byte-for-byte** (`n=4, seed=42 → C=5287fcb0…6c38`). All evidence
  binds to real WaveLock output. Reproduce with `bash audit/run_all.sh`.

## What the audit did NOT test

These were explicitly **out of scope** and carry **no** verdict from this audit:

- CurvaChain, OTS signatures, the replay/ledger layer, and Merkle structures
  (except where they would later bind into WaveLock replay verification).
- **Direct payload encryption** — WaveLock was never evaluated as a cipher.
- **Cross-host / multi-backend reproducibility on physically distinct hardware.**
  Only one host was available; C-1 is demonstrated by float-reassociation
  emulation plus the vendor's own GPU "non-consensus" guard, not by two physical
  CPUs/GPUs.
- A *formal* claim that WaveLock equals or beats SHA-256/SHA3. The audit measures
  the present construction, not a long-term cryptographic ambition.

---

## How to interpret the results

The single most important methodological point from the audit:

> `C = SHA256(...)` will pass every avalanche / monobit / NIST-style test no
> matter how degenerate `ψ*` is.

So any "WaveLock looks random" result measured on the **commitment `C`** is
**non-evidence about the PDE** — it measures SHA-256. Honest tests run on the
**pre-hash** objects `ψ*` and `Serialize(ψ*)`. Read the audit with that lens:

- Treat results on `C` as statements about **SHA-256**.
- Treat results on `ψ*` / `Serialize(ψ*)` as statements about **WaveLock's PDE**.

This is why the audit's findings split cleanly into *constraints* (things the
deployment must respect) and *negative results* (attacks that did not work).

---

## Why C-1, C-2, H-1, H-2 are design constraints, not "failures"

The audit's four substantive findings are best read as **operating-mode
boundaries**. Each maps directly to a hardening requirement (see
[`HARDENING_RECOMMENDATIONS.md`](./HARDENING_RECOMMENDATIONS.md)) and to a bounty
target (see [`BOUNTY_SCOPE.md`](./BOUNTY_SCOPE.md)).

| ID | Finding | Why it is a design constraint |
|----|---------|-------------------------------|
| **C-1** | Float reassociation (BLAS/SIMD/compiler/NumPy/GPU build differences) can change `ψ*` and therefore `C` for the **same seed**. | A *consensus* commitment must declare a **single deterministic backend**. The vendor already bans GPU from consensus. The constraint: only the reference NumPy (or a future fixed-point) backend may emit consensus commitments; everything else is research/non-consensus. |
| **C-2** | A low-entropy integer seed (shipped CLI default `42`, 20-bit secret) is recoverable by brute force from `C`. | The commitment's security ceiling is **seed entropy**, not SHA-256's 256 bits. The constraint: production mode must require high-entropy (≥128-bit) input and reject demo seeds. The OTS path already uses 256-bit `os.urandom`. |
| **H-1** | `Serialize(ψ*)` is non-canonical: `+0.0` vs `-0.0`, distinct NaN payloads, and LE (WLv2) vs BE (WLv3) bodies each map one logical field to multiple commitments. | A commitment must be a function of the **logical state**, not its encoding. The constraint: serialization must be canonicalized before hashing (fixed endian/schema/dtype, NaN/Inf rejected, signed zero normalized). |
| **H-2** | `Serialize(ψ*)` / `ψ*` are trivially distinguishable from random (byte χ² ≈ 5.5×10⁶; skew 195; neighbor correlation 0.57). Only SHA-256 hides this. | The PDE provides **no** cryptographic diffusion/confusion. The constraint: never expose `ψ*`, never use a truncated/linear finalizer, never rely on the PDE for entropy — rely on SHA-256/SHAKE over canonical bytes. |

None of these say "the idea is broken." They say: **WaveLock has a correct
operating envelope**, and the bounty/research effort should be about whether that
envelope holds — not about whether the PDE is a standalone hash.

---

## Why the negative results N-1 through N-5 still matter

The audit ran several strong attacks that **did not break** WaveLock. These
negatives are not filler — they bound the realistic attack surface and are what
make the M2M commitment framing credible.

| ID | Attack | Result | Why it matters |
|----|--------|--------|----------------|
| **N-1** | 2nd-preimage / collision over **1,000,000** seeds | **0** exact `ψ*` collisions, **0** commitment collisions; no seed-independent attractor to mass-produce collisions. | The forward map does not cheaply manufacture colliding states; generic collision cost stays SHA-bound (~2¹²⁸). |
| **N-2** | Attractor collapse / entropy | **No** collapse: 1e6/1e6 distinct commitments, all fields finite, entropy preserved over the sweep. | The kernel does not silently funnel many inputs to one output — a prerequisite for a usable commitment layer. |
| **N-3** | Jacobian / Newton inversion `ψ*→ψ0` | Map is chaotic and ill-conditioned; Newton/LM inversion does **not** converge. | No linear-algebra shortcut from a terminal field back to its seed field. |
| **N-4** | Neural / surrogate inversion | `ψ*→ψ0` surrogate R² ≈ 0.03, does not beat the predict-0.5 baseline. | No learned shortcut either; the forward map is not trivially invertible by an ML surrogate. |
| **N-5** | Parameter regimes | Default regime is chaotic/expansive; no tested setting forces collapse or NaN. | Defaults do not accidentally land in a degenerate regime. |

Read together: **no collision, no collapse, no inversion shortcut across a
million seeds.** That is exactly the property you want from a commitment /
attestation primitive whose job is binding and tamper-evidence — not secrecy.

---

## The reframing

The audit clarified the boundary: **WaveLock should currently be tested as M2M
commitment / attestation / replay / drift infrastructure, not as direct payload
encryption or a formal SHA replacement.** Payload secrecy belongs to standard
KEM/AEAD; WaveLock's contribution is commitment, attestation, replay
verification, drift/integrity signaling, and transcript binding — provided the
operating constraints (C-1, C-2, H-1, H-2) are enforced.

WaveLock may still be explored as a longer-term competitor to SHA-like
primitives. That is a **research** ambition, not the present bounty target, and
nothing in this folder claims WaveLock must beat SHA-256 to be useful.

---

## Note on preservation

`audit/REPORT.md` is the **raw hostile audit** and is preserved as-is. Where this
README reinterprets the *significance* of a finding, it never contradicts the
finding's technical content. If the two ever appear to disagree, the measured
results in `REPORT.md` and the JSON artifacts under `audit/artifacts/` are
authoritative.
