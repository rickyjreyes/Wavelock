# WaveLock Bounty Scope

This document defines the bounty boundary around WaveLock's **actual invention
boundary**: a machine-to-machine (M2M) **commitment / attestation / replay /
drift-detection** layer. It follows directly from the audit in
[`REPORT.md`](./REPORT.md) and the interpretation in [`README.md`](./README.md).

The bounty is **not** "prove WaveLock beats SHA-256." It is: **given WaveLock's
declared operating envelope, can an attacker produce an invalid acceptance or a
binding failure?** Payload secrecy is explicitly out of scope and handled by
standard KEM/AEAD.

Finding references (C-1, C-2, H-1, H-2, N-1…N-5) point back to
[`REPORT.md`](./REPORT.md).

---

## Definitions

- **Consensus mode** — a declared, deterministic configuration (reference NumPy
  backend, fixed parameters, canonical serialization) that is permitted to emit
  consensus commitments. GPU/CuPy/fast-math/reassociation builds are
  research/non-consensus unless made deterministic.
- **Canonical commitment** — `C = SHA256(CanonicalSerialize(ψ*, metadata))`
  where the serialization is canonicalized (fixed endian/schema/dtype, NaN/Inf
  rejected, `-0.0` normalized to `+0.0`) and bound to kernel hash, params,
  lattice dimensions, steps, and XOF id.
- **High-entropy input** — ≥128-bit secret/seed material (production mode).
  Integer/demo seeds are explicitly non-production.
- **Replay verification** — checking that a presented `ψ*` and its invariants
  match the declared kernel metadata, params, seed/input, and recorded
  transcript.

---

## Critical bounty targets

A valid submission for any of these demonstrates a real break of the invention
boundary.

1. **Canonical commitment collision (high-entropy).** Produce two **distinct
   high-entropy inputs** that yield the **same canonical WaveLock commitment**
   under a declared consensus mode. *(Tightens N-1, which found 0 collisions in
   1M low-entropy seeds.)*
2. **Preimage faster than brute force.** Recover a **≥128-bit high-entropy
   input**, or an equivalent preimage, **faster than brute force**. *(Distinct
   from C-2, which is a low-entropy-seed weakness, not a break of the primitive.)*
3. **Consensus nondeterminism.** Produce **different commitments for the same
   input** under a **declared supported consensus mode**. *(C-1 shows this is
   real for arbitrary float reassociation; the bounty target is to do it within
   a configuration the project declares as consensus-valid.)*
4. **Replay verification bypass.** Make replay verification **accept** any of:
   wrong kernel metadata, wrong params, wrong seed/input, wrong `ψ*`, or tampered
   invariants.
5. **Canonical serialization bypass.** Make one logical state map to **multiple
   valid commitments**, or two distinct logical states map to **one valid
   commitment**, despite canonicalization. *(Strengthens H-1 against the
   canonical encoding.)*
6. **Attestation spoof.** Spoof machine attestation with the **wrong
   runtime/kernel/config** while still passing verification.
7. **Drift-detection evasion.** **Evade drift detection** while **materially
   changing** the target's observable behavior.
8. **Silent ledger/record tampering.** Modify a ledger entry, record, or
   invariant field **without** triggering a verification failure.

---

## High bounty targets

Valid but lower-severity than Critical — typically a violation of a single
hardening constraint rather than a full binding break.

1. **Noncanonical serialization acceptance.** Show a `Serialize(ψ*)` that is
   **not** canonical yet is accepted. *(H-1.)*
2. **NaN/Inf acceptance.** Trigger acceptance of a **NaN or Inf** payload in
   commitment mode. *(H-1.)*
3. **Signed-zero divergence.** Cause `-0.0` vs `+0.0` to produce **divergent
   commitments** in consensus mode. *(H-1.)*
4. **Sub-threshold seed acceptance.** Show that seed/input entropy **below the
   required threshold** is accepted in production mode. *(C-2.)*
5. **Unsupported backend leakage.** Show that an **unsupported backend silently
   emits consensus commitments**. *(C-1.)*

---

## Out of scope

These do **not** qualify for the bounty:

1. **Breaking SHA-256 / SHA3 directly.** WaveLock's one-wayness is inherited from
   SHA-256; attacking the hash itself is out of scope.
2. **Direct payload encryption claims.** WaveLock is not a cipher; payload
   secrecy is delegated to standard KEM/AEAD.
3. **CurvaChain / OTS replay** — unless specifically **bound into** WaveLock
   replay verification.
4. **Denial-of-service** without an accompanying **invalid acceptance**.
5. **Merely distinguishing raw `ψ*` from random** *(H-2)* — unless it **leads to**
   a commitment, replay, attestation, or drift failure. Distinguishability of the
   pre-hash object alone is a known, documented constraint, not a finding.
6. **"WaveLock must beat SHA-256 to be useful"** claims. The bounty is about the
   M2M invention boundary, not a SHA replacement contest.

---

## How targets map to the audit

| Target | Related audit finding |
|--------|-----------------------|
| Critical 1 (canonical collision) | N-1 (0 collisions / 1M seeds) |
| Critical 2 (≥128-bit preimage) | N-3, N-4 (no inversion shortcut) |
| Critical 3 (consensus nondeterminism) | C-1 (float reassociation) |
| Critical 4 (replay bypass) | new layer (replay not tested in REPORT) |
| Critical 5 (serialization bypass) | H-1 (non-canonical serialization) |
| Critical 6 (attestation spoof) | new layer |
| Critical 7 (drift evasion) | new layer |
| Critical 8 (silent tampering) | new layer / N-2 |
| High 1–3 (noncanonical / NaN-Inf / signed zero) | H-1 |
| High 4 (sub-threshold seed) | C-2 |
| High 5 (backend leakage) | C-1 |

Replay, attestation, and drift layers were **not** exercised by the original
hostile audit; they are the natural next attack surface once the commitment-mode
constraints (C-1, C-2, H-1, H-2) are hardened.
