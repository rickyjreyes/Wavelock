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

**Pre-bounty implementation status:** the exact supported commitment contract
is [`docs/BOUNTY_SECURITY_PROFILE.md`](../docs/BOUNTY_SECURITY_PROFILE.md).
Use its normative API/profile for submissions; historical WLv* research
serializers do not emit that profile. The original targets below remain visible
so missing controls cannot be hidden by changing the wording. The
[machine-readable matrix](./artifacts/bounty_contract_matrix.json) records
Critical 6 and 7 as unimplemented and the broad Critical 8 claim as incomplete.
**The full bounty is not ready to offer.** See the
[hardening report](./BOUNTY_HARDENING_REPORT.md) for tested controls, remaining
failures and targets suitable for focused external research.

---

## Definitions

- **Consensus mode** — `WL-Consensus-Commitment-v1` produced by
  `wavelock.chain.consensus_commitment.commit_consensus_state`, with the fixed
  NumPy reference backend and complete parameters/encoding in the normative
  profile. GPU/CuPy/fast-math/reassociation and historical paths cannot emit
  this profile through the supported API.
- **Canonical commitment** — `C = SHA256(CanonicalSerialize(ψ*, metadata))`
  where the serialization is canonicalized (fixed endian/schema/dtype, NaN/Inf
  rejected, `-0.0` normalized to `+0.0`) and bound to kernel hash, params,
  lattice dimensions, steps, and XOF id.
- **High-entropy input** — independently generated ≥128-bit secret material.
  Production code requires explicit `bytes` of length ≥16 (recommended 32).
  This is a representation/length check, not statistical entropy estimation.
  Integer/string/demo seeds are rejected by the normative producer.
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
5. **Canonical serialization/binding bypass.** After all documented
   canonicalization rules are correctly enforced, demonstrate one of:
   - two semantically distinct authenticated logical states producing the same
     canonical commitment without breaking the underlying hash;
   - one logical state producing multiple independently valid canonical
     representations through a mechanism not already enumerated as a High
     canonicalization violation;
   - verifier/producer disagreement that accepts an invalid logical state as
     another authenticated state.

   **High 1–3 do not automatically escalate to Critical 5 merely because they
   are serialization-related.** A report must demonstrate the additional
   binding failure, not just repeat one of the enumerated High violations.
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
   required threshold** is accepted in production mode. The executable
   submission is acceptance of an integer/demo/string input or fewer than
   16 raw secret bytes by the normative API. Actual entropy of caller-chosen
   bytes cannot be inferred from their value or length. *(C-2.)*
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
hostile audit. The hardening branch now tests commitment computation replay
and OTS-authenticated record/replay controls. Remote attestation and a
behavior-wide drift verifier still do not exist; those are known implementation
gaps, not newly secured targets. Anchored record verification is also narrower
than a promise to detect every rollback or hostile local cache edit.
