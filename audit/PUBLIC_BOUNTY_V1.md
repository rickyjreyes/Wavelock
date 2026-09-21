# WaveLock Public Bounty v1

This is the authoritative technical scope for **the public reference
implementation in `rickyjreyes/Wavelock`**. It offers Critical 1–5, High 1–5,
and the six implemented authenticated-record/OTS categories below. It does not
offer the full historical research scope. Reports must target current master
or an explicitly designated current public-bounty release, with an exact commit.

Use the [submission template](PUBLIC_BOUNTY_SUBMISSION_TEMPLATE.md),
[reproduction guide](PUBLIC_BOUNTY_REPRODUCTION_GUIDE.md), and
[machine-readable matrix](artifacts/public_bounty_v1.json). These specify
technical eligibility; this document does not set reward amounts or payment
terms. Do not infer a security proof from a passing regression.

## Supported boundary

The normative profile is **`WL-Consensus-Commitment-v1`**. Its producer is
`wavelock.chain.consensus_commitment.commit_consensus_state`; its computation
replay verifier is `verify_consensus_commitment` in that same module.
`canonical_serialize` defines the preimage; `validate_canonical_bytes` checks
its encoding. The exact algorithm, parameters, shape, byte order, fixed
transcendental/reduction rules and vectors remain specified by
[BOUNTY_SECURITY_PROFILE.md](../docs/BOUNTY_SECURITY_PROFILE.md) and
[bounty_profile_v1.json](artifacts/bounty_profile_v1.json).

The supported implementation is the unmodified NumPy reference path with the
package's declared Python/NumPy requirements at the reported commit, the
profile's declared arithmetic, and no unsupported backend or evaluation mode.
The CI-tested platform matrix is Ubuntu 24.04/Python 3.9 and 3.12, and
Windows/Python 3.12, plus the CPU container. Dependency compatibility ranges
are not evidence that every combination has been tested. Include exact CPU,
OS, Python, NumPy, CPU-dispatch settings and dependency versions in a report;
the reproduction guide explains how to collect them.

Production input must be `bytes` of length at least 16; 32 random bytes are
recommended. The API checks **type and byte length, not true entropy**. For
Critical 1/2, describe the high-entropy generation/challenge process. Publishing
a witness does not itself make its original generation process low entropy,
but fixed public test vectors and deliberately weak seed spaces are not
high-entropy preimage challenges.

Computation replay, byte validation, OTS signature verification and stateful
record acceptance are distinct boundaries. A public artifact's outer JSON
whitespace/key order is not a second canonical preimage. Encoding a caller's
array alone does not prove it came from an accepted production input. Pure
`verify_ots` intentionally has no replay state. Normal record acceptance uses
OTS-required configuration, header v2, transcript v1 and shared persistent
state through the public node/CLI path. No private repository is needed.

## Critical findings

### Critical 1: canonical commitment collision

Produce two distinct accepted high-entropy production inputs with the same
complete `WL-Consensus-Commitment-v1` commitment. Supply both inputs, their
generation/search method, producer outputs and successful replay verification.
The result must remain within the supported profile. A direct break of SHA-256,
toy integer/demo seeds, historical serializers, rejected inputs or unsupported
backends do not qualify. Collision resistance is an **open attack target**;
existing vectors do not prove it.

### Critical 2: preimage or equivalent-preimage recovery

Given a public artifact generated from an independently sampled input with at
least 128 bits of entropy, recover the original input or another accepted
`bytes` input for which **`verify_consensus_commitment(challenge, recovered)`
returns `True`**. The challenge must not disclose the input, wave/preimage,
private signing material or a deliberately weak generation space to the attack.
Finding a state that only passes `validate_canonical_bytes` is insufficient.

Show substantially less work than generic brute-force search for that challenge
distribution and success probability. Count preprocessing, online work,
reference-producer/hash evaluations, memory and success rate; provide runnable
evidence or a reproducible algorithm with justified work analysis. Reading an
input embedded in a fixture, guessing a public vector, or asserting an
unverified theoretical shortcut does not qualify. The public implementation
has **no claimed formal PDE hardness reduction**; regressions do not prove
preimage hardness.

### Critical 3: supported consensus nondeterminism

Produce different valid canonical commitments for the same accepted production
input under configurations inside the supported profile. Report both complete
environments and preimages/outputs. CPU/SIMD dispatch, supported Python/NumPy
versions, platform differences, rounding and evaluation order are useful attack
surfaces. An unsupported GPU/CuPy, fast-math or reassociation configuration
alone is not a finding; a supported API incorrectly admitting it is High 5,
or Critical 3 when the supported-consensus divergence is independently shown.
Do not modify reference code or vectors to manufacture the disagreement.

### Critical 4: computation replay verification bypass

Make the normative replay verifier accept an invalid declaration involving
wrong input, state, invariants, kernel metadata, parameters, profile, backend,
canonical bytes or commitment. Show the invalid artifact and the actual
successful verifier call. Where authenticated context is involved, also show
acceptance by the OTS/record verifier that actually authenticates that context;
the commitment verifier alone does not authenticate arbitrary application data.
A crash, exception or denial of service without invalid acceptance is not a
replay bypass. Identify collision/preimage findings by their underlying cause
rather than assuming every successful equivalent preimage is a parser defect.

### Critical 5: canonical or semantic binding bypass

After the documented canonicalization rules are correctly followed, demonstrate:

1. Semantically distinct authenticated logical states accepted as the same
   commitment without directly breaking the underlying hash;
2. One logical state with multiple independently valid canonical
   representations through a mechanism not already classified as High 1–3;
3. Producer/verifier disagreement that authenticates one logical state as
   another; or
4. Authenticated metadata/body ambiguity that survives canonical verification.

**High 1–3 do not automatically become Critical 5.** Show the additional
semantic/authentication failure after the documented canonical rules hold.
The claimed rule, state distinction and acceptance boundary must be explicit.

## High findings

| ID | Valid attack |
|---|---|
| High 1 | `validate_canonical_bytes` or normative artifact/replay verification accepts a prohibited noncanonical preimage representation of an otherwise valid state. Equivalent in-memory layouts and outer JSON formatting are not prohibited representations. |
| High 2 | The normative producer, serializer or verifier accepts NaN, +Inf or -Inf in a security-relevant state, invariant or profile field where nonfinite values are prohibited. |
| High 3 | Positive and negative zero representing the same logical state produce different valid canonical bytes/commitments under the profile. |
| High 4 | The normative producer accepts an integer, string, demo seed mode or fewer than 16 raw input bytes as production input. Predictable bytes of adequate length do not bypass a statistical-entropy test: no such test is claimed. |
| High 5 | An unsupported backend/execution mode emits an artifact that the normative verifier accepts as supported profile output. Demonstrate an actual supported API bypass; merely writing a profile label or presenting arbitrary forged bytes is insufficient. |

High 5 tests admission and validation through the supported API. Independently
computing bytes identical to the reference outside that API does not by itself
show a bypass: artifacts do not attest the hardware that computed them. Runtime
provenance is the explicitly withheld Critical 6 property.

## AUTHENTICATED RECORD / OTS FINDINGS

These targets cover only the checks implemented in the public repository.
The accepting node requires OTS, uses intact public verification code, and
shares the configured same-host data/replay paths between cooperating writers.
Tests may inject corrupt records, lost caches or process crashes to exercise
the specified parsing/recovery rules. Controlling the whole host or replacing
the verifier is not part of this threat model.

| ID | Severity | Qualifying invalid acceptance | Public boundary |
|---|---|---|---|
| AR-1 | Critical | A consumed one-time key/leaf is accepted as fresh, including a second message signed with a copied key. | `PersistentOTSReplayLedger.accept`, normal node acceptance; default signer-use guard is also tested. |
| AR-2 | Critical | Body, transcript, authenticated metadata, block type or signed parent is substituted and accepted under an existing authorization. | `verify_ots_block`, canonical OTS transcript v1, node acceptance. |
| AR-3 | Critical | Public-key substitution, fingerprint mismatch, inconsistent Merkle root/committed key material or altered revealed slices pass authentication. | `verify_ots` and canonical public-key/fingerprint checks. |
| AR-4 | Critical | An unauthorized stable header field or mutated authenticated record is accepted into valid chain linkage. | Header v2, `verify_block_integrity`, `verify_ots_chain`, normal node acceptance. |
| AR-5 | High | A malformed persistent replay record is treated as valid, or the offered reload/reconstruction boundary accepts a previously consumed identity after restart. | Replay schema/load checks; `ChainState.load_from_disk`; reconstruction with accepted history retained. Demonstrated unauthorized reuse is adjudicated Critical under AR-1. |
| AR-6 | Critical | A same-host process race causes successful duplicate one-time-key acceptance or incompatible sibling appends through the shared normal acceptance path. | Persistent replay and chain-writer locks; tip reload before acceptance. |

Important distinctions for these tests:

- Repeated `True` from pure `verify_ots` for the same valid signature is
  intentional. Replay findings must cross a stateful acceptance boundary.
- `allow_reuse=True` can construct negative test inputs; its deliberate second
  signature alone is not a bypass. Acceptance of that reused key as fresh is.
- A block's nonce/hash may legitimately change during valid mining. Mutating
  signed timestamp/index/difficulty/context and still obtaining acceptance is
  the header target. Historical header-v1 compatibility is not a v2 guarantee.
- Loss of the replay cache with the accepted chain retained is covered by the
  implemented reconstruction checks. This is not arbitrary cache authentication
  or protection after rollback of all authoritative local state.
- A crash can consume a key before chain append, and malformed/partial storage
  can stop loading. These fail-closed availability outcomes are not successful
  invalid acceptance. Windows directory power-loss durability and distributed
  consensus are not offered properties.

## NOT PART OF PUBLIC BOUNTY V1

The following are **not offered security properties of Public Bounty v1**.
Their absence is not a vulnerability in this scoped public offering.

| Historical target | Public Bounty v1 status | Reason |
|---|---|---|
| Critical 6: remote runtime/machine/kernel/configuration attestation | WITHHELD | Requires a separate trust architecture not provided by the public reference repository. |
| Critical 7: behavior-wide drift-detection evasion | WITHHELD | The public repository does not provide the observation/enforcement boundary needed for the broader claim. |
| Broad Critical 8: hostile rollback, arbitrary valid-format cache tampering, or trusted-anchor guarantees | WITHHELD | Authenticated records and local persistence checks do not constitute a complete hostile-host rollback-resistant system. |

Private/internal components, deployment designs and unpublished implementation
details are out of scope. This scope supplies no proprietary architecture and
does not require access to it. The public optional `expected_tip` parameter is
not an offered mechanism for obtaining a trustworthy external anchor.

## Status and severity

Every public target has four independent matrix fields:

| Field | Values | Meaning |
|---|---|---|
| `implementation_status` | `IMPLEMENTED`, `PARTIAL`, `NOT_IMPLEMENTED` | Whether the offered public computation/checking boundary exists. |
| `regression_status` | `PASS`, `FAIL`, `NOT_APPLICABLE` | Outcome of the identified implementation regressions, not proof that an attack is impossible. |
| `bounty_status` | `OPEN`, `WITHHELD`, `OUT_OF_SCOPE` | Whether this technical target is eligible under this scope. |
| `security_proof` | `NONE`, `REDUCTION`, `FORMAL`, `OTHER` | Established proof status for the full stated target, not for a component hash in isolation. |

Critical 1/2 are `IMPLEMENTED / PASS / OPEN / NONE`: there is a producer/verifier
to attack and regression evidence, with no proof of collision/preimage hardness.
All offered targets remain open to new attacks. Critical severity requires a
core cryptanalytic, authenticated-state or supported-consensus break; High
severity covers an enumerated boundary violation. Classify the demonstrated
failure, not the report's title. Medium/Low categories are not offered here.

## Non-qualifying reports

Pure denial of service, crashes without invalid acceptance, style issues,
documentation typos, theoretical claims without a reproducer, brute forcing
deliberately weak demos, documented historical WLv* behavior, unsupported-mode
nondeterminism, or raw wave distinguishability without security consequences
do not qualify. Neither do standalone assertions that the PDE lacks a proof,
theoretical attacks on SHA-256 itself, generic criticism without a WaveLock
exploit, or attacks against private components absent from this repository.

Known behavior explicitly allowed or withheld above is not a new finding.
Historical audit attacks count only if they still break a currently offered
invariant at the reported current commit. Do not silently broaden exclusions
to dismiss a new invalid acceptance within that boundary.

## Submitting a finding

Supply the exact commit/profile/environment, executable minimal reproducer,
expected and observed behavior, violated invariant, deterministic reproduction
steps, resulting invalid artifact/acceptance where applicable, and severity
rationale. For probabilistic cryptanalysis, include the fixed reproducing
instance, generation process, success probability and complete work accounting.
For a race, record process synchronization and actual acceptance results.

Use the repository's private vulnerability-reporting option if available.
Otherwise request a private reporting route from the maintainer without posting
exploit details publicly. The [template](PUBLIC_BOUNTY_SUBMISSION_TEMPLATE.md)
contains the required evidence; private application or deployment data is not
needed to reproduce the public implementation.
