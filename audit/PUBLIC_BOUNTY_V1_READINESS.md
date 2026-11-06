# Public Bounty v1 finalization review

Public v1 defines **16 offered attack targets** on the existing public reference
implementation: Critical 1–5, High 1–5 and AR-1–AR-6. Its three withheld targets
are not offered security properties. The change adds scope, evidence mappings
and reproduction/submission instructions; it adds no security implementation,
private architecture or proprietary material.

Baseline: current master after merged PR #22,
`c4b1df112c93db8017b788d7f8f58b4f0e2087f9`.
Branch: `bounty/public-v1-finalization`. The PR is for review and must remain
unmerged until the maintainer chooses to merge it.

## Exact public text and changed files

The exact public bounty text is [PUBLIC_BOUNTY_V1.md](PUBLIC_BOUNTY_V1.md).
The complete change consists of these eight files:

| File | Change |
|---|---|
| `audit/PUBLIC_BOUNTY_V1.md` | Authoritative public scope, success predicates, severity, exclusions and withheld boundaries |
| `audit/artifacts/public_bounty_v1.json` | Sixteen targets with all required code/verifier/test/evidence fields and separate status dimensions; three withheld entries |
| `audit/PUBLIC_BOUNTY_SUBMISSION_TEMPLATE.md` | Commit/environment, reproducer, expected/observed acceptance and severity evidence |
| `audit/PUBLIC_BOUNTY_REPRODUCTION_GUIDE.md` | Public-only setup, challenge/verification examples and existing tests for every target |
| `audit/BOUNTY_SCOPE.md` | Index distinguishing current public scope, historical full research scope and withheld surfaces |
| `README.md` | Concise public-scope and reproduction links |
| `docs/BOUNTY_SECURITY_PROFILE.md` | Introductory status/eligibility links only; normative algorithm and vectors unchanged |
| `audit/PUBLIC_BOUNTY_V1_READINESS.md` | This review record |

## Final offered and withheld targets

Critical: C1 canonical commitment collision; C2 preimage/equivalent-preimage
recovery; C3 supported consensus nondeterminism; C4 computation replay bypass;
C5 deeper canonical/semantic binding failure.

High: H1 noncanonical acceptance; H2 NaN/Inf acceptance; H3 signed-zero
divergence; H4 production input boundary bypass; H5 unsupported backend admission.

Authenticated records: AR-1 one-time identity/reuse acceptance; AR-2 signed
body/transcript/context binding; AR-3 key/fingerprint/Merkle consistency;
AR-4 authenticated header/linkage; AR-5 replay parsing/restart reconstruction;
AR-6 same-host process-race exclusion. AR-5 is High for a parsing/recovery
boundary violation; demonstrated unauthorized reuse has Critical AR-1 impact.

Withheld: C6 remote runtime/machine/kernel/configuration attestation; C7
behavior-wide drift-detection evasion; broad C8 hostile rollback, arbitrary
valid-format cache tampering and trusted-anchor guarantees. These appear only
in the new matrix's `withheld_targets` array, not its offered `targets` array.

## Before and after

| Target group | Earlier inventory | Public v1 status |
|---|---|---|
| Critical 1/2 | A single PASS field with caveats could be mistaken for a hardness conclusion | `IMPLEMENTED / PASS / OPEN / NONE`: public attack boundary implemented, mapped regressions pass, attack target open, no full-property proof |
| Critical 3–5 | Defensive checks and historical full scope shared one matrix | `IMPLEMENTED / PASS / OPEN / NONE`, with precise success predicates and supported profile |
| High 1–5 | Hardened controls and regressions existed | `IMPLEMENTED / PASS / OPEN / NONE`, with explicit boundary-bypass definitions |
| Authenticated records/OTS | Spread across implementation tests and broad Critical 8 language | Six named categories limited to the checks actually exposed publicly |
| Critical 6/7 | NOT IMPLEMENTED in full-research inventory | `NOT_IMPLEMENTED / NOT_APPLICABLE / WITHHELD / NONE`; not offered public v1 properties |
| Broad Critical 8 | Partial implementation for the broad research ambition | `PARTIAL / NOT_APPLICABLE / WITHHELD / NONE`; public local checks separately offered as AR targets |

The historical matrix, hardening report, dry-run evidence and full-scope
readiness flag are preserved. No former failure is relabeled a defensive pass.
Public v1 is a separately specified offering authorized by this finalization
brief; its openness is not a claim of cryptographic proof.

## Validation counts

All existing test sources, runtime code, workflows and reference vectors remain
unchanged. Fresh local runs on Python 3.12 used the commands in the reproduction
guide, with no slow-test deselection:

| Suite/check | Result |
|---|---|
| Core suite | **424 passed, 7 skipped** |
| OTS security/red-team/consensus/restart/roundtrip subset | **74 passed**, included in core |
| Bounty contract subset | **153 passed**, included in core |
| Durability subset | **12 passed**, included in core |
| Header subset | **11 passed**, included in core |
| Curvature suite | **205 passed** |
| Full PDE suite | **114 passed**, including the 10,000-message slow test |
| Existing bounty dry runs | **13/13 PASS**, checked-in evidence unchanged |
| CPU-dispatch comparison | **3/3 configurations pass**, common fixed vectors |
| Supported CLI/OTS demo | PASS, including public-only verification after restart and replay rejection |

The total across the three disjoint suites is **743 passed, 7 skipped**;
subsets must not be added again. Six skips are optional historical CuPy tests;
one is the pre-existing legacy SIGv2 collection skip. Neither is an offered
normative public v1 control.

Final PR CI is the required release gate: Linux/Python 3.9, Linux/Python 3.12,
Windows/Python 3.12, and CPU container. The PR description and delivery report
record its exact run URL/results. The existing research workflows have path
filters and need not trigger on this documentation change; the full research
suites were run locally. A filtered job is not reported as passing.

## Wording self-audit

| Question | Resolution |
|---|---|
| More promised than enforced? | Success predicates name actual public APIs; arbitrary application authentication, hostile-host guarantees and runtime provenance are not inferred from commitment bytes. |
| Known behavior mistaken for a payable new finding? | Known historical/allowed/withheld behaviors are distinguished from a still-reproducible break at the current offered commit. No payment amount/terms are invented. |
| High 1–3 misclassified as Critical 5? | Explicitly prohibited; a deeper semantic failure after canonical rules hold is required. |
| Collision resistance or preimage hardness implied by PASS? | Four independent status fields; C1/C2 explicitly have `security_proof: NONE`. |
| False PDE reduction claim? | None asserted; cryptanalytic work must be justified against the stated challenge distribution. |
| Backend admission confused with runtime attestation? | H5 requires an actual supported API bypass; equivalent bytes computed elsewhere do not prove hardware provenance. |
| OTS replay confused with pure verification? | Repeated pure `verify_ots` success is intentional; replay requires stateful acceptance. |
| Header mutation overclaimed? | Stable signed context is covered; legitimate nonce/hash mining and historical v1 behavior are distinguished. |
| Cache recovery overclaimed? | Retained accepted history and documented schema checks are covered; all-state rollback and arbitrary valid-format cache authenticity are withheld. |
| Private architecture exposed? | Only existing public code and generic withheld-property names are used. No private repository/source was consulted or added. |
| Historical encodings confused with normative profile? | Explicit exclusion and exact `WL-Consensus-Commitment-v1` producer/verifier references. |
| Every offered target independently testable? | All enforcing symbols, verifier symbols and test node IDs resolve in the public repository; executable setup/challenge/verifier fixtures are documented. An undiscovered winning attack is not promised. |

No implementation defect was discovered in this finalization pass. No reference
vector was changed. A new implementation defect or failing invariant would stop
readiness and be reported without silently changing vectors.

`v0.2.0` remains the preserved stable commit
`9430442ae93e26c192d9be1f5ee9137369f5df37`; no tag move, new release, automatic
merge or bounty announcement is part of this PR. The readiness decision is
**YES only when the final PR's required platform/container checks pass**. It
applies to Public Bounty v1 under this document, not the full research scope.
