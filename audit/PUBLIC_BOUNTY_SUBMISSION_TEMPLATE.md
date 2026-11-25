# Public Bounty v1 submission

Use the repository's private vulnerability-reporting option if available.
Otherwise request a private reporting route from the maintainer before posting
exploit details. Use disposable local instances and test data. No private
WaveLock component is required.

## Identification and environment

- Title and target ID (`C1`–`C5`, `H1`–`H5`, `AR-1`–`AR-6`):
- Proposed severity and rationale:
- Exact repository commit (full SHA):
- Current master/current designated public-bounty release:
- Exact profile (`WL-Consensus-Commitment-v1`, or relevant OTS/header boundary):
- Entry point/verifier that accepts the invalid result:
- OS/version/architecture and CPU model/SIMD features:
- Python implementation/version; NumPy version/build/runtime report:
- Dependency lock or `pip freeze` output:
- Relevant configuration/environment, including CPU-dispatch settings:
- Replay/persistence reports: filesystem, data/replay/lock paths, process count:

Remove real secrets, access tokens, personal paths and unrelated private data
from diagnostic output. Use disposable keys and records.

## Published invariant and violation

- Exact rule in `audit/PUBLIC_BOUNTY_V1.md`:
- Expected behavior:
- Observed behavior:
- Why this is invalid acceptance, a cryptanalytic shortcut or supported divergence:
- Why it is not explicitly permitted behavior or a withheld target:
- For Critical 5: the deeper binding failure after canonical rules hold, and
  why it is not merely High 1–3:

## Minimal executable reproducer

Attach a self-contained script, fixtures and exact commands using the public
repository and unmodified verifier. Identify harness instrumentation separately.

1. Clean checkout and dependency installation:
2. Isolated data/signing/replay state setup:
3. Input and challenge construction:
4. Attack command:
5. Exact verification/acceptance command:
6. Expected versus actual output and exit status:
7. Deterministic repetition, restart or synchronization steps:

## Evidence and impact

- Invalid accepted artifact, canonical bytes, signature or record, as applicable:
- Complete verifier result/acceptance log:
- Legitimate baseline control and changes made to it:
- Fixed reproducing instance, reproduction rate and number of attempts:
- Concrete effect within the public offered boundary:
- Relevant regressions and known related reports:
- Optional remediation (do not replace vectors to hide disagreement):

For collision/preimage reports, include the original generation/challenge
process, qualifying input space, success predicate, algorithm, preprocessing,
online work, memory, success probability and a generic brute-force comparison
for the same distribution. An unreproduced complexity assertion or recovery of
a deliberately weak public seed is insufficient.

For races, include synchronization, shared state paths, both actual acceptance
results and resulting durable records. For restarts, identify exactly which
data survived. Pure signature verification does not consume identities;
complete host rollback is not covered by this offering.

The absence of a formal proof alone is not a finding. Withheld attestation,
behavior-wide drift, broad hostile rollback and private components absent from
this repository are outside Public Bounty v1.
