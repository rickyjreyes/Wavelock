# Pre-bounty hardening report

Branch: `hardening/bounty-contract-v020`. Baseline: unchanged `v0.2.0`, commit
`9430442ae93e26c192d9be1f5ee9137369f5df37`. Review:
[PR #22](https://github.com/rickyjreyes/Wavelock/pull/22). Do not merge
automatically.

**High 1–5 now have enforced controls and direct adversarial regressions. All
thirteen specified dry runs pass. The full advertised bounty is not ready:**
Critical 6 and 7 lack implementing verifiers, and Critical 8 remains incomplete
against rollback or arbitrary hostile edits to local storage. Those targets
remain in the scope and matrix as unresolved; documentation does not turn them
into passes. Passing enforcement tests is not a cryptographic security proof.

## Requested before/after matrix

`cc` below means `wavelock/chain/consensus_commitment.py`; `contract` means
`tests/test_bounty_contract.py`. The full machine-readable mapping is
[bounty_contract_matrix.json](artifacts/bounty_contract_matrix.json).

| Target | Before at v0.2.0 | After | Enforcing code | Regression |
|---|---|---|---|---|
| High 1 | Native-endian historical serializer; no complete normative descriptor | PASS | `cc.canonical_serialize`, `validate_canonical_bytes`, `_metadata` | `contract::test_high1_canonical_layout_and_dictionary_order`, `test_high1_alternate_wire_encoding_rejected` |
| High 2 | NaN, +Inf and -Inf serialize successfully | PASS | `cc._state`, `_float`, `_wire`, `reference_invariants` | Every state position, parameter and invariant injected with all three nonfinite values |
| High 3 | Isolated negative zero changes historical bytes | PASS | `cc._state`, `_float` | `contract::test_high3_signed_zero_in_state_and_invariants` |
| High 4 | `seed=42` produces an unqualified historical commitment | PASS | `cc._validate_input` | Integers/string/demo/8/15-byte rejection; 16/24/32-byte golden vectors |
| High 5 | No single programmatically enforced bounty producer | PASS | `cc.commit_consensus_state`; historical serializer label allowlist | Unsupported backend/profile/fast-math/reassociation rejection; historical output rejected by normative verifier |
| Critical 5 boundary | Overlapped High 1–3 | PASS boundary | `BOUNTY_SCOPE.md`, `cc.validate_canonical_bytes`, `verify_consensus_commitment` | Alternate encoding, bound metadata and producer/replay disagreement tests |

The normative API is `commit_consensus_state(secret_input: bytes)`. Its fixed
profile is `WL-Consensus-Commitment-v1`; complete byte construction, equations,
parameters, descriptor and input rules are in
[BOUNTY_SECURITY_PROFILE.md](../docs/BOUNTY_SECURITY_PROFILE.md). Historical
encodings are retained with non-consensus markers and cannot pass this verifier.
The new profile does not replace OTS or reinterpret WLv2.

## Confirmed vulnerabilities fixed

- Final-head CI found a real **Critical 3** numerical mismatch in the initial
  implementation: the 32-byte vector gave `aff7f2d5...` on one CPU dispatch path
  and `be8ea373...` on another. Disabling AVX512 reproduced it locally without
  changing NumPy versions. [Before evidence](artifacts/bounty_dispatch_before.json)
  retains both hashes and the failed CI run. Fixed decimal-80 exp/log rounding
  and an explicit 16-element reduction now eliminate that dispatch choice from
  the NumPy reference path. The PDE and parameters are unchanged. The updated,
  still-unreleased descriptor binds those math rules, so whole-preimage vectors
  change; the **original reference state/invariant bodies remain byte-identical**.
  [After evidence](artifacts/bounty_dispatch_after.json) and subprocess tests
  require one common vector set across three CPU-feature settings. There are
  no platform-specific expected hashes. NumPy's
  [runtime dispatcher](https://numpy.org/doc/stable/reference/simd/index.html)
  explains why selecting NumPy alone was insufficient.
- The old serializer emits 405 bytes for each nonfinite probe, diverges on
  signed zero, uses native-endian array bytes and does not bind step count,
  damping or XOF identity. Integer 42 also emits a historical commitment.
  [Baseline observations](artifacts/bounty_v020_baseline.json) were reproduced
  on a detached v0.2.0 checkout using
  [reproduce_v020_contract.py](reproduce_v020_contract.py). The new API closes
  those acceptance paths while preserving historical meanings.
- A concrete header ambiguity exists: `(index=1,timestamp="23")` and
  `(index=12,timestamp="3")`, with all other fields equal, hash to the same
  legacy value `4914a9e6406485270303bdc82222e8a2c57a74f99bfd4d5ea4df88ce4d2785e8`.
  This is the same byte-string input, not a SHA-256 collision. Header v2 frames
  named fields canonically. A reduced 40×40 enumeration demonstrates legacy
  duplicates and 1600 distinct v2 encodings/hashes.
- Merely rehashing a modified timestamp was possible under the original
  signature boundary. New OTS producers bind timestamp/index/difficulty/version
  through reserved signed metadata. Existing transcript-v1 structure/domain
  remain unchanged; old stored blocks keep their original verification rules.
- Exclusive artifacts could become visible before completion, and directory
  entries were not flushed after important writes. Complete temporary files
  are now flushed before atomic publication; POSIX parents are flushed too.
  A failed signer claim or uncertain durable write never authorizes reuse.
- Replay readers could overlook valid-JSON malformed records and race a writer.
  Record schema/version/identifier checks and locked reads now fail closed.
  Chain acceptance additionally locks and reloads the tip across processes,
  preventing distinct-key sibling blocks from both appending to one local tip.

## Suspected issues disproven or bounded

- Equivalent Python dictionary order, endian/layout/construction differences
  do not change normative bytes. Little-endian *wire substitution* is rejected.
- Signing does not need a new OTS transcript version. Its existing signed
  metadata extension binds the v2 header context, while historical v1 records
  remain readable. A v2-to-v1 downgrade fails verification/normal acceptance.
- An OTS body mutation or public-key substitution is rejected even after
  recomputing the Merkle root/header; replay after restart is rejected. These
  pre-existing OTS controls remain effective under the new persistence path.
- Same-host process races were exercised, including POSIX and forced SQLite
  replay locks. Two competing processes do not both consume the same OTS key;
  distinct-key sibling writers do not both append. This result says nothing
  about agreement between hosts.
- NumPy alone was **not** sufficient to guarantee CPU parity: the follow-up CI
  failure above disproved that assumption. After correction, parity with the
  independent historical NumPy evolution is tested using the same declared
  exp/log primitives. Historical libm output is not claimed to be byte-identical.
  New exact vectors also preserve the original reference body hashes. Bounded
  conformance still does not prove every arbitrary build/input deterministic.

The header concatenation concern was **confirmed**, not disproven. No result
here proves collision resistance, preimage resistance or absence of new attacks.

## Remaining known limitations

- **Critical 6: missing runtime attestation.** A bound kernel descriptor and
  reproducible output do not establish which process or machine executed it.
- **Critical 7: missing behavior-wide drift detection.** There is no declared
  observation oracle or verifier that could enforce the broad behavior claim.
- **Critical 8: incomplete tamper/freshness coverage.** Signed record mutations
  and malformed cache records are rejected, but a valid chain prefix passes
  without an external anchor. The new optional `expected_tip` check detects
  truncation only when supplied a trusted tip. The default node does not fetch
  such a checkpoint. Valid-format edits to the unsigned replay cache are not
  fully authenticated. An attacker controlling every local state copy can
  restore older signing/replay history. These are unresolved requirements,
  not scope exclusions invented to make the matrix green.
- Replay consumption precedes block append in two separate files. A crash can
  burn a key without accepting its block. Keep both replay and signing state;
  rebuilding from the chain recovers accepted identities, not every burned
  identity. Interrupted JSONL writes cause a load failure requiring recovery.
- Windows has atomic publication and flushed file content, but this portable
  implementation has no parent-directory power-loss flush guarantee. Tests
  simulate process death and failed fsync; they do not simulate physical media
  loss. Hostile storage and distributed replay consensus remain outside the
  guarantees actually established by same-host locking.
- Byte length cannot establish entropy. Floating-point reference conformance
  vectors are not an all-input determinism proof. Canonical replay requires
  the verifier to possess the input; public artifact consistency alone is not
  secret-input proof or remote attestation. OTS remains experimental.

## External research targets and offer decision

Appropriate for a **focused review of the implemented profile**: High 1–5,
Critical 1/2 (collision/preimage attacks with qualifying random inputs),
Critical 3 (nondeterminism within a supported reference configuration),
Critical 4 (computation replay binding), and Critical 5 (a deeper binding break
after canonical rules hold). Existing authenticated-body, identity and replay
checks are also meaningful research surfaces, including the anchored subset
of Critical 8. Label these as attack targets, never proven security guarantees.

**Do not offer the full advertised scope yet.** Critical 6/7 need implementation
and adversarial tests; broad Critical 8 needs authenticated checkpoint/storage
semantics and a recovery threat model. Preserve these requirements as open
work. Known missing controls are not challenges researchers should be paid to
rediscover. Thirteen specified dry-run passes do not close these wider gaps.

## Verification and reproduction

- Core/OTS suite: **424 passed, 7 skipped**, including canonicalization,
  signer/replay crash points, subprocess races, header/history and installed
  workflow tests. Optional/skipped historical tests retain their existing gates.
- Unchanged curvature suite: **205 passed**. Unchanged PDE suite:
  **113 passed, 1 deselected** (the existing slow gate).
- First implementation CI: Linux/Python 3.9, Linux/Python 3.12, Windows/Python
  3.12 and container all passed in
  [run 35531330185](https://github.com/rickyjreyes/Wavelock/actions/runs/35531330185).
  Follow-up [run 35531961179](https://github.com/rickyjreyes/Wavelock/actions/runs/35531961179)
  exposed the CPU-dispatch failure; it was retained and fixed, not rerun away.
  The same jobs plus explicit dispatch subprocess regressions run on every PR
  update; use the PR checks for the final head.
- **13/13 specified submissions PASS**, independently reproducible with
  `python audit/run_bounty_contract.py --check`. Each attempt includes target,
  attempt, expected, observed and status in
  [bounty_dry_run/INDEX.json](artifacts/bounty_dry_run/INDEX.json). CI reruns and
  compares these artifacts on every supported platform. Expected signed-zero
  equality and internal-endian normalization are distinguished from rejection.
- CPU dispatch has a separate before/after artifact; its checks preserve the
  requested thirteen-entry dry-run index and also fail CI on a mismatch.

Commands are in [RUNBOOK.md](../RUNBOOK.md). No research-suite source was
changed. The implementation branch remains separate; `v0.2.0` is unchanged.
