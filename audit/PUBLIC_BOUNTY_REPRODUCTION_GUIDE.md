# Reproducing Public Bounty v1

All offered targets use the public repository. Read the authoritative
[scope](PUBLIC_BOUNTY_V1.md) and [code/test matrix](artifacts/public_bounty_v1.json).
The commands reproduce defended boundaries and fixtures, not winning attacks.
No private repository, deployment or production secret is required.

## Checkout and environment

Use current master or the current designated public-bounty release. Record the
exact commit before testing:

```bash
git clone https://github.com/rickyjreyes/Wavelock.git
cd Wavelock
git switch master
git rev-parse HEAD
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Bash, or
`.\.venv\Scripts\Activate.ps1` on PowerShell, then:

```bash
python -m pip install -e ".[blake3]" pytest
python -c "import platform,sys,numpy; print(platform.platform()); print(platform.machine(), platform.processor()); print(sys.version); print(numpy.__version__); numpy.show_config()"
python -m pip freeze
```

When available, also run `python -c "import numpy; numpy.show_runtime()"`.
Record `NPY_DISABLE_CPU_FEATURES` and any rounding/configuration changes.
The reported commit determines dependency requirements and reference code;
compatibility ranges do not substitute for exact versions. The automated
platform matrix is Ubuntu 24.04/Python 3.9 and 3.12, Windows/Python 3.12 and
the CPU container.

## Complete existing suites

```bash
python -m pytest tests/ -q
python -m pytest curvature_audit/ -c curvature_audit/pytest.ini -q
python -m pytest pde_audit/ -c pde_audit/pytest.ini -q
python audit/run_bounty_contract.py --check
python hello_wavelock.py
```

These research commands include the existing PDE `slow` 10,000-message parity
test. For standard faster CI research gates, add `-m "not slow"` and report the
deselection. Core discovery retains existing exclusions; optional CuPy
historical tests and the legacy SIGv2 collection skip do not exercise the
offered NumPy profile. Preserve tests, vectors and existing fixtures.

The dry-run command repeats all thirteen saved attempts and three CPU-dispatch
configurations, then compares evidence without overwriting it. Its historical
`full_bounty_ready: false` field concerns the broader full research scope.
Public v1 eligibility comes from the separate public matrix; the old flag is
not rewritten to hide that distinction.

## Commitment and computation replay

Save and run this disposable mechanism check:

```python
import copy
from wavelock.chain import consensus_commitment as cc

seed = bytes(range(32))  # Public test vector, not a cryptanalytic challenge.
artifact = cc.commit_consensus_state(seed)
assert cc.validate_canonical_bytes(artifact.canonical_bytes)
assert cc.verify_consensus_commitment(artifact, seed)
changed = copy.deepcopy(artifact.to_dict())
changed["metadata"]["kernel"]["steps"] = 49
assert not cc.verify_consensus_commitment(changed, seed)
assert not cc.verify_consensus_commitment(artifact, bytes(range(1, 33)))
print(artifact.commitment)
```

Capture `artifact.canonical_bytes.hex()` for preimage comparisons. The golden
vector is pinned in `tests/test_bounty_contract.py` and the normative profile.
Outer JSON formatting is not canonical preimage formatting. Byte consistency
alone is not input provenance or runtime attestation.

A separate evaluator can generate a fresh public preimage challenge:

```python
import secrets
from wavelock.chain.consensus_commitment import commit_consensus_state, write_commitment_artifact

write_commitment_artifact("challenge.json", commit_consensus_state(secrets.token_bytes(32)))
```

The public file omits the input and wave and refuses overwrite. Freeze that
public challenge for repeatable experiments and document its generation
process without supplying the attack its originating secret. Verify a recovered
candidate using the actual success predicate:

```python
import json
from wavelock.chain.consensus_commitment import verify_consensus_commitment

challenge = json.load(open("challenge.json", encoding="utf-8"))
candidate = bytes.fromhex("REPLACE_WITH_RECOVERED_INPUT_HEX")
assert verify_consensus_commitment(challenge, candidate)
```

The candidate line is a report placeholder, not an attack supplied by the
project. Critical 2 also requires complete work/success analysis. For Critical
1, supply distinct qualifying inputs, equal producer commitments and successful
replay for both. No winning collision/preimage fixture is claimed to exist.

## Targeted commitment probes

Run `python -m pytest tests/test_bounty_contract.py -q`.

| Target | Existing entry points in that file |
|---|---|
| C1/C2 | `test_high4_accepted_inputs_and_pinned_reference_parity`, `test_commitment_artifact_publishing_is_public_and_exclusive`; interface checks, not hardness proofs |
| C3 | `test_reference_vectors_independent_of_simd_dispatch`, `test_reference_error_policy_does_not_inherit_ambient_numpy_settings` |
| C4 | `test_every_metadata_field_is_bound`, `test_replay_rejects_wrong_input_state_invariants_and_extra_fields` |
| C5 | Replay-binding and alternate-wire tests; a finding needs the additional semantic break defined in the scope |
| H1 | `test_high1_canonical_layout_and_dictionary_order`, `test_high1_alternate_wire_encoding_rejected`, `test_high1_wrong_dtype_rejected` |
| H2 | Every-state-position, every-parameter, every-invariant and derived-overflow/producer-injection tests prefixed `test_high2_` |
| H3 | `test_high3_signed_zero_in_state_and_invariants` |
| H4 | `test_high4_production_seed_threshold`, accepted-input vectors |
| H5 | `test_high5_unsupported_backend_or_profile`, `test_historical_modes_cannot_masquerade_as_consensus` |

C3's subprocess test compares default dispatch, `AVX512F` disabled, and
`AVX512F,AVX2,FMA3,AVX` disabled against one pinned vector set. A divergent
environment must still use the declared reference arithmetic; raw NumPy
research behavior does not replace the fixed transcendental/reduction rules.

## Authenticated records, OTS and replay

The existing tests isolate signer state and use disposable keys. The hello
demo uses temporary state. For a manual fixture, set `WAVELOCK_DATA_DIR` and
`WAVELOCK_OTS_STATE_DIR` to temporary directories before importing the node;
do not inherit a production `WAVELOCK_OTS_LEDGER` or node configuration.

```bash
python -m pytest tests/test_ots_security.py tests/test_ots_redteam.py tests/test_ots_consensus.py tests/test_ots_mythos_break.py tests/test_ots_roundtrip.py -q
python -m pytest tests/test_bounty_durability.py tests/test_bounty_headers.py tests/test_supported_workflow.py -q
```

| Target | Public entry points and existing fixtures |
|---|---|
| AR-1 | `PersistentOTSReplayLedger.accept`; same-signature replay and different-message/copied-key tests in `test_ots_consensus.py` |
| AR-2 | `build_signed_ots_block` / `verify_ots_block`; M1 body/transcript tests in `test_ots_mythos_break.py` |
| AR-3 | `verify_ots`; Merkle/key/fingerprint/revealed-slice tests in `test_ots_redteam.py` and `test_ots_consensus.py` |
| AR-4 | `verify_ots_chain`, `try_accept_block`; signed-header re-mining, downgrade/history/linkage tests in `test_bounty_headers.py` and `test_supported_workflow.py` |
| AR-5 | Replay ledger/chain reload; malformed-record, reopen, M2 reconstruction and interrupted-append tests |
| AR-6 | Shared-path worker fixtures in `test_bounty_durability.py`; M3 duplicate acceptance and SQLite fallback tests |

Pure `verify_ots` can repeatedly return `True`. Replay findings must demonstrate
successful stateful acceptance. Record exactly which state survives a restart.
Cache-loss reconstruction retains accepted history; it does not cover rollback
of all local history. Re-mining nonce/hash is permitted; unauthorized signed
header changes are not. Exact test node IDs are in the public target matrix.

## CI and container

The existing `Container and core CI` workflow runs core tests, dry-run/dispatch
checks, installed commands and the OTS demo on each platform, then builds and
smoke-tests the CPU image:

```bash
docker build --tag wavelock-public-bounty .
docker run --rm wavelock-public-bounty wavelock-cli --help
docker run --rm wavelock-public-bounty wavelock-ots --help
```

Record the CI run and exact head SHA. Local full research runs and path-filtered
research workflows are distinct; a filtered-out job is not a passed job.
Preserve any failed invariant and report the defect. Do not silently change
reference vectors or redefine scope to turn a failure into a pass.
