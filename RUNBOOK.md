# WaveLock supported workflow runbook

Install from the repository root:

```bash
python -m pip install -e ".[blake3]" pytest
python hello_wavelock.py
```

The demo uses temporary state and validates a complete OTS workflow.

## Create and accept an OTS block

```bash
wavelock-cli --data-dir demo-node keygen --out keys/block-1
wavelock-cli --data-dir demo-node sign --secret keys/block-1/wl_ots_secret.json --message "hello wavelock" --output signed-block-1.json
wavelock-cli --data-dir demo-node verify --signed-path signed-block-1.json
wavelock-cli --data-dir demo-node mine --signed-path signed-block-1.json
wavelock-cli --data-dir demo-node verify
wavelock-cli --data-dir demo-node view
```

Signing creates an authenticated block draft, consuming the OTS key once.
Mining changes only mining fields and accepts through the node's OTS/replay
checks. `verify --signed-path` checks authentication without requiring mining;
`verify` checks the accepted chain, including proof of work and key uniqueness.
Verification needs no secret files. Commands return nonzero on failure.

## Next block

Generate a new keypair in `keys/block-2` and choose a new signed output filename.
Never copy a consumed key or clear its use marker to sign again. If the chain
tip changes after signing, the signed parent is stale: use a fresh key and sign
again. Existing key files and signed-block outputs are not overwritten.

## Run a node using that same ledger

CLI mining and node acceptance coordinate through a same-host writer lock and
reload the current tip before acceptance. A competing append makes an old
signed parent stale; discard that draft and use a fresh key. Direct historical
file writers are outside this lock; do not mix them with the supported workflow.

Bash:

```bash
export WAVELOCK_DATA_DIR="$PWD/demo-node"
wavelockd --port 9001
```

PowerShell:

```powershell
$env:WAVELOCK_DATA_DIR = (Join-Path $PWD "demo-node")
wavelockd --port 9001
```

OTS is required by default. Optional JSON configuration:

```json
{"port":9001,"require_ots":true}
```

Pass it with `wavelockd --config node.json`; set `WAVELOCK_CONFIG` to use the same
file with CLI mining. Unrecognized configuration fields cause an error.

## Test and research commands

```bash
python -m pytest tests/ -m "not slow" -q
python -m pytest curvature_audit/ -c curvature_audit/pytest.ini -m "not slow" -q
python -m pytest pde_audit/ -c pde_audit/pytest.ini -m "not slow" -q
python audit/run_bounty_contract.py --check
```

Historical signing and diagnostics are under `wavelock-cli legacy ...`.
See `docs/MIGRATION_FROM_SIGV2.md` before using historical data. WaveLock-OTS
remains experimental; repeated signing under a stable Merkle root and full
cross-node replay consensus are separate unfinished features.

For canonical commitments, input requirements and persistence limits see
[BOUNTY_SECURITY_PROFILE.md](docs/BOUNTY_SECURITY_PROFILE.md). Keep signing
markers and the replay ledger when restoring state: a crash may burn a key
before its block is appended. Verification without an independently trusted
tip does not establish that a local chain is the latest complete history.
