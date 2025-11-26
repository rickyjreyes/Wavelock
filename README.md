# WaveLock / CurvaChain — Dev README

This repo contains a prototype curvature-locked ledger (“CurvaChain”) and helper tooling (“WaveLock”) to sign/verify curvature commitments, run a tiny P2P node, and exercise integrity checks (Merkle root, tamper detection, persistence). It’s designed for **local development** and demo scenarios.

> Status: prototype for demos & testing. Expect rough edges and evolving APIs.

---

## 0) Requirements

- **Python** 3.9–3.11
- **CUDA + CuPy** (GPU optional; CPU fallback for some paths may not exist in all files)
  - Suggested: `conda create -n cupy-env python=3.10 && conda activate cupy-env`
  - Install CuPy (per your CUDA): `conda install -c conda-forge cupy`
- **matplotlib** (for simple visualizations used by tests)
- Windows PowerShell examples below; macOS/Linux are analogous.

Optional but helpful:
- `pytest` if you want to wrap tests, though the repo ships runnable test scripts already.

---

## 1) Project layout (key files)

```text
WaveLock/
├─ chain/
│  ├─ Block.py               # Block object (index, prev_hash, merkle, messages, nonce, hash)
│  ├─ WaveLock.py            # CurvatureKeyPair, signing/verification helpers (WLv1 + WLv2/SIGv2)
│  ├─ CurvaChain.py          # Minimal chain object (append-only, PoW target optional)
│  ├─ chain_utils.py         # Load/save ledger, Merkle tools, visualization, reset helpers
│  ├─ UserRegistry.py        # Registry of users, commitments, signing convenience
│  ├─ pov.py                 # Proof-of-Verification records + VERIFICATION_TX blocks
│  ├─ artifacts.py           # Research artifact DAG (hash → parents, metadata)
│  ├─ kernel_decl.py         # KERNEL_DECL blocks binding kernel version + spec hash
│  ├─ cli.py                 # WaveLock / CurvaChain CLI (keygen, sign, mine, audit, peers…)
│  └─ __init__.py            # package marker
├─ network/
│  ├─ __init__.py
│  ├─ client.py              # Simple TCP client demo (GET_CHAIN, etc.)
│  ├─ peer_utils.py          # Peer list helpers
│  ├─ peers.json             # Local peer config (usually git-ignored)
│  ├─ protocol.py            # Socket message types + encode/decode
│  └─ server.py              # P2P server (TCP), validates and stores blocks
├─ scripts/
│  ├─ start_node.ps1         # Windows helper to start node
│  ├─ start_node.sh          # Unix helper to start node
│  ├─ start_miner.ps1        # Windows helper to start miner
│  └─ start_miner.sh         # Unix helper to start miner
├─ storage/
│  ├─ __init__.py
│  ├─ storage.py             # Disk IO helpers (append-only .jsonl, header index)
│  └─ ledger/                # On-disk blockchain (runtime files)
└─ tests/
   ├─ conftest.py
   ├─ test_artifact_dag.py
   ├─ test_curvachain_typed_blocks.py
   ├─ test_curvature_hash.py
   ├─ test_kernel_decl.py
   ├─ test_merkle.py
   ├─ test_pov.py
   ├─ test_signature_commitment_v2.py
   ├─ test_utils.py
   └─ test_wcc_rails.py

```

Runtime files:
- `ledger/blk*.jsonl` — append-only ledger blocks
- `trusted_commitments.json` — allow-list of trusted commitments
- `commitments/*.npz` — published ψ* snapshots (for full strict verification)

All of these are normally ignored via `.gitignore`.

---

## 2) Quick start

From the `WaveLock` folder:

```powershell
conda create -n cupy-env python=3.10 -y
conda activate cupy-env
python -m pip install matplotlib
```

Install CuPy appropriate for your CUDA:

```powershell
conda install -c conda-forge cupy
```

---

## 3) Core workflows

### 3.1 Generate a curvature keypair and trust it

```powershell
python -m chain.cli keygen
python -m chain.cli add ricky --n 4 --seed 42
```

This adds user `ricky` with a deterministic ψ* commitment to `users.json`.

Update trust list:

```powershell
python tools/publish_trusted.py
```

This ensures ricky’s commitment is in `trusted_commitments.json` and publishes `commitments/<hash>.npz`.

---

### 3.2 Start the server

```powershell
$env:WAVELOCK_REQUIRE_FULL_VERIFY="1"
python -m network.server --port 9001
```

Strict mode ON means blocks must:
- come from a trusted commitment (in `trusted_commitments.json`)
- AND have a published ψ* snapshot in `commitments/`.

---

### 3.3 Create and mine a block

```powershell
python -m chain.cli sign ricky --message "hello wlv2" --output signed.json
python -m chain.cli mine --signed_path signed.json
```

The mined block is broadcast to peers and saved on disk.

---

### 3.4 Verify and audit

```powershell
python -m chain.cli view        # show ledger
python -m chain.cli audit       # baseline audit (single commitment)
python tools/audit_multi_trust.py   # multi-trust audit with full ψ* verification
```

Expected: trusted commitments accepted, curvature signature valid for any block whose ψ* snapshot exists.

---

## 4) Publishing ψ* snapshots

Run:

```powershell
python tools/publish_trusted.py
```

This script:
- Reads `users.json` and `psi_keypair.json` (if present)
- Publishes `.npz` files in `commitments/`
- Updates `trusted_commitments.json`

---

## 5) Demos

### Clean ledger + single user

```powershell
python -m chain.cli reset
python -m chain.cli keygen
python -m chain.cli add ricky --n 4 --seed 42
python tools/publish_trusted.py
$env:WAVELOCK_REQUIRE_FULL_VERIFY="1"
python -m network.server --port 9001
python -m chain.cli sign ricky --message "strict test" --output signed.json
python -m chain.cli mine --signed_path signed.json
python tools/audit_multi_trust.py
```

### Multi-trust ledger

1. Mine blocks under different commitments.
2. Run `python tools/publish_trusted.py` to publish ψ* and update trust.
3. Audit with `python tools/audit_multi_trust.py` → all blocks pass.

---

After installing in editable mode you can also use the convenience entrypoints (if configured in `pyproject.toml`):

```bash
pip install -e .

# node (port 9001)
wavelockd

# add peer to it from another terminal (optional if running seeds)
wavelock-cli peer 127.0.0.1 9001

# miner
wavelock-miner mine-daemon --peer 127.0.0.1:9001 --user ricky
```

---

## 6) Security notes

- Prototype only; do not expose to untrusted networks.
- Trust is managed by static allow-list (`trusted_commitments.json`).
- Full strict verification requires publishing ψ* snapshots.
- Earlier blocks without ψ* snapshots can still be allow-listed but will show as “trusted, no published ψ*”.

---

## 7) License

Copyright © 2025 Ricky Reyes. All rights reserved.  
WaveLock / CurvaChain — research prototype.
