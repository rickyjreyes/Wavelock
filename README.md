# WaveLock / CurvaChain — Dev README

This repo contains a prototype curvature-locked ledger (“CurvaChain”) and helper tooling (“WaveLock”) to sign/verify curvature commitments, run a tiny P2P node, and exercise integrity checks (Merkle root, tamper detection, persistence). It’s designed for **local development** and demo scenarios.

> Status: prototype for demos & testing. Expect rough edges and evolving APIs.

---


## Current research architecture

WaveLock now separates three distinct layers:

1. **CC-Core-v1-B — curvature/path commitment.** This is the current primary
   experimental research core. It co-evolves an accumulator with the wavefield so
   the commitment depends on the ordered trajectory rather than only on the
   terminal state. Candidate B uses the linear injection
   `j_B(u,v) = u(1 + γv) mod p`, removes Candidate A's generic 2-to-1 injection
   weakness, and separates all 47 known Phase 8J terminal-collapse states
   (minimum pairwise Hamming distance 105/256).
2. **WaveLock-OTS — asymmetric one-time signatures.** This is the current
   experimental signing layer. It provides public verification without revealing
   `ψ★`, but every one-time key must be used exactly once.
3. **CurvaChain — ledger and replay enforcement.** This binds canonical block
   bodies, OTS identities, commitments, and accepted-chain replay state.

`CC-Core-v1-B` is the best current realization of the original “pebble leaves a
wake” idea, but it is **not a proven cryptographic primitive**. The Phase CC-3
audit found:

- all 47 known terminal-state collapse cases remain distinct under the path
  commitment;
- the singular value `v_star = -γ^-1 mod p = 195225786` is unreachable at round
  0 under the normative message protocol;
- a replay-verified 191-byte message reaches `v_star` at one coordinate at round
  1, but the tested singular event did not erase the path commitment or produce
  a structural collision;
- 205 fast tests pass and GitHub Actions completed successfully on the research
  branch;
- multi-coordinate singular reachability, general binding, collision hardness,
  second-preimage hardness, and any Layer-3 lower bound remain unresolved.

The curvature-capacity research code is isolated in:

```text
wavelock/curvature_capacity/       # frozen CC-Core-v0-A baseline
wavelock/curvature_capacity_v1/    # current CC-Core-v1-B candidate
curvature_audit/                   # adversarial tests, artifacts, and reports
```

Primary documents:

- `docs/CC_CORE_V1_SPEC.md`
- `docs/CC_CORE_V1_NORMATIVE_PROTOCOL.md`
- `docs/CC_CORE_V1_VSTAR_REACHABILITY.md`
- `docs/WAVELOCK_CURVATURE_CAPACITY_RESULTS.md`

**Use classification**

| Component | Current status | Intended use |
|---|---|---|
| `CC-Core-v1-B` | Experimental research core | Path/trajectory commitment studies |
| WaveLock-OTS | Experimental asymmetric OTS | One-time signing tests and scoped demos |
| CurvaChain | Prototype ledger | Local replay, persistence, and consensus experiments |
| WLv2 / SIGv2 | Deprecated and insecure | Compatibility/testing only |
| Ed25519 / SLH-DSA / LMS / XMSS | Established alternatives | Production security |

Do not describe any WaveLock component as “provably secure,” “collision-resistant,”
“one-way,” “256-bit secure,” or as forcing full sequential execution. No such
general theorem has been proved.

---

## ⚠️ Security notice — read this first

- **Legacy WaveLock SIGv2 (`WLv2`) is DEPRECATED and INSECURE.** It signs with
  `H("SIGv2" ‖ message ‖ header ‖ ψ★)` and verifies by recomputing the same
  hash, so **the verifier must possess ψ★ — and anyone who can verify can
  forge.** It is a symmetric MAC, not an asymmetric signature. See
  `attacks/WAVELOCK_THEORY_BREAK_AUDIT.md` and `docs/MIGRATION_FROM_SIGV2.md`.
  The legacy CLI (`wavelock-cli keygen` / `sign`) now prints a deprecation
  warning.

- **WaveLock-OTS is the new, experimental asymmetric construction.** It is a
  Lamport/WOTS-style one-time signature with WaveLock ψ-state binding
  underneath (`wavelock/crypto/wavelock_ots.py`, CLI `wavelock-ots`). Key
  guarantees:
  - **Public verification never requires ψ★.** The public key is only
    commitments, hashes, a Merkle root, parameters, and metadata.
  - **The verifier cannot forge**, because it only ever sees the
    *message-selected* secret slices (one of two per digest bit). The
    unrevealed halves stay secret.
  - **Strict, fail-closed verification.** The public key has an exact canonical
    field set; `verify_ots`/`load_public_key` recompute the Merkle root from
    `pk_commitments` and recompute a `public_key_fingerprint`, and signatures
    have an exact canonical field set bound to that fingerprint (no malleability,
    no key substitution). See `docs/WAVELOCK_OTS_DESIGN.md` §6a.
  - **Keys are one-time, enforced at block acceptance.** Reuse is rejected by
    default; signing also claims a host-local atomic key-state registry
    (defense-in-depth only). The load-bearing control is a **durable replay
    ledger** (`wavelock/crypto/ots_ledger.py`) wired into block acceptance: a
    reused `one_time_key_id`/leaf is rejected when a block is accepted, so a
    *copied* key cannot get a second OTS block accepted on a node.
  - Seeds are ≥128-bit (default 256-bit); there are no tiny integer seeds, and
    no ψ★/seed is exported in any public artifact.

- **WaveLock-OTS is NOT yet a formal cryptographic standard** and has no
  security proof. **Do not use it for production funds.** For production, use
  **Ed25519, SLH-DSA, LMS, or XMSS**. See `docs/WAVELOCK_OTS_DESIGN.md` for the
  threat model and known limitations.

- **Known, documented limits (red-team status).** Red-team **A/B are fixed**
  (canonical-field / Merkle / fingerprint binding, fail-closed verification).
  **C is inherent**: reuse → total forgery is intrinsic to Lamport-style OTS
  (never reuse a key) — the PoC is preserved as a regression test. **D is fixed
  at the ledger/consensus layer**: OTS verification + a durable replay ledger are
  now wired into block acceptance (`server.try_accept_block`), which rejects a
  reused `one_time_key_id`/leaf and never accepts legacy SIGv2 where OTS is
  required — but only *fully* closed once every accepting node runs this
  rejection against agreed chain state; the host-local registry is
  **defense-in-depth only**. WaveLock-OTS is still experimental and **not
  production-ready**. See `attacks/WAVELOCK_OTS_REDTEAM.md` and
  `docs/WAVELOCK_MERKLE_ROADMAP.md`.

- **Mythos integration-layer fixes (M1/M2/M3).** A later red-team pass closed
  three block acceptance/replay-layer blockers: OTS block signatures now bind the
  **canonical block body** (M1, no more free-text `meta.ots_auth.message`);
  consumed OTS identities are **reconstructed from accepted chain state
  independent of current config** and fail closed on malformed auth (M2, deleting
  `ots_replay.jsonl` no longer reopens replay); and the replay ledger's accept
  critical section is **inter-process `flock`-locked** with a single authoritative
  ledger (M3). Cross-node/global consensus enforcement remains future work. With
  M1/M2/M3 fixed the system may be ready for a **scoped** bounty (canonical
  verification, replay protection, OTS block acceptance) — not value transfer.
  See `attacks/WAVELOCK_MYTHOS_BREAK_REPORT.md` and
  `tests/test_ots_mythos_break.py`.

### WaveLock-OTS quick start

```bash
wavelock-ots ots-keygen  --out keys/
wavelock-ots ots-sign    --secret keys/wl_ots_secret.json --message "pay alice 5" --sig sig.json
wavelock-ots ots-verify  --public keys/wl_ots_public.json --message "pay alice 5" --sig sig.json
wavelock-ots ots-inspect --public keys/wl_ots_public.json
```

(Each key signs **once**. Generate a fresh key per message.)

### WaveLock-Encrypt quick start (experimental)

`WaveLock-Encrypt v1` (`wavelock/crypto/wavelock_encrypt.py`, CLI
`wavelock-encrypt`) is an **experimental** hybrid public-key encryption wrapper.
It is **not a new raw cipher** — confidentiality and integrity come entirely
from X25519 (ephemeral-static), HKDF-SHA256, and ChaCha20-Poly1305. The
WaveLock contribution is **canonical transcript/context binding**: decryption
fails closed if the authenticated context (purpose, ψ-commitment, block digest,
OTS fingerprint, …) changes. See
[`docs/WAVELOCK_ENCRYPT_SECURITY_NOTE.md`](docs/WAVELOCK_ENCRYPT_SECURITY_NOTE.md).
**Not production audited.**

```bash
wavelock-encrypt keygen  --private wlenc_private.pem --public wlenc_public.pem
wavelock-encrypt encrypt --public wlenc_public.pem --input msg.bin --output env.json \
    --purpose "transport/demo" --psi-commitment <hex>
wavelock-encrypt decrypt --private wlenc_private.pem --input env.json --output out.bin \
    --purpose "transport/demo" --psi-commitment <hex>
```

---

## 0) Requirements

- **Python** 3.9+
- **numpy** (required)
- **matplotlib** (for visualizations and attack battery plots)
- **pytest** (for running test suite)

Optional:
- **CuPy** (GPU acceleration; falls back to NumPy automatically if unavailable)
  - Install CuPy per your CUDA version: `pip install cupy-cuda12x`

```bash
pip install numpy matplotlib pytest
# Or install the package:
pip install -e .
```

---

## 1) Project layout (key files)

```text
WaveLock/
├─ wavelock/
│  ├─ curvature_capacity/       # frozen CC-Core-v0-A research baseline
│  ├─ curvature_capacity_v1/    # current CC-Core-v1-B path-commitment candidate
│  └─ crypto/                   # OTS, replay ledger, and encryption wrappers
├─ curvature_audit/             # adversarial audit suite and machine-readable artifacts
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

```bash
pip install numpy matplotlib pytest
# Or install as editable package:
pip install -e .
```

Run the demo:

```bash
python hello_wavelock.py
```

Run the ledger/crypto test suite:

```bash
python -m pytest tests/ -v
```

Run the fast curvature-capacity audit suite:

```bash
python -m pytest curvature_audit/ -c curvature_audit/pytest.ini -m "not slow" -q
```

---

## 3) Core workflows

### 3.1 Generate a curvature keypair and trust it

```powershell
python -m wavelock.chain.cli keygen
python -m wavelock.chain.cli add ricky --n 4 --seed 42
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
python -m wavelock.network.server --port 9001
```

Strict mode ON means blocks must:
- come from a trusted commitment (in `trusted_commitments.json`)
- AND have a published ψ* snapshot in `commitments/`.

---

### 3.3 Create and mine a block

```powershell
python -m wavelock.chain.cli sign ricky --message "hello wlv2" --output signed.json
python -m wavelock.chain.cli mine --signed_path signed.json
```

The mined block is broadcast to peers and saved on disk.

---

### 3.4 Verify and audit

```powershell
python -m wavelock.chain.cli view        # show ledger
python -m wavelock.chain.cli audit       # baseline audit (single commitment)
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
python -m wavelock.chain.cli reset
python -m wavelock.chain.cli keygen
python -m wavelock.chain.cli add ricky --n 4 --seed 42
python tools/publish_trusted.py
$env:WAVELOCK_REQUIRE_FULL_VERIFY="1"
python -m wavelock.network.server --port 9001
python -m wavelock.chain.cli sign ricky --message "strict test" --output signed.json
python -m wavelock.chain.cli mine --signed_path signed.json
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
