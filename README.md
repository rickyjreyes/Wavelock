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
  Historical CLI tools are isolated under `wavelock-cli legacy`. The normal
  `keygen`, `sign`, `mine`, and `verify` commands implement WaveLock-OTS.

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
  critical section is **inter-process locked (`flock` on POSIX; SQLite on Windows)** with a single authoritative
  ledger (M3). Cross-node/global consensus enforcement remains future work.
  [Public Bounty v1](audit/PUBLIC_BOUNTY_V1.md) defines the offered commitment,
  replay and authenticated-record attack surfaces.
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

## Install and run

Python 3.9+ is supported. The reference workflow uses NumPy and requires no GPU.

```bash
python -m pip install -e ".[blake3]" pytest
python hello_wavelock.py
```

The demo generates a fresh OTS key, signs a canonical block body, mines and
accepts it, deletes the secret key, verifies in a new process using public
material, and checks replay rejection. It uses disposable state and leaves
existing ledgers alone.

## Supported block workflow

These normal commands now use **WaveLock-OTS**. A signature authenticates the
canonical block body and its parent. Mining uses that existing signature;
it does not sign a second message with the same key.

```bash
wavelock-cli --data-dir demo-node keygen --out keys/demo-1
wavelock-cli --data-dir demo-node sign --secret keys/demo-1/wl_ots_secret.json --message "research artifact abc" --output signed-block.json
wavelock-cli --data-dir demo-node verify --signed-path signed-block.json
wavelock-cli --data-dir demo-node mine --signed-path signed-block.json
wavelock-cli --data-dir demo-node verify
```

`add ricky` is a convenience for generating a fresh pair in `keys/ricky/`;
`sign ricky --message "..."` uses that pair. Labels are local aliases, not
long-lived authenticated identities. Each pair signs once. Generate another
pair in a new directory for the next block. Existing key files are never
silently overwritten. A stale signed parent requires a fresh key and signature.

The signed artifact contains the public key and selected OTS slices only.
Verification never loads a secret key, integer seed, registry, or psi snapshot.
The standalone `wavelock-ots` commands remain available for detached messages;
those detached signatures are not block authorizations.

## Node configuration and persistence

CLI and node use the same `WAVELOCK_DATA_DIR` (default: the existing user data
location, `$XDG_DATA_HOME/wavelock` or `~/.wavelock`). The CLI also accepts
`--data-dir` before its subcommand. Set `WAVELOCK_DATA_DIR` for the node.
Accepted blocks and the reconstructable OTS replay cache live under `ledger/`.
Signer-use markers live under `ots-state/`. Keep backups of signing state;
never reuse an OTS key across hosts or restored copies.

CLI mining and node acceptance share a same-host writer lock and reload the
tip before appending. A draft signed against an old parent is rejected: use a
fresh key to sign for the new tip. This coordinates local writers; it does not
provide distributed consensus or authenticated recovery from a full rollback.

[Public Bounty v1](audit/PUBLIC_BOUNTY_V1.md) covers Critical 1–5, High 1–5 and
implemented OTS/authenticated-record checks in the public reference implementation.
See its [reproduction guide](audit/PUBLIC_BOUNTY_REPRODUCTION_GUIDE.md) and the
unchanged [normative profile](docs/BOUNTY_SECURITY_PROFILE.md). Remote attestation,
behavior-wide drift and broad hostile-host rollback guarantees are not offered
properties of this public bounty.

```bash
wavelockd --port 9001
```

The node defaults to `require_ots=true`. It verifies stored OTS history on
startup and rejects legacy SIGv2 on the normal acceptance path. Configure via
`WAVELOCK_CONFIG` or `wavelockd --config node.json`:

```json
{"port": 9001, "require_ots": true}
```

`WAVELOCK_REQUIRE_OTS=1` also enables this policy. Explicit `false` / `0` is
reserved for historical compatibility experiments. Unknown JSON configuration
keys are errors. File settings override environment defaults.

New OTS ledgers do not silently import or rewrite an old SIGv2 ledger. Historical
package-local data can be inspected with the tools described in
[the migration guide](docs/MIGRATION_FROM_SIGV2.md).

Replay acceptance is serialized across processes on one filesystem (POSIX
flock or SQLite locking), and tip validation plus append are serialized within
one node. This remains a single-node prototype: block and replay files are not
a distributed transactional store; crash gaps can conservatively consume a key
without a completed block. Full cross-node consensus and Merkle many-key
signing remain on the [roadmap](docs/WAVELOCK_MERKLE_ROADMAP.md).

## Verification

```bash
python -m pytest tests/ -m "not slow" -q
python -m pytest curvature_audit/ -c curvature_audit/pytest.ini -m "not slow" -q
python -m pytest pde_audit/ -c pde_audit/pytest.ini -m "not slow" -q
```

`tests/test_supported_workflow.py` exercises the real CLI across fresh processes,
public-only verification, key reuse, malformed artifacts, startup replay
reconstruction, configuration, and concurrent acceptance.

## Research and historical tools

- `wavelock/curvature_capacity_v1/`: current CC-Core-v1-B research candidate.
- `wavelock/curvature_capacity/`: frozen Candidate A baseline.
- `wavelock/pde_hash/`: historical hash-free PDE core and regression target.
- `wavelock/crypto/`: OTS, replay protection and encryption wrappers.
- `wavelock/chain/ots_blocks.py`: shared OTS transcripts and pure public checks.
- `wavelock-cli legacy ...`: historical SIGv2 tools, retained for reproduction.

Historical SIGv2 findings remain documented and reproducible.

## License

Copyright © 2025 Ricky Reyes. All rights reserved.
See `LICENSE` and `PATENT_NOTICE.md`.
