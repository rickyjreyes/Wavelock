# WaveLock-OTS — Final Report

Short summary of what changed, what attacks are now blocked, and what remains
experimental. Full design: `docs/WAVELOCK_OTS_DESIGN.md`. Migration:
`docs/MIGRATION_FROM_SIGV2.md`. Red-team audit + status:
`attacks/WAVELOCK_OTS_REDTEAM.md`.

## Red-team remediation status (A/B/C/D + integration)

- **A — FIXED.** `verify_ots` and `load_public_key` recompute the Merkle root
  from `pk_commitments` and recompute the canonical `public_key_fingerprint`;
  both must match. Garbage roots and fingerprint/key-substitution are rejected.
- **B — FIXED.** Strict canonical signature verification: exact field set (no
  missing/extra), enforced `scheme`/`version`/`hash_alg`, present+correct
  `message_digest`, and `one_time_key_id`/`public_key_fingerprint`/`params_hash`/
  `psi_commitment` bound to the key.
- **C — INHERENT, NOT FIXED.** Reuse → total forgery is inherent to
  Lamport-style OTS. The PoC is preserved and still succeeds on reuse. The fix
  is operational (never reuse) + ledger duplicate-key rejection, not code.
- **D — FIXED at the ledger/consensus layer (with the honest caveat below).**
  OTS verification is now wired into the real block-acceptance path
  (`server.try_accept_block` → `server._verify_ots_block`) backed by a durable,
  reconstructable replay ledger (`wavelock/crypto/ots_ledger.py`,
  `PersistentOTSReplayLedger`). Block acceptance rejects any OTS signature whose
  `one_time_key_id` or OTS leaf id (`public_key_fingerprint`) was already
  accepted — so a replayed signature, a copied key signing a *different* message,
  and a cold-copy/second-host duplicate are all rejected after the first
  acceptance. The consumed set is a function of accepted chain state
  (`index_signature` rebuilds it from accepted blocks), so it is canonical per
  node. **Caveat:** D is only *fully* closed when every accepting node runs this
  rejection against a ledger derived from agreed chain state; the host-local
  signing registry (`_claim_one_time_key`) remains **defense-in-depth only**.
- **Integration — DONE (single-node consensus path).** Block acceptance routes
  OTS-required blocks (`block_type == "OTS"`, or `meta.auth_scheme ==
  WaveLock-OTS-v1`, or `cfg.require_ots`) to fail-closed OTS verification +
  durable replay rejection, and **never** falls back to legacy SIGv2 on that
  path. Legacy curvature blocks still use the legacy fail-closed path.

**WaveLock-OTS is experimental and is NOT production- or bounty-ready.** The
ledger integration closes the host-copy replay gap on a node, but the scheme is
still single-time Lamport (Finding C is inherent), unproven, and unreviewed.

## What changed

- **New asymmetric construction: WaveLock-OTS** (`wavelock/crypto/wavelock_ots.py`).
  A Lamport/WOTS-style one-time signature whose secret slices are derived
  through the WaveLock ψ pipeline (seed → ψ₀ → ψ★ → `psi_commitment` → `sk`).
  Public verification uses only public commitments + the message-selected
  revealed slices — **ψ★ is never required to verify**.
  - APIs: `generate_ots_keypair`, `sign_ots`, `verify_ots`,
    `export_public_key`, `export_secret_key`, `load_public_key`,
    `load_secret_key`.
  - Hash: SHAKE256-256 with length-prefixed domain separation throughout.
  - Seeds: `os.urandom`, ≥128-bit floor, 256-bit default; tiny/constant seeds
    rejected on generate and on load.
  - One-time enforcement: `used` flag; second signature raises
    `OTSKeyReuseError` (CLI exit code 2) unless `--unsafe-allow-reuse`.
  - No ψ★/ψ₀/seed/raw-slice in any public artifact (asserted on export/load).
    Optional encrypted-at-rest seed; ψ★ only via loud
    `--unsafe-export-secret-state`.

- **CLI** (`wavelock-ots`, also `wavelock-cli ots-*`): `ots-keygen`,
  `ots-sign`, `ots-verify`, `ots-inspect`, `ots-mark-used`.

- **Legacy SIGv2 deprecated**: `wavelock-cli keygen`/`sign` print a loud
  deprecation/insecurity warning; README and migration doc mark it insecure.

- **Server fail-closed** (`wavelock/network/server.py`): removed the
  non-strict / trust-only acceptance path and the "allow unpublished ψ" branch.
  A valid signature is always required; trust-list membership alone never
  passes; any verification error rejects.

- **OTS wired into block acceptance** (`wavelock/network/server.py` +
  `wavelock/crypto/ots_ledger.py`): `try_accept_block` routes OTS-required
  blocks to `_verify_ots_block`, which verifies with the pure `verify_ots` and
  consumes/rejects via the durable `PersistentOTSReplayLedger`. Legacy SIGv2 is
  refused on the OTS path. Replay tests in `tests/test_ots_consensus.py`.

- **Attacks preserved as regression evidence** (`attacks/`):
  `forge_from_snapshot.py`, `seed_bruteforce.py`,
  `WAVELOCK_THEORY_BREAK_AUDIT.md`.

- **Tests** (all passing): `tests/test_ots_security.py`,
  `tests/test_ots_roundtrip.py`, `tests/test_legacy_sigv2_broken.py`,
  `tests/test_server_verification.py`.

## What attacks are blocked

| Attack | Legacy SIGv2 | WaveLock-OTS |
|--------|--------------|--------------|
| Forge from a ψ★ snapshot (`forge_from_snapshot.py`) | **succeeds** (by design) | **blocked** — public key has no `sk`; forging needs a 256-bit preimage |
| Verifier-can-forge (capability sets identical) | **yes** | **no** — verifier only sees message-selected halves |
| Small integer seed brute-force (`seed_bruteforce.py`) | **succeeds** (~7-bit seeds) | **blocked** — 256-bit seeds, sub-128-bit rejected |
| Cleartext seed/ψ★ in public artifact | **yes** | **no** — export/load reject forbidden fields |
| Server trust-only / fail-open acceptance | **yes** | **removed** — fail-closed |
| One-time key reuse | unprotected | **detected and rejected** |
| Replay / copied-key second-sign at block acceptance | n/a | **rejected** — durable replay ledger refuses a consumed `one_time_key_id`/leaf |
| Legacy SIGv2 accepted where OTS is required | n/a | **rejected** — no fallback on the OTS path |

Each row is pinned by a test; the legacy rows assert the break still works
(documenting it), the OTS rows assert it fails.

## What remains experimental / unproven

- No formal security proof; we rely on the standard one-time Lamport argument
  plus SHAKE256 one-wayness.
- The ψ-binding currently adds *structure and key/kernel binding*, not proven
  additional hardness — the "curvature hardness" thesis is unproven.
- One-time only (no many-signature mode yet).
- Side channels and quantization choices not formally analyzed.

**Do not use WaveLock-OTS for production funds.** Use Ed25519, SLH-DSA, LMS, or
XMSS for production until WaveLock-OTS is independently reviewed.

## Roadmap

WaveLock-Merkle (many signatures) → WaveLock-ZK (commitment-only ψ-evolution
proof) → formal EUF-CMA proof → post-quantum parameter review.
