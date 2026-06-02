# Changelog

All notable, consensus-affecting changes to the WaveLock canonical
reference implementation are recorded here. Schema-bumping changes are
deliberate protocol upgrades; pre-upgrade and post-upgrade commitments
are not interchangeable.

## [Unreleased] — WaveLock-OTS Mythos integration-layer fixes (M1/M2/M3)

Claude Mythos found three integration-layer bounty blockers after the A/B/D
remediation. The pure `verify_ots` core held; the block acceptance/replay layer
had cheap outs. All three are now fixed (writeup:
`attacks/WAVELOCK_MYTHOS_BREAK_REPORT.md`; pins: `tests/test_ots_mythos_break.py`).
**WaveLock-OTS remains experimental and is NOT production-ready.**

- **M1 — OTS signature now binds the canonical block body.** Added
  `server.canonical_ots_block_message(block) -> bytes` and
  `canonical_ots_block_digest(block) -> str` binding `messages`, `block_type`,
  `previous_hash`, `auth_scheme`, and the carried public key (excluding the
  self-referential signature/message and mining outputs). `_verify_ots_block`
  recomputes this from the received block, requires `meta.ots_auth.message` to
  equal it, and verifies the signature against it — a benign `"hello world"`
  signature can no longer authorize an unrelated block body. New helper
  `server.build_signed_ots_block(...)` constructs correctly-bound OTS blocks.
- **M2 — replay reconstruction is independent of current config.**
  `server._reconstruct_consumed_ots` (run by `load_from_disk`) rebuilds consumed
  OTS identities from accepted chain state **by structure**, regardless of
  `cfg.require_ots`, and **fails closed** (`OTSLedgerError`) on a malformed
  OTS-claiming block. Deleting `ots_replay.jsonl` and restarting no longer
  reopens replay while the accepted chain still contains the OTS block.
- **M3 — inter-process ledger file lock.** `PersistentOTSReplayLedger.accept`
  now runs the entire read-check-append-fsync-update critical section under an
  OS-level `flock` (POSIX) and re-scans the file under the lock, so two
  instances/processes on the same file cannot both accept the same identity
  (`ots_ledger.INTERPROCESS_LOCKING` advertises availability; POSIX-only). The
  previously-separate `OTS_LEDGER` is now an **alias** for the single
  authoritative `CONSENSUS_OTS_LEDGER` (no independent bypass ledger).
- **Cross-node/global consensus enforcement remains future work.** The ledger is
  authoritative per host/filesystem, not yet replicated across nodes.
- New PoCs (now fail closed): `attacks/ots_block_body_unbound.py`,
  `attacks/ots_ledger_reconstruction_failopen.py`,
  `attacks/ots_ledger_concurrent_double_accept.py`. The inherent reuse PoC
  (`attacks/ots_reuse_to_total_forgery.py`) is unchanged and still demonstrates
  Finding C.
- Tests: `tests/test_ots_mythos_break.py` (14 hard-passing M1/M2/M3 pins);
  `tests/test_ots_consensus.py` updated to build body-bound OTS blocks.

Consensus-affecting: OTS-required blocks must now carry a signature over the
canonical block body; the prior free-text `meta.ots_auth.message` no longer
authorizes a block.

## [Unreleased] — WaveLock-OTS consensus replay protection (Finding D wired in)

OTS verification is now wired into the real block-acceptance path with a durable
replay ledger. **WaveLock-OTS remains experimental and is NOT production- or
bounty-ready** (Finding C is inherent; full Finding-D closure needs every node to
run this rejection against agreed chain state).

- **Durable replay ledger (NEW).** `wavelock/crypto/ots_ledger.py` adds
  `PersistentOTSReplayLedger`: an append-only, fsync'd JSONL ledger of consumed
  `one_time_key_id` + OTS leaf ids (`public_key_fingerprint`). Fail-closed:
  verification, parsing, hash, fingerprint, Merkle, replay, and durable-write
  errors all reject and consume nothing. The consumed set is reconstructable from
  accepted blocks (`index_signature`), so it is canonical per node.
- **Block acceptance integration (Finding D — FIXED at the ledger layer).**
  `server.try_accept_block` routes OTS-required blocks (`block_type == "OTS"`,
  `meta.auth_scheme == WaveLock-OTS-v1`, or `cfg.require_ots`) to
  `_verify_ots_block`: pure `verify_ots` + durable replay rejection. Rejects a
  replayed signature, a copied key signing a different message, and a
  cold-copy/second-host duplicate after the first acceptance.
- **Legacy SIGv2 refused on the OTS path.** An OTS-required block never falls
  back to the legacy curvature path; a non-WaveLock-OTS scheme is rejected.
- **Ledger reconstructed on load.** `ChainState.load_from_disk` folds accepted
  OTS blocks' identifiers into the ledger so it survives a lost JSONL cache.
- Tests: `tests/test_ots_consensus.py` (valid first block; replay; copied-key
  second-sign; cold-copy/second-host; malformed/missing auth; legacy SIGv2
  refused; durability across reopen; corrupt-ledger fail-closed).
- Docs: report/roadmap/README updated — A/B fixed, C inherent, D fixed only when
  global ledger/consensus replay rejection is active; host-local signing
  registry is defense-in-depth only.

Consensus-affecting: OTS-required blocks are now enforced on the acceptance path
(the legacy curvature path is unchanged for non-OTS blocks).

## [Unreleased] — WaveLock-OTS red-team remediation (A/B fixed; C/D documented)

Hardening of the experimental WaveLock-OTS construction
(`wavelock/crypto/wavelock_ots.py`) in response to the red-team audit
(`attacks/WAVELOCK_OTS_REDTEAM.md`). **WaveLock-OTS remains experimental and is
NOT production- or bounty-ready.**

- **Finding A (FIXED) — fingerprint binding.** Public keys now have an exact
  canonical field set with a `public_key_fingerprint`. `verify_ots` and
  `load_public_key` recompute the Merkle root from `pk_commitments` and
  recompute the fingerprint over the canonical public key, rejecting garbage
  roots and fingerprint/key-substitution. `params` is no longer a public-key
  field (bound via `params_hash`/fingerprint; kept in the secret key).
- **Finding B (FIXED) — strict signatures.** Signatures have an exact canonical
  field set (`revealed` → `revealed_slices`, adds `public_key_fingerprint`).
  `verify_ots` rejects missing/extra fields, wrong scheme/version/hash_alg, a
  missing or wrong `message_digest`, and binds `one_time_key_id`/fingerprint/
  `params_hash`/`psi_commitment` to the key. Versioned domain separators added.
- **Finding C (INHERENT).** Reuse → total forgery is inherent to Lamport-style
  OTS; the PoC is preserved and still succeeds on reuse.
- **Finding D (MITIGATED host-locally).** Signing atomically claims the
  `one_time_key_id` in a host-local registry (`WAVELOCK_OTS_STATE_DIR`); a
  copied key cannot sign twice on the same host. `OTSReplayLedger` models the
  server/ledger duplicate-key rejection production requires.
- **Integration.** `server.verify_ots_payload` adds a fail-closed OTS entry
  point (duplicate-key rejection; never accepts legacy SIGv2 where OTS is
  expected). OTS is **not** yet wired into block/consensus acceptance.
- Docs: new `docs/WAVELOCK_MERKLE_ROADMAP.md`; updated OTS design/report/README.

Not a consensus change (OTS is not on the consensus path).

## [Unreleased] — WLv3.1 SHAKE-256 ψ₀ derivation upgrade

### Consensus break

The canonical seed → ψ₀ derivation on the consensus path is now
SHAKE-256 (NIST FIPS 202 XOF) with the `WL-PSI-INIT-v1` domain
separation tag, replacing `numpy.random.seed(s); numpy.random.rand(...)`
(Mersenne Twister).

This matches the patent's Best Mode and Claim 9, which specify an
extendable-output function for ψ₀ derivation. The pre-upgrade Mersenne
Twister path produced backend- and library-version-bound bytes that
were not byte-stable across implementations and therefore not a valid
basis for cross-implementation consensus.

A new schema label `SCHEMA_V3_SHAKE = "WLv3.1"` distinguishes commitments
produced under the SHAKE-256 regime from legacy `WLv2` commitments.
Commitments are not interchangeable across the two regimes; the schema
prefix in the commitment string is the authoritative discriminator.

### Behavior changes

- `wavelock.chain.Wavelock_numpy.CurvatureKeyPairV3.__init__`:
  default `use_xof_init` is now `True` (was `False`). Callers that
  explicitly want the legacy Mersenne Twister path must pass
  `use_xof_init=False`. Commitments produced with `use_xof_init=True`
  carry the `WLv3.1` schema label; legacy ones retain `WLv2`.

- `wavelock.chain.WaveLock.CurvatureKeyPair.__init__`: gains a
  `use_xof_init: Optional[bool] = None` parameter. When `None`
  (default), it auto-enables whenever any of `use_v3..use_v7` is set,
  which is exactly the consensus-emitting subset of the GPU class. The
  consensus guard at the same call site continues to forbid the GPU
  class from emitting consensus commitments outside `test_mode`; the
  XOF wiring is so that test-mode consensus runs on GPU produce
  byte-identical ψ₀ to the NumPy reference path.

- `_serialize_commitment_v2(psi)` is now a thin wrapper over a new
  `_serialize_commitment(psi, schema)` that writes the schema label
  into the binary header. The legacy V2 wrapper is preserved so
  external callers that imported it directly continue to work.

### Required follow-up

- Golden vectors in `tests/general/test_golden_vectors.py` have been
  reset to `PLACEHOLDER_GENERATE_ME` so the parametrized regression
  tests skip instead of asserting against stale Mersenne-Twister
  hashes. Repopulate with:

      python tests/general/test_golden_vectors.py generate

  and paste the printed dict over the placeholder one. Every entry
  will then carry `"schema": "WLv3.1"` — that's the loud signal that
  the canonical commitment format has flipped.

- Any persisted ledger or registry state produced before this change
  remains parseable but will fail re-derivation from seed under the
  new default. Such state is historical-format-only and must be
  re-issued through the new canonical path before being treated as
  consensus-binding.

### Patent-enablement effect

- Claim 9 / Best Mode (SHAKE-256 ψ₀ derivation) is now exercised on
  the actual canonical commitment path, not as an opt-in side branch.

- Claim 8 Markush group (SHA-256, SHA3-256, BLAKE3 as disjoint hash
  families) is now exercised end-to-end for BLAKE3 via two new tests
  in `tests/test_blake3_strict.py`:
  `test_blake3_end_to_end_through_keypair_commitment` and
  `test_blake3_dual_signature_through_keypair`. Both construct a
  `CurvatureKeyPairV3` with `secondary_family=HashFamily.BLAKE3`,
  generate a commitment, and round-trip it through `verify_commitment`
  and `verify_strict`.
