# WaveLock-OTS — Mythos Integration-Layer Break Report (M1 / M2 / M3)

> **Status: M1, M2, M3 FIXED.** Three integration-layer bounty blockers found by
> Claude Mythos after the A/B/D remediation are now closed, each with a PoC that
> previously demonstrated the break and now fails closed, and a hard-passing
> secure-behavior pin in `tests/test_ots_mythos_break.py`. The pure cryptographic
> core (`verify_ots`) was never the problem; the **block acceptance / replay
> layer** was. Cross-node / global consensus enforcement remains future work.
> WaveLock-OTS is still experimental and **not** production-ready.

## Scope

Mythos confirmed the pure stateless verifier held up under A/B/D hardening, then
attacked the layer that turns a valid signature into an *accepted block* and an
identity into a *consumed-once* fact. Three cheap outs existed there.

| ID | Severity | Title | Status |
|----|----------|-------|--------|
| M1 | High (bounty-blocking) | OTS block signature does not bind the block body | **FIXED** |
| M2 | High (bounty-blocking) | Durable replay control fails open after ledger delete + restart | **FIXED** |
| M3 | Medium | No inter-instance/file lock on `PersistentOTSReplayLedger` | **FIXED** |

---

## M1 — OTS block signature did not bind the block body

**Break.** `_verify_ots_block` verified the OTS signature against the arbitrary
string `meta.ots_auth.message`, but the actually-accepted payload is
`Block.messages` (plus `block_type`, parent ref, etc.). Those were never tied
together. An attacker holding *any* public, valid OTS signature over a benign
message (e.g. `"hello world"`) could wrap it into a block whose body said
something else and have it accepted — the signature "authorized" content nobody
agreed to.

**PoC.** `attacks/ots_block_body_unbound.py`
- `benign_signature_authorizes_malicious_body()` — splices a `"hello world"`
  signature into a `["transfer 1000000 to attacker"]` block.
- `body_tampering_after_signing()` — signs a block correctly, then rewrites
  `Block.messages`.
Both previously returned `True` (accepted); now return `False` (rejected).

**Fix.** A canonical block-signing transcript is defined and recomputed from the
*received* block:
- `server.canonical_ots_block_message(block) -> bytes` — the canonical preimage.
- `server.canonical_ots_block_digest(block) -> str` — its domain-separated hex
  digest; this is the message an OTS signer actually signs.

The transcript binds: `messages` (body payload), `block_type`, `previous_hash`
(parent ref), `auth_scheme`, the carried **public key**, and any extra `meta`
fields. It excludes the self-referential `ots_auth.signature` / `ots_auth.message`
and the mining outputs (`nonce`/`hash`) and `timestamp`/`index` (not signed — a
node may legitimately reposition an identical body; the replay ledger, not the
index, prevents reuse). `previous_hash` is bound, so a body cannot be replayed
under a different parent.

`_verify_ots_block` now (a) recomputes `expected = canonical_ots_block_digest(b)`,
(b) requires `meta.ots_auth.message == expected` (constant-time), and (c) verifies
the signature against `expected`, not against attacker-controlled text. Correct
blocks are built with `server.build_signed_ots_block(...)`, which signs the
transcript digest.

**Pin.** `tests/test_ots_mythos_break.py::test_m1_*` (benign signature rejected,
body tampering rejected, transcript changes with messages / block_type /
previous_hash, PoC blocked).

---

## M2 — Durable replay control failed open after ledger delete + restart

**Break.** Acceptance used `block_requires_ots(b, cfg)`, but reconstruction in
`load_from_disk` called `block_requires_ots(b)` **without** cfg and only folded
in signatures for blocks detected as OTS by `block_type` / `meta.auth_scheme`.
Worse, the consumed set lived only in the side cache `ots_replay.jsonl` plus
memory (`index_signature` is memory-only). Deleting `ots_replay.jsonl` and
restarting produced an **empty** consumed set, so a copied key could replay an
already-consumed OTS identity. Blocks accepted only because `cfg.require_ots` was
set could also be missed during reconstruction.

**PoC.** `attacks/ots_ledger_reconstruction_failopen.py`
- `replay_after_side_ledger_delete()` — accept an OTS block, delete the side
  ledger, restart, replay. Previously `True` (accepted); now `False`.
- `reconstruction_indexes_cfg_only_block()` — an accepted block carrying OTS auth
  but with `GENERIC` type and no `auth_scheme` is now indexed by structure
  (`True`).

**Fix.** `server._reconstruct_consumed_ots(blocks, ledger)` (run by
`ChainState.load_from_disk`) rebuilds the consumed set from **accepted chain
state by structure**: every accepted block carrying a well-formed WaveLock-OTS
signature is folded in via `index_signature`, **independent of the current
config**. If a block *structurally* claims OTS (`block_type == "OTS"` or
`meta.auth_scheme == WaveLock-OTS-v1`) but carries malformed/missing auth, it
raises `OTSLedgerError` (**fail closed**) instead of being silently skipped.
Deleting the side ledger no longer resurrects consumed identities while the
accepted chain still contains the block that consumed them.

**Pin.** `tests/test_ots_mythos_break.py::test_m2_*` (replay rejected after delete
+ `load_from_disk` restart; reconstruction indexes by structure independent of
cfg; malformed OTS auth fails closed; PoC blocked).

---

## M3 — No inter-instance/file lock on the replay ledger

**Break.** The accept critical section used only a per-instance
`threading.RLock`. Two `PersistentOTSReplayLedger` instances (or two processes)
sharing the same ledger file each kept their own in-memory consumed set loaded
once at construction; their read-check-append-fsync-update sections could
interleave, so **both could accept the same OTS identity** — one-time becomes
two-time.

**PoC.** `attacks/ots_ledger_concurrent_double_accept.py`
- `concurrent_accept_count()` — N instances race on one identity; previously > 1,
  now exactly `1`.
- `two_instances_double_accept()` — previously `True`, now `False`.

**Fix.** The entire read-check-append-fsync-update critical section now runs under
an OS-level `flock` (POSIX, `fcntl.flock` on a sibling `<path>.lock`) in addition
to the per-instance `RLock`. Under the lock the ledger **re-scans the file**
(`_scan_file_into_sets`, additive so chain-reconstructed ids survive) so a
concurrent append by another instance/process is seen before the duplicate check.
At most one of N competing instances accepts a given identity.
`ots_ledger.INTERPROCESS_LOCKING` advertises availability; the inter-process
pins are POSIX-only (skipped where `fcntl` is absent — documented).

**Ledger unification.** There used to be two independent ledgers:
`CONSENSUS_OTS_LEDGER` (durable, used by block acceptance) and `OTS_LEDGER` (an
in-memory `OTSReplayLedger` used by the standalone `verify_ots_payload`). Two
ledgers that each "consume" independently is a bypass risk. `OTS_LEDGER` is now
an **alias** for the single authoritative `CONSENSUS_OTS_LEDGER`, and
`verify_ots_payload` defaults to it, so a signature consumed on either path is
consumed on both. The pure in-memory `OTSReplayLedger` class is still importable
for callers that explicitly want an ephemeral ledger (passed via `ledger=`).

**Pin.** `tests/test_ots_mythos_break.py::test_m3_*` and
`test_single_authoritative_ledger_no_bypass`.

---

## Preserved prior security work

Unchanged and still enforced (no weakening):
strict canonical public-key checking, strict canonical signature checking,
recomputed Merkle root checks, recomputed public-key fingerprint checks,
`message_digest` binding, fail-closed behavior, legacy SIGv2 rejection on the
OTS-required path, and the pure/stateless `verify_ots`. All prior PoCs and
reports are kept. The **inherent OTS reuse PoC**
(`attacks/ots_reuse_to_total_forgery.py`) still demonstrates that key reuse
leaks both halves and yields total forgery — that is inherent to Lamport-style
OTS (Finding C); the defense is the acceptance-layer replay rejection, not the
pure crypto.

## Remaining caveats / future work

- **Cross-node / global consensus enforcement is NOT implemented.** The replay
  ledger is authoritative *per host/filesystem* (now correctly so across
  instances/processes). Full closure requires every accepting node to run this
  rejection against a ledger derived from agreed chain state.
- File locking gives correct exclusion on the **same** host/filesystem, not
  across hosts or network filesystems with weak `flock` semantics.
- WaveLock-OTS is still single-time Lamport (Finding C inherent), unproven, and
  unreviewed. **Do not use for production funds.**

## Bounty readiness

With M1/M2/M3 fixed, the system is materially closer to a **scoped** bounty
focused on canonical verification, replay protection, and OTS block acceptance.
It is **not** ready for an open-scope or value-transfer bounty: cross-node
consensus replication and the multi-signature (WaveLock-Merkle) tree remain
future work, and WaveLock-OTS remains experimental.
