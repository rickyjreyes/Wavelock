# WaveLock-OTS — Claude-Mythos Adversarial Break Report

**Target:** WaveLock-OTS after the A/B/C/D red-team remediation
**Branch:** `claude/gifted-volta-Ou5JB`
**Date:** 2026-06-01
**Posture:** ruthless cryptographic red-team; no source modified before verification; PoCs and regression tests added under `attacks/` and `tests/`.

---

## TL;DR

The A/B hardening of the **pure** verifier (`verify_ots`) holds. I could **not**:

* forge a signature from public material only,
* make message A verify for message B at the `verify_ots` layer,
* substitute commitments / Merkle root / fingerprint / params hash / ψ-commitment / key id,
* downgrade to legacy SIGv2 on an OTS-required path,
* extract seed / ψ★ / unrevealed slices from any public artifact, log, error, or temp file,
* bypass the canonical public-key or signature field-set checks,
* harvest enough Lamport halves from a *single* signature.

The cryptographic core is sound. **The remaining cheap outs are at the integration boundary** — exactly the "boring implementation bypasses" the bounty wants closed:

| ID | Title | Severity | Bounty-blocking |
|----|-------|----------|-----------------|
| **M1** | OTS block signature does not bind the block body (`Block.messages`) | **High** | **Yes** |
| **M2** | Replay-ledger reconstruction predicate is narrower than acceptance → delete JSONL + restart re-opens replay (fail-open) | **High** | **Yes** |
| **M3** | `PersistentOTSReplayLedger` has no inter-instance/file lock → double-accept across instances/processes | **Medium** | Partially (overlaps the disclaimed multi-node gap) |

All three are demonstrated by runnable scripts in `attacks/` and pinned by `tests/test_ots_mythos_break.py` (live-break demos pass; desired-secure-behavior pins are `xfail(strict=True)`).

---

## M1 — OTS block signature does not bind the block body

* **ID:** M1
* **Title:** WaveLock-OTS block authentication binds only `meta.ots_auth.message`, not `Block.messages`; the block's transaction payload is unauthenticated.
* **Severity:** High (block acceptance authenticates content the signer never signed).
* **Affected files:**
  * `wavelock/network/server.py` (`build_ots_block_meta`, `_verify_ots_block`)
  * `wavelock/chain/Block.py` (`messages` vs `meta`)
* **Exploit command:**
  ```bash
  PYTHONPATH=. python attacks/ots_block_body_unbound.py
  PYTHONPATH=. python -m pytest tests/test_ots_mythos_break.py -k body -q
  ```
* **Expected behavior:** An OTS-authenticated block should be rejected unless the signed message commits to the block's actual contents (its `messages`/transaction list and identity).
* **Actual behavior:** `_verify_ots_block` verifies the signature against `meta.ots_auth.message` only. `Block.messages` — what the rest of the chain treats as the block's payload (cf. the legacy curvature path which parses `b.messages`) — is never tied to the signed message. A block whose body is `["TRANSFER 1000000 TO ATTACKER"]` but whose `meta.ots_auth.message` is `"hello world"` (with a valid signature over `"hello world"`) is **accepted**.
* **Minimal reproduction:**
  ```python
  meta = server.build_ots_block_meta(pub, "hello world", sign_ots(sec, "hello world"))
  evil = Block(index=1, messages=["TRANSFER 1000000 TO ATTACKER"],
               previous_hash="0"*64, difficulty=1, block_type="OTS", meta=meta)
  assert server._verify_ots_block(evil, None, ledger=led) is True   # BREAK
  ```
* **Why it matters / attacker story:** The threat model grants the attacker *public signatures*. A signer who ever publishes a benign OTS signature (a CLI demo `sig.json`, an off-chain receipt, a "proof of key control") has signed a blank cheque: anyone holding that public `(message, signature)` pair can mint a **first-acceptance** OTS block whose `messages` payload is arbitrary. The replay ledger does not help — this is the *first* use of that `one_time_key_id`. `Block.calculate_hash` covers `merkle_root` (derived from `messages`) and `meta`, so the block hash is well-formed; the gap is that the *signature* is just an opaque field inside `meta` and authenticates a disjoint string.
* **Suggested fix:** Make the signed message a function of the block body. E.g. sign `H("WL-OTS-BLOCK-v1" || canonical(index, previous_hash, merkle_root, block_type, meta_without_ots_auth))` and have `_verify_ots_block` recompute that digest and require `signature`'s message to equal it; reject any block whose recomputed body digest differs. Equivalently, forbid `ots_auth.message` from being attacker-chosen and bind it to `b.messages`.
* **Blocks scoped bounty:** **Yes** — "OTS block acceptance" is explicitly in scope, and here acceptance authenticates content the key holder never authorised.

---

## M2 — Replay-ledger reconstruction is narrower than acceptance (durable fail-open)

* **ID:** M2
* **Title:** `ChainState.load_from_disk` reconstructs consumed ids with `block_requires_ots(b)` (no cfg), which is narrower than the acceptance predicate `block_requires_ots(b, cfg)`; deleting the JSONL and restarting re-opens replay for `require_ots`-classified blocks.
* **Severity:** High (replayed OTS identity accepted after first acceptance; the durable replay control fails OPEN).
* **Affected files:**
  * `wavelock/network/server.py` (`ChainState.load_from_disk`, `block_requires_ots`, `_verify_ots_block`)
  * `wavelock/crypto/ots_ledger.py` (`PersistentOTSReplayLedger._load`, `index_signature`)
* **Exploit command:**
  ```bash
  PYTHONPATH=. python attacks/ots_ledger_reconstruction_failopen.py
  PYTHONPATH=. python -m pytest tests/test_ots_mythos_break.py -k "replay or reconstruction" -q
  ```
* **Expected behavior:** Deleting the local JSONL cache must not weaken replay protection: the consumed set should be fully recoverable from agreed chain state, so a previously-accepted OTS signature can never be accepted again.
* **Actual behavior:** Two predicates disagree:
  * **Acceptance** (`try_accept_block` → `block_requires_ots(b, cfg)`) treats a block as OTS if `block_type == "OTS"` **OR** `meta.auth_scheme == WaveLock-OTS-v1` **OR** `cfg.require_ots`.
  * **Reconstruction** (`load_from_disk` → `block_requires_ots(b)` with **no cfg**) cannot see `cfg.require_ots`.

  A block that was OTS-verified and consumed *only because* the node runs with `cfg.require_ots = True` (generic-typed, no `auth_scheme` in meta, just an `ots_auth` payload) is **not** recognised as OTS during reconstruction, so `index_signature` is never called for it. `index_signature` is memory-only, so once the durable JSONL is deleted, reconstruction is the only recovery path — and it has this hole. Result: after `rm ledger/ots_replay.jsonl` + restart, the previously-consumed signature is accepted a second time.
* **Minimal reproduction:** see `attacks/ots_ledger_reconstruction_failopen.py::replay_after_delete_and_restart` (returns `True` = fail-open). Verified end-to-end:
  ```
  1) first accept           : True
     replay on same ledger  : False
  2) after delete+restart, is_consumed: False
     REPLAY accepted after restart    : True   <-- FAIL-OPEN
  ```
* **Suggested fix:** Use the **same** predicate everywhere. Reconstruct from any block carrying `meta.ots_auth` (i.e. `if _extract_ots_auth(b) is not None`) regardless of cfg, and/or persist `index_signature` folds durably so reconstruction is not the sole recovery path. More robustly, make the consumed set a deterministic function of *all* accepted blocks that carry OTS auth, independent of runtime config.
* **Blocks scoped bounty:** **Yes** — "replay protection" is in scope; this is a concrete durable fail-open within the single-node closure the project claims to provide. The attacker abilities used (delete local ledger, set env, restart) are all explicitly granted.

---

## M3 — Durable ledger lacks inter-instance/file locking (double-accept)

* **ID:** M3
* **Title:** `PersistentOTSReplayLedger` serialises `accept` with a per-*instance* `threading.RLock` and appends with `fsync` but no `flock`/`O_EXCL` claim; two instances on the same file both accept the same signature.
* **Severity:** Medium (overlaps the project's disclaimed cross-node gap, but the mechanism is concrete and also bites single-host multi-process / dual-ledger deployments).
* **Affected files:**
  * `wavelock/crypto/ots_ledger.py` (`PersistentOTSReplayLedger.accept`, `_append`)
  * `wavelock/network/server.py` (two distinct ledgers: `CONSENSUS_OTS_LEDGER` vs `OTS_LEDGER`)
* **Exploit command:**
  ```bash
  PYTHONPATH=. python attacks/ots_ledger_concurrent_double_accept.py
  PYTHONPATH=. python -m pytest tests/test_ots_mythos_break.py -k double -q
  ```
* **Expected behavior:** At most one accept of a given `one_time_key_id` across any instances sharing the same durable ledger file.
* **Actual behavior:** The RLock is per-object. Two instances opened on the same JSONL each have an empty in-memory set, both pass the check, both append a record, both return `True`. The on-disk JSONL then holds two consumption records for one id — direct evidence the one-time invariant was violated. Note also the single node keeps two unrelated ledgers (`CONSENSUS_OTS_LEDGER` for blocks, `OTS_LEDGER` behind `verify_ots_payload`) that never share state.
* **Minimal reproduction:** `attacks/ots_ledger_concurrent_double_accept.py::two_instances_double_accept` returns `True`.
* **Suggested fix:** Guard the read-check-append critical section with an OS file lock (`fcntl.flock(LOCK_EX)`) on the ledger file (or a sidecar lockfile), re-reading any appended-since records under the lock before deciding; or claim ids with `O_CREAT|O_EXCL` marker files as the host-local signing registry already does. Unify the two server-level ledgers.
* **Blocks scoped bounty:** Partially. The project's honest caveat explicitly defers full cross-node closure; two instances ≈ two nodes. But the *single-host multi-process* and *dual-ledger* cases are within one deployment and should be called out, so I rate it Medium rather than "out of scope".

---

## Attacks attempted that FAILED (the core holds)

These are the strongest cheap attacks I ran against the cryptographic core; all were correctly defended.

1. **Public-key-only forgery** (`attacks/forge_from_snapshot.py::attempt_forge_ots_from_public`): submitting public commitments as if they were revealed secrets. Fails — each revealed slice must be a SHAKE256-256 preimage of `pk[i][bit]`; the public commitment is not. (256-bit preimage per slice.)
2. **Merkle-root / commitment substitution** (`attacks/ots_key_substitution.py`): garbage root and victim-fingerprint-over-attacker-commitments. Fails — `verify_ots`/`load_public_key` recompute the Merkle root from `pk_commitments` and recompute the fingerprint over the canonical payload (which includes `pk_commitments` and `merkle_root`).
3. **Signature malleability** (`attacks/ots_signature_malleability.py`): mutating `one_time_key_id`, `version`, `hash_alg`, dropping/adding fields, dropping `message_digest`. All fail — exact canonical field-set, constant checks, present-and-equal `message_digest`, and fingerprint/key-id/params/ψ binding.
4. **Message A → message B at `verify_ots`:** the digest is `H(WL-OTS-MSG-v1 || params_hash || message)` and the verifying bits are recomputed from the message (the carried `message_digest` is cross-checked, not trusted). Two messages collide only on a SHAKE256-256 collision. Infeasible.
5. **Encoding / canonicalization:** duplicate JSON keys resolve identically (Python stdlib `json` last-wins on both the CLI and server paths); hex case, slash-escaping, and field reordering on the public key all change the recomputed Merkle root/fingerprint and are rejected; no Unicode/CRLF normalization is applied, so "equivalent-looking" messages simply produce different (correctly non-matching) digests — fail-closed, not a forge.
6. **Legacy SIGv2 downgrade on OTS path:** `_verify_ots_block`, `verify_ots_payload`, and `PersistentOTSReplayLedger.accept` all reject any non-`WaveLock-OTS-v1` scheme on either object before verification; there is no fallback to `_verify_curvature` for OTS-required blocks.
7. **Single-signature half-harvest:** one signature reveals exactly one half per bit; forging a different message needs the *other* half at every position where the target digest differs — i.e. a digest collision. Reuse (Finding C) remains the only route and is correctly preserved as a warning.
8. **Secret leakage:** public key, signature, block meta, server reject logs, and the host-local `.used` markers contain only ids/commitments/hashes. ψ★ appears only behind the explicit `--unsafe-export-secret-state` flag (loudly warned). No seed/ψ★/unrevealed-slice leak found.
9. **Corrupt ledger fails open:** a corrupt JSONL raises `OTSLedgerError` at construction (module import of `server` constructs `CONSENSUS_OTS_LEDGER`), so the node fails closed rather than open. (The *deletion* path is the real hole — see M2.)

---

## Remaining unproven assumptions (not converted to a runnable break)

* **SHAKE256-256 preimage/collision resistance** — assumed; all forgery resistance rests on it.
* **One-time Lamport assumption** — reuse → total forgery is inherent (Finding C); only key-management and the replay ledger defend it.
* **ψ-binding adds no proven extra hardness** — `psi_commitment` and `params_hash` are correctly *bound* (equality-checked and folded into the fingerprint), but they provide commitment/agility, not demonstrated additional cryptographic hardness. Not a break.
* **Cross-node / global replay closure** — still future work; the project says so. M2/M3 show the *single-node* closure is not yet airtight even before reaching the multi-node question.
* **Merkle multi-signature roadmap** — still future work; not exercised here.

---

## Verdict on bounty readiness

**Not yet.** The cryptographic core (canonical verification, fingerprint/Merkle binding, strict field sets, one-time semantics) is solid and resisted every cheap forge/substitution/downgrade attempt. But two integration-layer bypasses remain that a bounty hunter would hit immediately:

* **M1** — block authentication does not bind the block body, so "OTS block acceptance" does not actually authenticate the block's transactions.
* **M2** — the durable replay control fails OPEN after a ledger delete + restart for `require_ots`-classified blocks, because reconstruction uses a narrower predicate than acceptance.

Close M1 and M2 (and tighten M3's locking / unify the two ledgers) and the system is ready for a scoped bounty focused on the actual cryptographic construction (SHAKE/Lamport/ψ-binding) and the future cross-node consensus work, rather than on implementation plumbing.
