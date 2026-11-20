# WaveLock-OTS acceptance: formal verification of single use and replay rejection

Issue: [#41](https://github.com/rickyjreyes/Wavelock/issues/41). Related: #25, #26, #27, #28, #31, #32, #37.

**Scope.** Stateful acceptance of WaveLock-OTS signatures and OTS blocks on one host: identity
consumption, replay rejection, canonical block binding, crash and restart behaviour, same-host
concurrency. This is a protocol-level result. It is **not** a proof that WaveLock-OTS is
unforgeable, collision resistant, post-quantum secure, 256-bit secure, or one-time across hosts,
and it imports nothing from CC-Core-v1-B or the historical PDE commitment.

**Frozen source.** `master` at `b310ea5`, scheme `WaveLock-OTS-v1`, signature/public-key
`VERSION = 1`, replay `LEDGER_VERSION = 1`, `OTS_BLOCK_TRANSCRIPT_VERSION = 1`, block header
version 2 on the OTS-required path, `require_ots = true` (the default). This change modifies two
functions of `wavelock/crypto/ots_ledger.py` (below); no wire format, record format or protocol
version changes.

Lean 4.23.0, Mathlib v4.23.0 (the pinned toolchain). Modules under `lean/WaveLockProofs/`:
`OTSKeyConsumption`, `LedgerLinearizability`, `KeyReuseCounterexample`, `OTSVerification`,
`AuthenticatedBlockBinding`, `OTSConformance` (generated).

## 1. Verdict

Under a **trusted, durable, single-host store** (no hostile edits, no rollback, `fsync` honest,
working `flock`/SQLite exclusion), the corrected acceptance protocol never accepts a block whose
`one_time_key_id` or `public_key_fingerprint` is already consumed, over any finite interleaving of
any number of same-host processes, with write failures and crashes at every modelled point,
restarts, and loss of the side replay file. Proved in Lean for every pure verifier satisfying two
named hypotheses, which are proved for the Python instance.

That guarantee is **per identity label**, and it stops at the host:

* The identities are labels inside the public key. A holder of the secret seed can mint a fresh
  `one_time_key_id` (hence a fresh fingerprint) over the same Lamport key material and have a
  second block accepted. On the standalone payload path anyone can relabel an already-accepted
  signature and replay it. One-time use of the *key material* is not enforced (COUNTEREXAMPLE,
  reproduced against the implementation).
* Two hosts without shared state, or one host whose storage is rolled back, can accept one
  identity twice (COUNTEREXAMPLE).
* At `b310ea5` the guarantee itself failed for one identity (the key id string `"None"`) under the
  documented side-ledger-loss recovery path, and four key-id shapes let one block make a node
  permanently unable to accept (present defects, reproduced, corrected here).

## 2. Model

`OTSKeyConsumption.lean` is a transition system over a host:

| Lean | Python (`b310ea5` + this change) |
|---|---|
| `Disk.chain`, `chainTail` | `blk*.jsonl` (complete block lines; tail state clean / record-without-newline / torn) |
| `Disk.replay`, `replayTail` | `ots_replay.jsonl` records `(one_time_key_id, leaf_id)`; `transcript` is written but never read for decisions |
| `Phase.*`, `Mem` | one process: `ChainState` + `PersistentOTSReplayLedger` in-memory sets |
| `Sys.chainLock`, `ledgerLock` | `chain-writer.lock`, `<ledger>.lock` (`flock`; SQLite `BEGIN EXCLUSIVE` without `fcntl`) |
| `Event.begin` | `try_accept_block`: take chain lock, `load_all_blocks`, `verify_ots_chain`, `_reconstruct_consumed_ots`, `_verify_pow_and_linkage`, `verify_ots_block` |
| `Event.ledger` | `PersistentOTSReplayLedger.accept`: take ledger lock, `_scan_file_into_sets`, test `kid ∈ keys ∨ leaf ∈ leaves` |
| `Event.writeRecord o` | `_append` (`write; flush; fsync; fsync(dir)`) |
| `Event.writeBlock o` | `CHAIN.append` → `save_block_to_disk` |
| `Event.payBegin`, `payWrite` | `server.verify_ots_payload` on the shared ledger |
| `Event.crash`, `restart` | process death (the kernel drops `flock`s); process start + `load_from_disk` |
| `Event.deleteReplay` | side replay file deleted or lost (the Mythos M2 recovery model) |
| `WriteOutcome` | `ok`; `fail e` (exception into the caller); `crash e` (death) with `e` ∈ nothing / complete line / torn line / record without newline |

Pure checks are parameters (`AcceptSpec`): `ident`, `blockOK`/`linkOK` (per-block conditions of
`verify_ots_chain`), `acceptOK` (all pure acceptance checks), `payOK`, `recon`
(`index_signature`), `reconFail`, `recordOK`. `pySpec fixed` instantiates them for the Python
code: `fixed = false` is `b310ea5`, `fixed = true` adds this change.

Checks happen in separate events from the writes they authorise. Indivisibility is **derived** from
lock ownership, not assumed: the invariant records what each lock holder knows about the files.

**Named omissions.** A blocked lock acquisition is a no-op event (the process waits). Acquiring a
free lock and reading the protected file is one event (no other process can write that file in
between; an external deletion commutes with the read). Threads of one process are serialized by
`RLock` before the `flock` and are not separated. `fsync` is honest. Storage that loses
acknowledged writes or is edited by an adversary is outside the system (see §6). Identity values
are the strings the ledger stores (`str(...)`); `CBlock`/`RawKid` record where Python's raw and
string views differ (`str(None) == "None"`; arrays are unhashable).

## 3. Main theorems (`LEAN_PROVED`)

Hypotheses, both proved for `pySpec true` (`pySpec_acceptSound`, `pySpec_losslessRecon`):

* `AcceptSound`: every pure acceptance check implies the chain validator's per-block conditions.
* `LosslessRecon`: `index_signature` folds back both identities of every block acceptance admits.

```lean
theorem inv_run (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) : Inv S ad (run Discipline.implemented S ad es)

theorem chain_single_use (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) :
    (S.kids (run Discipline.implemented S ad es).disk.chain).Nodup ∧
      (S.fps (run Discipline.implemented S ad es).disk.chain).Nodup

theorem consume_fresh (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) {p : ℕ} {m : Mem K F} {b : B}
    (hp : (run Discipline.implemented S ad es).procs p = .checked m b) :
    (S.ident b).1 ∉ S.consumedKeys (run Discipline.implemented S ad es).disk ∧
      (S.ident b).2 ∉ S.consumedLeaves (run Discipline.implemented S ad es).disk

theorem at_most_one_acceptance (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es₁ es₂ : List (Event B)) {p q : ℕ} {o : WriteOutcome} {m₁ m₂ : Mem K F} {b₁ b₂ : B}
    (hadds : o.effect.adds = true)
    (h₁ : (run Discipline.implemented S ad es₁).procs p = .recorded m₁ b₁)
    (h₂ : (run Discipline.implemented S ad (es₁ ++ .writeBlock p o :: es₂)).procs q =
      .recorded m₂ b₂) :
    (S.ident b₁).1 ≠ (S.ident b₂).1 ∧ (S.ident b₁).2 ≠ (S.ident b₂).2

theorem wavelock_ots_chain_single_use (ad : Bool) (es : List (Event CBlock)) :
    ((pySpec true).kids (run Discipline.implemented (pySpec true) ad es).disk.chain).Nodup ∧
      ((pySpec true).fps (run Discipline.implemented (pySpec true) ad es).disk.chain).Nodup
```

`es` is an arbitrary finite trace: any processes, any interleaving of their micro-steps, any write
outcomes including crashes mid-write, restarts, and with `ad = true` loss of the side file.
`consume_fresh` is the issue's statement: a durable consumption happens only for identities absent
from the authoritative pre-transition state (accepted chain *and* replay records).
`at_most_one_acceptance` is one-winner over arbitrary interleavings: once a block is durably
appended, no process can even reach the consumption phase for a block sharing either identity.

| Brief item | Result |
|---|---|
| 1. first valid acceptance consumes both identifiers | `writeRecord_consumes_both`, `writeBlock_appends` |
| 2. reuse of either identifier rejected | `consume_fresh`, `at_most_one_acceptance` |
| 3. consumed set monotone | `consumed_monotone` (every event except side-file loss), `chain_suffix_step` (chain append-only under every discipline) |
| 4. rejection authorises nothing | `control_events_keep_disk`; uncertain writes only add consumption (`written_then_error_is_durable`) |
| 5. restart reconstruction | `restart_recovers` (needs `LosslessRecon`; false at `b310ea5`: `legacy_reconstruction_not_lossless`) |
| 6. check/consume indivisible | derived from lock ownership in `Inv`; necessity: `unlocked_check_then_consume_double_accepts` |
| 7. serialized concurrent attempts | `python_one_winner`; contrast `locked_same_schedule_single_accept` |
| liveness | `wavelock_ots_no_self_dos`: without torn writes, no sequence of blocks or payloads makes the node's files unloadable |

## 4. Verification and block binding (`LEAN_PROVED`, conditional where stated)

`OTSVerification.lean` transcribes `verify_ots`/`_check_public_key_canonical` over parsed JSON
(`JVal`, `JObj`); hashes are opaque (`OTSCrypto`); `str(x)` and `x == y` are parameters with only
CPython's guarantees (`PyEnv`).

* `verifyOTS_sound` — fail-closed: acceptance implies every check. Rejected: non-object key or
  signature; missing or extra fields on either; wrong scheme, version or hash algorithm; malformed
  commitment table; Merkle root or public-key fingerprint that does not recompute; mismatched key
  id, fingerprint, parameter hash or ψ-commitment; unparseable parameter hash; wrong carried digest;
  any unopened slice. A legacy SIGv2 signature fails the field and scheme checks.
* `verifyOTS_identity` — the replay identities of an accepted signature are those of the verified
  key: a string key id equals the key's, the fingerprint is recomputed from the key's payload.
* `fingerprint_binding` — equal fingerprints ⇒ equal canonical payloads **or** an explicit
  `Collision` of the fingerprint function (`CONDITIONAL` on collision resistance for uniqueness).
* `kid_relabel_verifies` — for any accepted `(pk, m, sig)` and any value `k`, the triple relabelled
  with key id `k` and the recomputed fingerprint verifies, with the same revealed slices. The key id
  is a free label.
* `version_check_accepts_noncanonical` — the version checks are Python numeric equality: `true` and
  `1.0` pass (canonicalisation finding; no replay consequence).

`AuthenticatedBlockBinding.lean` transcribes `ots_blocks.py` (transcript v1):

* `verifyOTSBlock_binds` — the carried message equals the digest of the *received* block's
  transcript and the signature verifies against exactly that digest (Mythos M1 fix).
* `transcriptPayload_injective`, `block_binding`, `body_change_needs_collision` — authentication
  copied to another block verifies only if the transcripts (body, type, parent, metadata including
  the public key and signed header) are equal or the transcript digest collides.
* `header_bound` — under header v2 the signed context is the block's own header, given that
  canonical JSON separates header contexts.
* `block_relabel_needs_second_message` — a relabelled public key on the block path yields a digest
  collision or a second signed message under the same commitment table (`TableUsedTwice`), i.e. a
  one-time-unforgeability break. Contrast `kid_relabel_verifies` on the payload path.
* `preM1_body_independent` — the pre-M1 verifier ignored the body (historical).

## 5. Findings

### Present defects at `b310ea5`, corrected here (`COUNTEREXAMPLE` → fixed; conformance tested)

`verify_ots` compares key ids with `!=` and does not type them; the replay record schema requires a
non-empty string; reconstruction treated the *string* `"None"` as a missing field.

| Input (one valid, mined block) | `b310ea5` | Lean |
|---|---|---|
| `one_time_key_id = ""` | accepted, then the replay file never loads; `server` cannot be imported | `legacy_empty_key_id_bricks_replay_ledger` |
| `one_time_key_id = null` | accepted, then every chain reload raises in `_reconstruct_consumed_ots` | `legacy_null_key_id_bricks_chain_reload` |
| `one_time_key_id = ["x"]` | accepted, then `verify_ots_chain` raises `TypeError` on every reload | `legacy_list_key_id_breaks_chain_validation` |
| `one_time_key_id = "None"`, side file lost, restart, different key reusing the id | **second block accepted**: the chain holds the key id twice and fails its own validator | `legacy_none_sentinel_double_accepts_key_id`, `legacy_reconstruction_not_lossless` |

Corrections in `wavelock/crypto/ots_ledger.py`:

1. `PersistentOTSReplayLedger.accept` rejects a signature whose `one_time_key_id` is not a
   non-empty string or whose fingerprint is not a string (`_identity_types_ok`), so the ledger never
   writes a record its own loader rejects. Behaviour change: a numeric key id (previously accepted,
   harmless) is now rejected. Honest keys use `uuid4` strings; no reference vector changes.
2. `index_signature` skips only absent fields, not the string `"None"`.

The pure verifier, chain validator, wire formats and record format are unchanged. After the
correction the Lean hypotheses hold (`pySpec_acceptSound`, `pySpec_losslessRecon`,
`pySpec_recordAdmissible`, `pySpec_reconTotal`), and `fixed_*` theorems show each input rejected.

### Boundaries: present behaviour outside the guarantee (`COUNTEREXAMPLE`, not corrected)

* **Key material reuse under fresh labels.** The fingerprint covers `one_time_key_id`, so the same
  `pk_commitments`/`merkle_root` under a new key id is a fresh identity. Reproduced: two blocks over
  one seed accepted, 109 of 256 bit positions then had both secret halves public
  (`copied_key_material_accepted_twice`). The previous `_ids_for` docstring claimed the opposite;
  it is corrected. Closing this needs a versioned change (consume `merkle_root` in a new ledger
  record version and in `verify_ots_chain`), so it is left as an obligation.
* **Third-party relabelling on the payload path.** `verify_ots_payload` accepts an already-consumed
  signature (same message, same revealed slices) relabelled with an observer-chosen key id
  (`kid_relabel_verifies`; `test_boundary_payload_path_relabelled_replay_is_accepted`). The block
  path is not affected (`block_relabel_needs_second_message`).
* **Revealed-slice hex is malleable** (case, whitespace): distinct verifying signatures for one
  message; the replay identity is unchanged.
* **Return value is not authoritative.** An I/O error after the block line lands reports rejection
  while the block is accepted (`written_then_error_is_durable`).

### Historical, fixed before this change (counterexamples to weaker models)

* Mythos M1 (body not bound): `preM1_body_independent`.
* Mythos M2 (no reconstruction from the chain): `historical_m2_replay_after_side_file_loss`.
* Mythos M3 (no re-scan under the lock): `historical_m3_no_rescan_double_consumes`.

## 6. Crash, persistence, concurrency and distribution

| Scenario | Classification | Evidence |
|---|---|---|
| verification succeeds before consumption | `LEAN_PROVED`: checks and writes are separate events; `consume_fresh` holds at write time | `inv_run` |
| consumption durable before chain append | `LEAN_PROVED` + tested | `Inv.chain_in_replay`; trace `crash_between_consumption_and_append_burns_identity` |
| crash after consumption, before acceptance | `LEAN_PROVED`: identity burned, never accepted; `IMPLEMENTATION_CONFORMANCE_TESTED` | `crash_after_record_burns_identity` |
| partially written JSONL record | `LEAN_PROVED` fail-closed (no acceptance, no restart until repaired); a record without its newline poisons the next append | `torn_record_fail_closed`, `record_without_newline_poisons_next_append`, `chain_garbage_absorbing` |
| storage reports an error | `LEAN_PROVED`: before bytes → nothing consumed; after bytes → consumed (burn) or durably accepted despite a `False` return | traces `record_write_*`, `block_write_*` |
| ledger reconstructed from accepted blocks | `LEAN_PROVED` with lossless reconstruction (true after correction); burned identities are not recovered (#25) | `restart_recovers`, `burned_identity_lost_with_side_file` |
| two workers, one identity | `LEAN_PROVED` for any interleaving under the locks; real processes tested | `python_one_winner`; `tests/test_bounty_durability.py::test_same_host_multiprocess_exclusion` |
| replay cache and chain disagree | `LEAN_PROVED`: the check uses chain ∪ cache, so either source suffices (`LosslessRecon ∨ ¬ad`) | `legacy_none_sentinel_safe_without_file_loss` |
| earlier valid prefix restored | `COUNTEREXAMPLE`: chain-only truncation loses blocks undetected but still refuses the identity; restoring chain *and* cache re-accepts it | `chain_only_rollback_still_rejects_replay`, `rollback_reaccepts_identity` |
| two independent hosts | `COUNTEREXAMPLE`: each accepts a different block under one identity | `independent_nodes_both_accept` |
| in-process atomicity | `LEAN_PROVED` (serialized threads are one process) | model assumption, `RLock` |
| same-host interprocess exclusion | `LEAN_PROVED` assuming exclusive `flock`/SQLite locks | `Inv` lock clauses; `no_chain_lock_appends_siblings` |
| process-crash durability | `CONDITIONAL` on honest `fsync` | `WriteOutcome` |
| physical power loss | `OPEN` (#26): no Windows directory flush; media loss untested | — |
| trusted persistence / hostile rollback | `OPEN` (#31): needs an authenticated, fresh checkpoint | rollback counterexamples |
| distributed agreement | `OPEN` (#28) | independent-hosts counterexample |

## 7. Implementation correspondence

`lean/ots_conformance_traces.json` holds 20 traces with their expected outcome (reported
acceptances, final chain, final replay records, whether each file loads). `scripts/gen_ots_conformance.py`
turns them into `OTSConformance.lean`, one `decide`-checked theorem per trace.
`tests/test_ots_formal_conformance.py` replays the same traces against the implementation (real
keys, signatures and mined blocks; `server.try_accept_block_dict`; several ledger/chain instances
as processes; injected failures at `_append` and `save_block_to_disk`; `b310ea5` behaviour
restored by monkeypatching for `legacy` traces) and checks the same expected outcome, plus that the
committed Lean file equals the generator output. Mutating an expectation, or running a `legacy`
trace against the corrected code, fails the test.

Covered: first use; repeated signature; different message under a copied key; same material under a
fresh key id; tampered body, slice, legacy signature, unbound message, stale fingerprint (none
consumes); malformed key ids; second process; payload-then-block; restart after side-file loss;
crash between consumption and append; write errors before/after bytes; torn and unterminated
records; torn block line.

| Lean | Python | Tests |
|---|---|---|
| `step .begin` | `server.try_accept_block`, `ChainState.acceptance_lock/load_from_disk`, `_verify_pow_and_linkage`, `ots_blocks.verify_ots_block` | `test_ots_consensus.py`, conformance traces |
| `step .ledger`, `.writeRecord` | `PersistentOTSReplayLedger.accept`, `_scan_file_into_sets`, `_append` | `test_ots_mythos_break.py` (M3), `test_bounty_durability.py` |
| `step .writeBlock` | `ChainState.append`, `chain_utils.save_block_to_disk` | `test_bounty_durability.py::test_process_exit_between_replay_and_append_stays_consumed` |
| `AcceptSpec.chainValid` | `ots_blocks.verify_ots_chain` | conformance `chain_loads` |
| `reconFixed` / `reconLegacy` | `index_signature` after / at `b310ea5` | `fixed_none_*`, `legacy_none_*` traces |
| `verifyOTS` | `wavelock_ots.verify_ots` | `test_ots_security.py`, `test_ots_redteam.py` |
| `verifyOTSBlock` | `ots_blocks.verify_ots_block` | `test_ots_mythos_break.py` (M1) |

**Refinement obligations not discharged.** The Python code is related to the model by inspection
and by finite traces, not by a semantics proof. In particular: `CBlock.authentic` abstracts
`verify_block_integrity ∧ verify_ots_block`; the opaque hash functions are not shown to be the
SHAKE256/SHA-256 code; Python dictionary, `json` and `str` behaviour is assumed as stated in
`PyEnv`; `flock`, SQLite and `fsync` semantics are assumed.

## 8. Layers

* **A. Protocol correctness** (this work): §3–§6, `LEAN_PROVED` under the stated store model.
* **B. Cryptographic assumptions** (not proved, never hypotheses of §3): one-time unforgeability
  of the Lamport construction over SHAKE256-256 (needed to exclude `TableUsedTwice` for a party
  without the seed); collision resistance of the fingerprint and transcript digests (to exclude the
  `Collision` alternatives); entropy of `os.urandom` seeds. The ψ-evolution adds no proven hardness.
* **C. Operational/distributed**: honest `fsync`, exclusive locks on one filesystem, no hostile
  storage, a single authoritative host. Rollback resistance and cross-host agreement are not
  provided.

## 9. Remaining obligations

1. **Key-material single use** (highest value): consume `merkle_root` (or a digest of
   `params_hash ‖ psi_commitment ‖ merkle_root`) alongside the labels — a new `LEDGER_VERSION`,
   a `verify_ots_chain` rule, and a migration — then extend `AcceptSpec.ident` and re-run `inv_run`.
   Alternatively bind `one_time_key_id` to the key material in a new key version.
2. Payload path: bind relabelling, or document that `verify_ots_payload` deduplicates labels only.
3. Strict canonical types for `version` and hex fields (verifier-level, versioned).
4. Crash recovery tooling for torn tails and burned identities (#25); Windows power loss (#26).
5. Trusted checkpoints / rollback resistance (#31); cross-host agreement (#28).
6. A refinement proof from Python semantics to the model (beyond the finite traces).

## 10. Issue #41 acceptance criteria

| Criterion | Status |
|---|---|
| Freeze versioned keys, signatures, block identities, ledger states | done (§2, `OTSVerification`, `AuthenticatedBlockBinding`) |
| No already-consumed identity consumed, consistent durable local ledger | `LEAN_PROVED` (§3) |
| Malformed, legacy, duplicate, fingerprint-mismatched objects fail closed | `LEAN_PROVED` (`verifyOTS_sound`) + tested |
| Counterexamples: independent nodes, rollback without anchor, key reuse | `LEAN_PROVED` counterexamples + reproduced |
| Functional verification separated from cryptographic reduction | done (§8; conditional statements extract `Collision` / `TableUsedTwice`) |
| Kernel-checked, no `sorry` or unreviewed axioms | 129 theorems; axioms ⊆ {`propext`, `Classical.choice`, `Quot.sound`} |
| Encodings and negative tests cross-checked against the implementation | 20 traces + boundary pins |
| No claim of 256-bit security, collision resistance or global one-time use | none made |

Estimate: about **90%**. Every stated obligation and acceptance criterion is met, for the trusted,
durable, single-host model only. The remainder is not a missing theorem for that model: Python is
related to the model by finite traces rather than a refinement proof (§7), and whether "single use"
should mean per key material rather than per label (§9.1) is a protocol decision for the
maintainers. #41 is left open for that decision.
