# WaveLock-OTS acceptance: formal verification of single use and replay rejection

Issue: [#41](https://github.com/rickyjreyes/Wavelock/issues/41). Related: #25, #26, #27, #28, #31, #32, #37.

**Scope.** Stateful acceptance of WaveLock-OTS signatures and OTS blocks on one host: consumption
of identities and of one-time key material, replay rejection, canonical block binding, crash and
restart behaviour, same-host concurrency. This is a protocol-level result. It is **not** a proof
that WaveLock-OTS is unforgeable, collision resistant, post-quantum secure, 256-bit secure, or
one-time across hosts, and it imports nothing from CC-Core-v1-B or the historical PDE commitment.

**Revisions.** `b310ea5` (`master` before this work); **PR #47 phase I** (identity typing and the
reconstruction sentinel); **phase II, this revision** (consumption of the commitment table). Scheme
`WaveLock-OTS-v1`, signature/public-key `VERSION = 1`, `OTS_BLOCK_TRANSCRIPT_VERSION = 1`, block
header version 2 on the OTS path, `require_ots = true` (the default). Phase II changes the replay
record format (`LEDGER_VERSION` 1 → 2, version 1 still loads) and tightens the chain validity rule
(§5.2); keys, signatures and block transcripts are unchanged.

Lean 4.23.0, Mathlib v4.23.0 (the pinned toolchain). Modules under `lean/WaveLockProofs/`:
`OTSKeyConsumption`, `LedgerLinearizability`, `KeyReuseCounterexample`, `OTSVerification`,
`AuthenticatedBlockBinding`, `OTSConformance` (generated).

## 1. Verdict

Under a **trusted, durable, single-host store** (no hostile edits, no rollback, `fsync` honest,
working `flock`/SQLite exclusion), the current acceptance protocol never accepts a block whose
`one_time_key_id`, `public_key_fingerprint` **or commitment table** is already consumed. This holds
over any finite interleaving of any number of same-host processes, with write failures and crashes
at every modelled point, restarts, and loss of the side replay file. Hence **no public commitment
table authorises two accepted blocks**, whatever labels they carry. Proved in Lean for every pure
verifier satisfying two named hypotheses, which are proved for the Python instance; the table
statement uses no cryptographic assumption.

On the standalone payload path (`verify_ots_payload`) the replay file never records an atom twice,
and each consumption is checked against that file and the process's in-memory sets. While the
side file is intact every chain identity is in it; after it is lost, only a process that has
reloaded the chain still knows the chain's identities.

PR #47's counterexample `copied_key_material_accepted_twice` (two blocks over one table under
different key ids, both secret halves exposed at 109 of 256 positions) is now rejected, and so is
third-party relabelling of a consumed signature on the payload path.

Boundaries (§5.3): the guarantee is per **public table**. A holder who deliberately recommits the
same secret values under a different `params_hash` creates a different table, which is accepted;
this exposes secret values but cannot authorise anything under the first table. Two hosts without
shared state, or a rolled-back store, can still accept one table twice.

## 2. Model

`OTSKeyConsumption.lean` is a transition system over a host. Each signature consumes a list of
**identity atoms**; the theorems say no atom is consumed twice, for whichever atoms a revision
consumes:

| Revision (`PyRev`) | Atoms per signature |
|---|---|
| `b310ea5` | `kid`, `fp` (reconstruction drops the key id `"None"`; no identity typing) |
| `pr47` | `kid`, `fp` |
| `current` | `kid`, `fp`, `mat` (`ots_material_id`) |

| Lean | Python |
|---|---|
| `Disk.chain`, `chainTail` | `blk*.jsonl` (complete block lines; tail clean / record-without-newline / torn) |
| `Disk.replay`, `replayTail` | `ots_replay.jsonl` records, each the atoms it consumes (`transcript` is written, never read) |
| `Phase.*` (memory `List A`) | one process: `ChainState` + `PersistentOTSReplayLedger` in-memory sets |
| `Sys.chainLock`, `ledgerLock` | `chain-writer.lock`, `<ledger>.lock` (`flock`; SQLite `BEGIN EXCLUSIVE` without `fcntl`) |
| `Event.begin` | `try_accept_block`: chain lock, `load_all_blocks`, `verify_ots_chain`, `_reconstruct_consumed_ots`, `_verify_pow_and_linkage`, `verify_ots_block` |
| `Event.ledger` | `PersistentOTSReplayLedger.accept`: ledger lock, `_scan_file_into_sets`, duplicate test on every atom |
| `Event.writeRecord o` | `_append` (`write; flush; fsync; fsync(dir)`) |
| `Event.writeBlock o` | `CHAIN.append` → `save_block_to_disk` |
| `Event.payBegin`, `payWrite` | `server.verify_ots_payload` on the shared ledger |
| `Event.crash`, `restart` | process death (the kernel drops `flock`s); process start + `load_from_disk` |
| `Event.deleteReplay` | side replay file deleted or lost (Mythos M2 recovery model) |
| `WriteOutcome` | `ok`; `fail e` (exception into the caller); `crash e` (death); `e` ∈ nothing / line / torn / no newline |

Pure checks are parameters (`AcceptSpec`: `atoms`, `blockOK`/`linkOK`, `acceptOK`, `payOK`, `recon`,
`reconFail`, `recordOK`). Checks happen in separate events from the writes they authorise;
indivisibility is **derived** from lock ownership, not assumed.

**Named omissions.** A blocked lock acquisition is a no-op event. Acquiring a free lock and reading
the protected file is one event. Threads of one process are serialized by `RLock` before the
`flock`. `fsync` is honest. Hostile storage is outside the system (§6). Atom values are the strings
the ledger stores; `CBlock`/`RawKid` record where Python's raw and string views differ.

## 3. Key-material identity

### 3.1 Candidates, tested against the verifier

Each relabelling below is applied to an accepted `(pk, m, sig)` and then run through the real
`verify_ots` (all verify, except the `params_hash` change):

| Relabelling | A: `merkle_root` | B: digest of table strings | B′: digest of decoded table | C: digest(`params_hash`, `psi_commitment`, table) |
|---|---|---|---|---|
| new `one_time_key_id` | same | same | same | same |
| new `psi_commitment` | same | same | same | **changed** |
| other hex case / spacing of an unselected half | same | **changed** | same | **changed** |
| new `params_hash` (same slices) | rejected by `verify_ots` | | | |

* **C fails**: `psi_commitment` is a free label like the key id. `verify_ots` only compares the
  signature's copy with the key's; slices are opened under `params_hash`
  (`psi_relabel_verifies`).
* **B over strings fails**: the verifier string-compares only the *selected* halves; the others
  can be re-spelled (`bytes.fromhex` accepts any case and spacing) without changing the decoded
  table.
* **A and B′ both work.** A is already recomputed by `verify_ots` from the decoded bytes and must
  equal the key's `merkle_root` field, so it is the minimal choice: `wavelock_ots.ots_material_id`.

`ots_material_id(public_key)` recomputes the Merkle root from `pk_commitments`, requires it to equal
`merkle_root`, and raises otherwise. It is **not** `Block.merkle_root`, the tree over a block's
messages.

### 3.2 What is proved (`OTSVerification`)

`OTSCrypto.merkleRoot` now factors as `rootOf ∘ decodeTable` (`merkleRoot_eq`), matching
`_merkle_root_from_commitments`.

```lean
theorem materialId_eq_of_decode_eq {pk₁ pk₂ : JObj}
    (h : C.decodeTable (JObj.field pk₁ "pk_commitments") =
      C.decodeTable (JObj.field pk₂ "pk_commitments")) :
    materialId C pk₁ = materialId C pk₂

theorem distinct_material_distinct_tables {ι : Type} (key : ι → JObj) (l : List ι)
    (h : (l.map fun i => materialId C (key i)).Nodup) :
    (l.map fun i => C.decodeTable (JObj.field (key i) "pk_commitments")).Nodup

theorem material_binding {pk₁ pk₂ : JObj} {t₁ t₂ : List (List ℕ × List ℕ)}
    (h₁ : C.decodeTable (JObj.field pk₁ "pk_commitments") = some t₁)
    (h₂ : C.decodeTable (JObj.field pk₂ "pk_commitments") = some t₂)
    (h : materialId C pk₁ = materialId C pk₂) : t₁ = t₂ ∨ Collision C.rootOf t₁ t₂
```

* Equality preservation and its contrapositive (distinct ids ⇒ distinct tables) need **no
  assumption**. These are the directions safety uses.
* The converse (equal ids ⇒ equal tables) holds only modulo an explicit `Collision` of the Merkle
  root. It matters for liveness: two honest keys sharing a root would make the second one look
  consumed.
* `materialId_of_verified`: for a verified key the identifier is its verified `merkle_root`.
* `relabelled_replays_keep_material`: both relabellings the verifier admits (key id,
  `kid_relabel_verifies`; ψ-commitment, `psi_relabel_verifies`) keep the identifier, so a ledger
  consuming it rejects them.

## 4. Main theorems (`LEAN_PROVED`)

Hypotheses, proved for every typed revision (`pySpec_acceptSound`, `pySpec_losslessRecon`):
`AcceptSound` (acceptance implies the chain validator's per-block checks) and `LosslessRecon`
(`index_signature` folds back every atom of every block acceptance admits).

```lean
theorem chain_single_use (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) :
    (run Discipline.implemented S ad es).disk.chain.Pairwise (Disjoint S)

theorem consume_fresh (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) {p : ℕ} {m : List A} {b : B}
    (hp : (run Discipline.implemented S ad es).procs p = .checked m b) :
    ∀ a ∈ S.atoms b, a ∉ S.consumed (run Discipline.implemented S ad es).disk

theorem at_most_one_acceptance (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es₁ es₂ : List (Event B)) {p q : ℕ} {o : WriteOutcome} {m₁ m₂ : List A} {b₁ b₂ : B}
    (hadds : o.effect.adds = true)
    (h₁ : (run Discipline.implemented S ad es₁).procs p = .recorded m₁ b₁)
    (h₂ : (run Discipline.implemented S ad (es₁ ++ .writeBlock p o :: es₂)).procs q =
      .recorded m₂ b₂) :
    Disjoint S b₂ b₁

theorem wavelock_ots_material_single_use (ad : Bool) (es : List (Event CBlock)) :
    ((run Discipline.implemented (pySpec .current) ad es).disk.chain.map CBlock.material).Nodup

theorem wavelock_ots_fresh_triple (ad : Bool) (es : List (Event CBlock)) {p : ℕ} {m : List Atom}
    {b : CBlock} (hp : (run Discipline.implemented (pySpec .current) ad es).procs p = .checked m b) :
    b.kid.render ∉ consumedK (run Discipline.implemented (pySpec .current) ad es).disk ∧
      b.fp ∉ consumedF (run Discipline.implemented (pySpec .current) ad es).disk ∧
      b.material ∉ consumedM (run Discipline.implemented (pySpec .current) ad es).disk
```

`es` is an arbitrary finite trace: any processes, any interleaving of their micro-steps, any write
outcomes including crashes mid-write, restarts, and with `ad = true` loss of the side file.
`wavelock_ots_fresh_triple` is `Fresh(s, b) ⟺ K(b) ∉ K_s ∧ F(b) ∉ F_s ∧ M(b) ∉ M_s` at the moment
of consumption, for the authoritative state `s` (chain and replay records);
`wavelock_ots_records_triple` shows one record append consumes all three. `python_one_winner`: once
a block is durably appended, no block sharing its key id, fingerprint **or** table can even be
consumed, over any interleaving.

**Composition.** `wavelock_ots_material_single_use` gives pairwise distinct material ids for the
chain; `distinct_material_distinct_tables` turns that into pairwise distinct decoded commitment
tables. The link between the model's `CBlock.material` and `ots_material_id` of the real key is
by construction and by the conformance traces (§7), not a proof.

| Brief item | Result |
|---|---|
| consumed state `(K, F, M)` | `consumedK/F/M`; `wavelock_ots_fresh_triple`, `wavelock_ots_records_triple` |
| first valid acceptance consumes every atom | `writeRecord_consumes_all`, `writeBlock_appends` |
| reuse of any atom rejected | `consume_fresh`, `at_most_one_acceptance`, `python_one_winner` |
| consumed set monotone | `consumed_monotone`, `chain_suffix_step` |
| rejection authorises nothing | `control_events_keep_disk`; uncertain writes only add consumption |
| restart reconstruction | `restart_recovers` (needs `LosslessRecon`; false at `b310ea5`) |
| check/consume indivisible | derived from lock ownership; necessity: `unlocked_check_then_consume_double_accepts` |
| liveness | `wavelock_ots_no_self_dos` |

## 5. Findings and protocol changes

### 5.1 Phase I (PR #47): key-id defects at `b310ea5`, corrected

| Input (one valid, mined block) | `b310ea5` | Lean |
|---|---|---|
| `one_time_key_id = ""` | accepted, then the replay file never loads | `legacy_empty_key_id_bricks_replay_ledger` |
| `one_time_key_id = null` | accepted, then every chain reload raises | `legacy_null_key_id_bricks_chain_reload` |
| `one_time_key_id = ["x"]` | accepted, then `verify_ots_chain` raises `TypeError` | `legacy_list_key_id_breaks_chain_validation` |
| `"None"` + side-file loss + restart, another key reusing the id | **second block accepted** | `legacy_none_sentinel_double_accepts_key_id` |

Fixes: the ledger rejects key ids that are not non-empty strings (`_identity_types_ok`); reconstruction
skips only absent fields.

### 5.2 Phase II: commitment tables are consumed

| Path | Change |
|---|---|
| `wavelock_ots.ots_material_id` | new; the only definition of the material identity |
| `PersistentOTSReplayLedger.accept` | rejects if the key id, fingerprint **or** material id is consumed; records all three in one append |
| record format | `LEDGER_VERSION = 2` adds `material_id`; version 1 records still load (labels only); malformed or unknown versions fail closed |
| `verify_ots_chain` | a chain in which two blocks carry the same commitment table is invalid |
| `_reconstruct_consumed_ots`, restart | folds the material of every chain block back in; an OTS block whose table cannot be identified fails closed |
| `verify_ots_payload` | through `accept`: a relabelled consumed signature is rejected |
| `is_consumed(signature, public_key=None)`, CLI `mine` pre-check | with the public key, also tests the table |
| in-memory `OTSReplayLedger` | consumes key id, fingerprint and material |

Reproduced against the implementation before and after: same seed under a fresh key id, under a
fresh ψ-commitment and key id (holder-signed), and third-party payload relabelling (key id; key id
and hex spelling) are accepted under PR #47 and rejected now.

**Compatibility.** Honest keys have distinct tables, so honest chains and ledgers are unaffected.
This is a **consensus-affecting** tightening: a stored chain that reuses a table (which PR #47-era
code could accept) no longer validates, so the node refuses to load it (fail closed). Nodes running
older code accept blocks this code rejects. Version 1 records keep their labels consumed; their
tables are unknown, so a table burned (consumed, never appended) before upgrading is not protected
by material identity.

### 5.3 Boundaries (present behaviour, documented)

* **Secret values under a new table.** A holder with custom code can commit the same secret slice
  values under another `params_hash`. That yields a different public table and a fresh material id,
  so it is accepted and exposes secret values
  (`test_boundary_secret_values_under_a_new_table`). It cannot authorise anything under the first
  table, which stays consumed. Honest key generation derives the slices from `params_hash` and never
  does this. A party without the secret cannot do it either, because the slices would have to open
  a new digest.
* **Revealed-slice hex is malleable.** Distinct verifying signatures exist for one message; every
  replay identity, including the table, is unchanged.
* **The return value is not authoritative.** An I/O error after the block line lands reports
  rejection while the block is accepted (`written_then_error_is_durable`).
* **Non-canonical version encodings** (`true`, `1.0`) are accepted
  (`version_check_accepts_noncanonical`).

### 5.4 Historical (counterexamples to weaker models)

* PR #47 boundary, closed here: `copied_key_material_accepted_twice`, `pr47_payload_relabel_replayed`
  (`pySpec .pr47`); now `copied_key_material_rejected`, `current_payload_relabel_rejected`.
* Mythos M1/M2/M3: `preM1_body_independent`, `historical_m2_replay_after_side_file_loss`,
  `historical_m3_no_rescan_double_consumes`.

## 6. Crash, persistence, concurrency and distribution

| Scenario | Classification | Evidence |
|---|---|---|
| verification before consumption | `LEAN_PROVED`: freshness holds at write time | `consume_fresh` |
| consumption durable before chain append | `LEAN_PROVED` + tested | `Inv.chain_in_replay`; trace `crash_between_consumption_and_append_burns_identity` |
| crash after consumption, before acceptance | `LEAN_PROVED`: burned, never accepted | `crash_after_record_burns_identity` |
| partially written JSONL record | `LEAN_PROVED` fail-closed; an unterminated record poisons the next append | `torn_record_fail_closed`, `record_without_newline_poisons_next_append` |
| storage reports an error | `LEAN_PROVED`: before bytes nothing consumed; after bytes consumed or durably accepted | traces `record_write_*`, `block_write_*` |
| reconstruction from accepted blocks | `LEAN_PROVED` with lossless reconstruction, including tables; burned identities are not recovered (#25) | `restart_recovers`, `copied_key_material_rejected_after_side_file_loss`, trace `material_reconstructed_after_side_ledger_loss` |
| two workers, one identity or table | `LEAN_PROVED` for any interleaving under the locks; real processes tested | `python_one_winner`; `test_same_host_multiprocess_exclusion` |
| earlier valid prefix restored | `COUNTEREXAMPLE`: chain-only truncation still refuses; restoring chain and cache re-accepts | `chain_only_rollback_still_rejects_replay`, `rollback_reaccepts_identity` |
| two independent hosts | `COUNTEREXAMPLE`: one table accepted on both | `independent_nodes_both_accept` |
| same-host interprocess exclusion | `LEAN_PROVED` assuming exclusive `flock`/SQLite locks | `Inv`; `no_chain_lock_appends_siblings` |
| process-crash durability | `CONDITIONAL` on honest `fsync` | `WriteOutcome` |
| physical power loss | `OPEN` (#26) | — |
| hostile rollback / trusted checkpoints | `OPEN` (#31) | rollback counterexamples |
| distributed agreement | `OPEN` (#28) | independent-hosts counterexample |

## 7. Implementation correspondence

`lean/ots_conformance_traces.json` holds 25 traces, each tagged with a revision and an expected
outcome (reported acceptances, final chain, final replay records with their atoms, whether each
file loads). `scripts/gen_ots_conformance.py` generates `OTSConformance.lean` (one
`decide`-checked theorem per trace). `tests/test_ots_formal_conformance.py` replays every trace
against the implementation: real keys, signatures and mined blocks, `try_accept_block_dict`,
several ledger/chain instances as processes, failures injected at `_append` and
`save_block_to_disk`. For `current` traces it compares key id, fingerprint and the real
`ots_material_id`. `pr47` traces run with every table made to look fresh; `b310ea5` traces also
restore that revision's ledger functions. Mutation checks: running `pr47` traces against current
code, or the material traces under the PR #47 emulation, fails them.

`tests/test_ots_key_material.py` pins: identifier invariance under every admitted relabelling; fail
closed on malformed tables; v1/v2 record loading and malformed versions; chain validator and reload
rejecting a table-reusing chain; reconstruction recovering tables and failing closed; the
in-memory ledger; the secret-value boundary.

**Refinement obligations not discharged.** Python is related to the model by inspection and finite
traces, not a semantics proof: `CBlock.authentic` abstracts `verify_block_integrity ∧
verify_ots_block`; `CBlock.material` abstracts `ots_material_id`; hash functions, `bytes.fromhex`,
`json` and `str` behaviour, `flock`, SQLite and `fsync` semantics are assumed as stated.

## 8. Layers

* **A. Protocol correctness** (this work): §3–§6, `LEAN_PROVED` under the stated store model.
* **B. Cryptographic assumptions** (never hypotheses of §4): Merkle-root and fingerprint collision
  resistance (only for the converse of material identity, i.e. no false rejection of honest keys);
  one-time unforgeability of the Lamport construction over SHAKE256-256 (to exclude
  `TableUsedTwice` for a party without the secret); seed entropy. The ψ-evolution adds no proven
  hardness.
* **C. Operational/distributed**: honest `fsync`, exclusive locks on one filesystem, no hostile
  storage, one authoritative host.

## 9. Remaining obligations

1. **Secret-value single use** (highest remaining value): consume a digest of every revealed secret
   slice, so that recommitting the same secrets under a new table is detected (two signatures over
   the same secrets share about half their revealed slices). This needs a new record version and a
   chain rule.
2. A migration or recovery tool for version 1 ledgers whose burned identities lack tables (#25).
3. Strict canonical types for `version` and hex fields (verifier-level, versioned).
4. Crash recovery tooling for torn tails (#25); Windows power loss (#26).
5. Trusted checkpoints / rollback resistance (#31); cross-host agreement (#28).
6. A refinement proof from Python semantics to the model.

## 10. Issue #41 acceptance criteria

| Criterion | Status |
|---|---|
| Freeze versioned keys, signatures, block identities, ledger states | done (§2, §3, record v2) |
| No already-consumed identity consumed, durable local ledger | `LEAN_PROVED`, now for key id, fingerprint and table (§4) |
| Malformed, legacy, duplicate, fingerprint-mismatched objects fail closed | `LEAN_PROVED` (`verifyOTS_sound`) + tested |
| Counterexamples: independent nodes, rollback without anchor, key reuse | `LEAN_PROVED` + reproduced; same-table key reuse is now rejected |
| Functional verification separated from cryptographic reduction | done (§8) |
| Kernel-checked, no `sorry` or unreviewed axioms | 156 theorems; axioms ⊆ {`propext`, `Classical.choice`, `Quot.sound`} |
| Encodings and negative tests cross-checked against the implementation | 25 traces + regression pins |
| No claim of 256-bit security, collision resistance or global one-time use | none made |

Estimate: about **95%** of #41 for the trusted, durable, single-host model. Each public commitment
table is now single use (proved, and tested against the implementation). What remains: a refinement
proof from Python semantics (§7), and secret-value single use (§9.1), which is a further protocol
decision.
