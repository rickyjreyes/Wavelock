import WaveLockProofs.OTSKeyConsumption

/-!
# Kernel-checked counterexample traces for WaveLock-OTS identity consumption

Each theorem evaluates the executable model of `OTSKeyConsumption` on a concrete trace with
`decide`.  Traces against `pySpec .b310ea5` reproduce behaviour of `master` before this work,
traces against `pySpec .pr47` that of PR #47 before the key-material change, and traces against
`pySpec .current` the corrected acceptance (each class also replayed against the Python
implementation, see `tests/test_ots_formal_conformance.py`).
The remaining traces show why the hypotheses of `chain_single_use` cannot be dropped:
shared authoritative state, a trusted (non-rolled-back) store, and lossless reconstruction.

Classification used in `docs/WAVELOCK_OTS_FORMAL_VERIFICATION.md`:
* *present defect* — reproduced against `b310ea5`, corrected in this change;
* *boundary* — present behaviour outside the stated guarantee (not corrected: a correction
  would be a protocol change);
* *historical* — a weaker model matching code that was already fixed (Mythos M2/M3).
-/

namespace WaveLockProofs
namespace OTS

/-! ## Present defects at `b310ea5` (corrected here) -/

/-- An honest block on the empty chain and on top of a one-block chain with hash `h`. -/
def honest1 : CBlock := blk 1 0 13 (.str 7) 60 600

section EmptyKid
def bEmpty : CBlock := blk 1 0 11 (.str 0) 50 500
def bAfterEmpty : CBlock := blk 2 11 12 (.str 7) 60 600
def emptyTrace : List (Event CBlock) := acceptAll 0 bEmpty ++ acceptAll 0 bAfterEmpty

/-- **Present defect (DoS).**  At `b310ea5` a block whose `one_time_key_id` is `""` is accepted;
its replay record violates the loader's own schema, so the replay file never loads again and
every later acceptance (here, an honest next block) is rejected. -/
theorem legacy_empty_key_id_bricks_replay_ledger :
    let s := run Discipline.implemented (pySpec .b310ea5) false emptyTrace
    s.disk.chain = [bEmpty] ∧ (pySpec .b310ea5).replayLoads s.disk = false := by
  decide

/-- With the correction the `""` key id is rejected and the node stays healthy. -/
theorem fixed_empty_key_id_rejected :
    let s := run Discipline.implemented (pySpec .current) false (acceptAll 0 bEmpty ++ acceptAll 0 honest1)
    s.disk.chain = [honest1] ∧ (pySpec .current).replayLoads s.disk = true ∧
      (pySpec .current).chainLoads s.disk = true := by
  decide
end EmptyKid

section NullKid
def bNull : CBlock := blk 1 0 21 .null 51 501
def bAfterNull : CBlock := blk 2 21 22 (.str 7) 60 600

/-- **Present defect (DoS).**  A `null` key id passes `verify_ots` (both sides are `None`), is
recorded as the string `"None"`, and is appended; `_reconstruct_consumed_ots` then raises on every
reload, so the chain never loads again. -/
theorem legacy_null_key_id_bricks_chain_reload :
    let s := run Discipline.implemented (pySpec .b310ea5) false (acceptAll 0 bNull ++ acceptAll 0 bAfterNull)
    s.disk.chain = [bNull] ∧ (pySpec .b310ea5).chainLoads s.disk = false := by
  decide

theorem fixed_null_key_id_rejected :
    let s := run Discipline.implemented (pySpec .current) false (acceptAll 0 bNull ++ acceptAll 0 honest1)
    s.disk.chain = [honest1] ∧ (pySpec .current).chainLoads s.disk = true := by
  decide
end NullKid

section ListKid
def bList : CBlock := blk 1 0 31 (.arr 1) 52 502

/-- **Present defect (DoS).**  An array key id is accepted; `verify_ots_chain` then raises
`TypeError` (unhashable) on every reload, i.e. the stored chain is no longer valid. -/
theorem legacy_list_key_id_breaks_chain_validation :
    let s := run Discipline.implemented (pySpec .b310ea5) false (acceptAll 0 bList)
    s.disk.chain = [bList] ∧ (pySpec .b310ea5).chainValid s.disk.chain = false := by
  decide

theorem fixed_list_key_id_rejected :
    (run Discipline.implemented (pySpec .current) false (acceptAll 0 bList)).disk.chain = [] := by
  decide
end ListKid

section NoneSentinel
def bNone1 : CBlock := blk 1 0 41 (.str 1) 53 503
def bNone2 : CBlock := blk 2 41 42 (.str 1) 54 504
/-- Accept a block keyed `"None"`, lose the side replay file, restart, accept a *different* key
that reuses the key id `"None"`. -/
def noneTrace : List (Event CBlock) :=
  acceptAll 0 bNone1 ++ [.deleteReplay, .crash 0, .restart 0] ++ acceptAll 0 bNone2

/-- **Present defect (single-use violation).**  `index_signature` skipped every key id whose
string form is `"None"`, so reconstruction lost the *string* key id `"None"`.  Under the Mythos M2
recovery model (side file lost, restart) a second block with the same `one_time_key_id` is
accepted: the chain holds that key id twice and fails its own validator. -/
theorem legacy_none_sentinel_double_accepts_key_id :
    let s := run Discipline.implemented (pySpec .b310ea5) true noneTrace
    s.disk.chain = [bNone2, bNone1] ∧ ¬ (kidCodes s.disk.chain).Nodup ∧
      (pySpec .b310ea5).chainValid s.disk.chain = false := by
  decide

/-- The precise failure: reconstruction is not lossless at `b310ea5`. -/
theorem legacy_reconstruction_not_lossless : ¬ (pySpec .b310ea5).LosslessRecon := by
  intro h
  have := h [] bNone1 (by decide) (.kid 1) (by decide)
  revert this; decide

theorem fixed_none_sentinel_second_use_rejected :
    (run Discipline.implemented (pySpec .current) true noneTrace).disk.chain = [bNone1] := by
  decide

/-- Without loss of the side file the legacy code is safe on the same schedule: the replay file
holds the key id.  This is the `LosslessRecon ∨ ad = false` hypothesis of `chain_single_use`. -/
theorem legacy_none_sentinel_safe_without_file_loss :
    (run Discipline.implemented (pySpec .b310ea5) false noneTrace).disk.chain = [bNone1] := by
  decide
end NoneSentinel

/-! ## Historical defect (Mythos M2): no reconstruction from the chain -/

/-- The M2-era behaviour: nothing is folded back from the accepted chain. -/
def noReconSpec : AcceptSpec CBlock Atom := { pySpec .current with recon := fun _ => [] }

def bM2 : CBlock := blk 1 0 71 (.str 20) 80 800
/-- The same one-time identity replayed in a fresh block at height 2. -/
def bM2replay : CBlock := blk 2 71 72 (.str 20) 80 800

def m2Trace : List (Event CBlock) :=
  acceptAll 0 bM2 ++ [.deleteReplay, .crash 0, .restart 0] ++ acceptAll 0 bM2replay

/-- **Historical (M2, fixed before this change).**  Without reconstruction, deleting the side
file and restarting reopens replay of the identical one-time identity. -/
theorem historical_m2_replay_after_side_file_loss :
    (run Discipline.implemented noReconSpec true m2Trace).disk.chain = [bM2replay, bM2] := by
  decide

theorem current_m2_replay_rejected :
    (run Discipline.implemented (pySpec .current) true m2Trace).disk.chain = [bM2] := by
  decide

/-! ## Crash between consumption and append -/

def bBurn : CBlock := blk 1 0 91 (.str 30) 95 950

/-- Crash after the replay record is durable but before `save_block_to_disk`: the identity is
burned (consumed, never accepted) and stays consumed across restart; it is never accepted. -/
theorem crash_after_record_burns_identity :
    let s := run Discipline.implemented (pySpec .current) false
      ([.begin 0 bBurn, .ledger 0, .writeRecord 0 .ok, .crash 0, .restart 0] ++ acceptAll 0 bBurn)
    s.disk.chain = [] ∧ s.disk.replay = [[.kid 30, .fp 95, .mat 950]] := by
  decide

/-- Under the recovery model the burned identity is forgotten when the side file is lost, and is
then accepted once (issue #25): reconstruction recovers accepted identities, not burned ones. -/
theorem burned_identity_lost_with_side_file :
    (run Discipline.implemented (pySpec .current) true
      ([.begin 0 bBurn, .ledger 0, .writeRecord 0 .ok, .crash 0, .deleteReplay, .restart 0] ++
        acceptAll 0 bBurn)).disk.chain = [bBurn] := by
  decide

/-- An I/O error after the block line landed: `try_accept_block` reports rejection, yet the block
is durably accepted (the return value is not authoritative); a resubmission is rejected. -/
theorem written_then_error_is_durable :
    let es := acceptWith 0 bBurn .ok (.fail .line) ++ acceptAll 0 bBurn
    (runLog Discipline.implemented (pySpec .current) false Sys.init es).2 = [] ∧
      (run Discipline.implemented (pySpec .current) false es).disk.chain = [bBurn] := by
  decide

/-- A torn replay record (power loss mid-line) leaves the node permanently fail-closed: no later
acceptance and no restart, until an operator intervenes. -/
theorem torn_record_fail_closed :
    let s := run Discipline.implemented (pySpec .current) false
      ([.begin 0 bBurn, .ledger 0, .writeRecord 0 (.crash .torn), .restart 0] ++ acceptAll 1 honest1)
    s.disk.chain = [] ∧ (pySpec .current).replayLoads s.disk = false := by
  decide

/-- A record that lost only its newline still loads, but the next append merges two lines and the
file never loads again. -/
theorem record_without_newline_poisons_next_append :
    let s := run Discipline.implemented (pySpec .current) false
      ([.begin 0 bBurn, .ledger 0, .writeRecord 0 (.crash .noNewline), .restart 0] ++ acceptAll 0 honest1)
    s.disk.chain = [honest1] ∧ (pySpec .current).replayLoads s.disk = false := by
  decide

/-! ## Key material: a boundary in PR #47, enforced by this change -/

def bMat1 : CBlock := blk 1 0 61 (.str 8) 70 900
/-- The holder of the same secret seed mints a fresh `one_time_key_id`; the fingerprint, which
covers the key id, changes too; the commitment table (material `900`) is unchanged. -/
def bMat2 : CBlock := blk 2 61 62 (.str 9) 71 900

/-- **PR #47 boundary (reproduced against that revision).**  Two blocks over the *same*
commitment table are both accepted under distinct key ids: the identities consumed in PR #47
are labels chosen by the key holder. -/
theorem copied_key_material_accepted_twice :
    let c := (run Discipline.implemented (pySpec .pr47) false (acceptAll 0 bMat1 ++ acceptAll 0 bMat2)).disk.chain
    c = [bMat2, bMat1] ∧ ¬ (materialCodes c).Nodup := by
  decide

/-- **This change.**  Consuming the commitment table rejects the second block, whatever labels it
carries; the chain keeps one block per table. -/
theorem copied_key_material_rejected :
    let c := (run Discipline.implemented (pySpec .current) false (acceptAll 0 bMat1 ++ acceptAll 0 bMat2)).disk.chain
    c = [bMat1] ∧ (materialCodes c).Nodup := by
  decide

/-- The same holds after the side replay file is lost: the material is reconstructed from the
chain. -/
theorem copied_key_material_rejected_after_side_file_loss :
    (run Discipline.implemented (pySpec .current) true
      (acceptAll 0 bMat1 ++ [.deleteReplay, .crash 0, .restart 0] ++ acceptAll 0 bMat2)).disk.chain
      = [bMat1] := by
  decide

/-! ## Boundary: independent nodes -/

def bFork1 : CBlock := blk 1 0 81 (.str 10) 90 910
/-- The same one-time identity authorising a different body at the same height. -/
def bFork2 : CBlock := blk 1 0 82 (.str 10) 90 910

/-- **Boundary.**  Two nodes without shared authoritative consumption state each accept a
different block under one identity.  Each local chain is valid; together they use it twice. -/
theorem independent_nodes_both_accept :
    let a := (run Discipline.implemented (pySpec .current) false (acceptAll 0 bFork1)).disk.chain
    let b := (run Discipline.implemented (pySpec .current) false (acceptAll 0 bFork2)).disk.chain
    a = [bFork1] ∧ b = [bFork2] ∧ (pySpec .current).chainValid a = true ∧
      (pySpec .current).chainValid b = true ∧ ¬ (kidCodes (a ++ b)).Nodup ∧
      ¬ (materialCodes (a ++ b)).Nodup := by
  decide

/-! ## Boundary: hostile rollback without a trusted anchor -/

/-- Environment with a storage adversary that can restore an older snapshot. -/
inductive HEvent where
  | ev (e : Event CBlock)
  | restoreChain (c : List CBlock)
  | restoreAll (d : Disk CBlock Atom)

def hstep (S : AcceptSpec CBlock Atom) (ad : Bool) (acc : Sys CBlock Atom × List CBlock) :
    HEvent → Sys CBlock Atom × List CBlock
  | .ev e => (step Discipline.implemented S ad acc.1 e, acc.2 ++ (acceptedBlock? acc.1 e).toList)
  | .restoreChain c => ({ acc.1 with disk := { acc.1.disk with chain := c } }, acc.2)
  | .restoreAll d => ({ acc.1 with disk := d }, acc.2)

def hrun (S : AcceptSpec CBlock Atom) (ad : Bool) (es : List HEvent) : Sys CBlock Atom × List CBlock :=
  es.foldl (hstep S ad) (Sys.init, [])

def emptyDisk : Disk CBlock Atom := ⟨[], .clean, [], .clean⟩

/-- **Boundary.**  Restoring a complete earlier snapshot (chain *and* replay file) and restarting
lets the same one-time identity be accepted again: over the node's history it is accepted twice.
Nothing local distinguishes the restored state from a genuine one. -/
theorem rollback_reaccepts_identity :
    let r := hrun (pySpec .current) false
      ((acceptAll 0 bFork1).map .ev ++ [.restoreAll emptyDisk, .ev (.crash 0), .ev (.restart 0)] ++
        (acceptAll 0 bFork2).map .ev)
    r.2 = [bFork1, bFork2] ∧ r.1.disk.chain = [bFork2] := by
  decide

/-- Truncating only the chain to a valid prefix is undetected (the accepted block is lost), but the
intact replay file still refuses the identity. -/
theorem chain_only_rollback_still_rejects_replay :
    let r := hrun (pySpec .current) false
      ((acceptAll 0 bFork1).map .ev ++ [.restoreChain [], .ev (.crash 0), .ev (.restart 0)] ++
        (acceptAll 0 bFork2).map .ev)
    r.2 = [bFork1] ∧ r.1.disk.chain = [] := by
  decide

end OTS
end WaveLockProofs
