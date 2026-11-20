import WaveLockProofs.OTSKeyConsumption

/-!
# Same-host serialization of OTS acceptance: one winner, and why each lock is needed

`OTSKeyConsumption.at_most_one_acceptance` proves, for every interleaving of the *micro-steps* of
any number of processes, that once a block is durably appended no process can reach the
post-consumption phase for a block sharing either identity.  The proof uses only mutual exclusion
of the two `flock`s (`chain-writer.lock`, `<ledger>.lock`; SQLite `BEGIN EXCLUSIVE` where `fcntl`
is absent), the re-scan of the replay file under the ledger lock, and the reload/reconstruction of
the chain under the chain-writer lock.

This file instantiates that theorem for the Python model and shows, on concrete interleavings,
that removing any one of these elements breaks it.  The same event lists are run under the
implemented discipline, where blocked lock acquisitions are no-ops (the process would wait).

Scope: exclusion between threads and processes that share one filesystem with working `flock`
semantics.  Nothing here concerns different hosts or network filesystems.
-/

namespace WaveLockProofs
namespace OTS

/-- **One winner (Python model).**  Over any interleaving of any number of same-host processes,
with write failures, crashes, restarts and (if `ad`) loss of the side replay file: after a block
`b₁` is durably appended, no block sharing its `one_time_key_id` or its
`public_key_fingerprint` can even be durably consumed. -/
theorem python_one_winner (ad : Bool) (es₁ es₂ : List (Event CBlock)) {p q : ℕ} {o : WriteOutcome}
    {m₁ m₂ : Mem ℕ ℕ} {b₁ b₂ : CBlock} (hadds : o.effect.adds = true)
    (h₁ : (run Discipline.implemented (pySpec true) ad es₁).procs p = .recorded m₁ b₁)
    (h₂ : (run Discipline.implemented (pySpec true) ad (es₁ ++ .writeBlock p o :: es₂)).procs q =
      .recorded m₂ b₂) :
    b₁.kid.render ≠ b₂.kid.render ∧ b₁.fp ≠ b₂.fp :=
  at_most_one_acceptance (pySpec true) pySpec_acceptSound (Or.inl pySpec_losslessRecon)
    es₁ es₂ hadds h₁ h₂

/-- A copied secret key (same `one_time_key_id`, same fingerprint) signing two different bodies
for the same parent. -/
def race1 : CBlock := blk 1 0 101 (.str 40) 140 1400
def race2 : CBlock := blk 1 0 102 (.str 40) 140 1400

/-- Both processes check before either consumes. -/
def raceSchedule : List (Event CBlock) :=
  [.begin 0 race1, .begin 1 race2, .ledger 0, .ledger 1,
   .writeRecord 0 .ok, .writeRecord 1 .ok, .writeBlock 0 .ok, .writeBlock 1 .ok]

def noLocks : Discipline := { chainLock := false, ledgerLock := false }

/-- **Necessity of atomic check-then-consume.**  Without the locks, the interleaving above makes
both processes report acceptance of one identity; the chain then fails its own validator. -/
theorem unlocked_check_then_consume_double_accepts :
    let r := runLog noLocks (pySpec true) false Sys.init raceSchedule
    r.2 = [race1, race2] ∧ (pySpec true).chainValid r.1.disk.chain = false := by
  decide

/-- The identical event list under the implemented discipline: process 1 cannot take the chain
lock while process 0 holds it, so only process 0 accepts. -/
theorem locked_same_schedule_single_accept :
    let r := runLog Discipline.implemented (pySpec true) false Sys.init raceSchedule
    r.2 = [race1] ∧ r.1.disk.chain = [race1] := by
  decide

/-- Process 1 retrying after process 0 finished is rejected by the reloaded chain. -/
theorem locked_retry_rejected :
    let r := runLog Discipline.implemented (pySpec true) false Sys.init
      (raceSchedule ++ acceptAll 1 race2)
    r.2 = [race1] ∧ r.1.disk.chain = [race1] := by
  decide

/-- Distinct identities competing for the same parent. -/
def sib1 : CBlock := blk 1 0 111 (.str 41) 141 1410
def sib2 : CBlock := blk 1 0 112 (.str 42) 142 1420

/-- **Necessity of the chain-writer lock** (hardening report: "distinct-key sibling blocks").
With only the ledger lock, two blocks with different identities are both appended on one tip;
identities stay unique but linkage is broken and the stored chain is invalid. -/
theorem no_chain_lock_appends_siblings :
    let r := runLog { chainLock := false } (pySpec true) false Sys.init
      [.begin 0 sib1, .begin 1 sib2, .ledger 0, .writeRecord 0 .ok, .ledger 1, .writeRecord 1 .ok,
       .writeBlock 0 .ok, .writeBlock 1 .ok]
    r.2 = [sib1, sib2] ∧ (pySpec true).chainValid r.1.disk.chain = false ∧
      ((pySpec true).kids r.1.disk.chain).Nodup := by
  decide

/-- A payload signature consumed through `verify_ots_payload` (no chain). -/
def pay1 : CBlock := blk 0 0 0 (.str 43) 143 1430

/-- **Historical (Mythos M3, fixed before this change).**  With the locks but without re-reading
the replay file under the lock, a second ledger instance whose memory predates the first append
consumes the same identity again. -/
theorem historical_m3_no_rescan_double_consumes :
    let s := run { rescan := false } (pySpec true) false
      [.payBegin 0 pay1, .payWrite 0 .ok, .payBegin 1 pay1, .payWrite 1 .ok]
    s.disk.replay.length = 2 ∧ replayUnique s.disk.replay = false := by
  decide

theorem current_rescan_rejects_second_consumer :
    let s := run Discipline.implemented (pySpec true) false
      [.payBegin 0 pay1, .payWrite 0 .ok, .payBegin 1 pay1, .payWrite 1 .ok]
    s.disk.replay = [((RawKid.str 43).render, 143)] := by
  decide

/-- The standalone payload path and block acceptance share one ledger (`OTS_LEDGER is
CONSENSUS_OTS_LEDGER`): an identity consumed as a payload cannot then authorise a block. -/
theorem payload_consumption_blocks_later_block :
    (run Discipline.implemented (pySpec true) false
      ([.payBegin 0 pay1, .payWrite 0 .ok] ++ acceptAll 1 (blk 1 0 120 (.str 43) 143 1430))).disk.chain
      = [] := by
  decide

end OTS
end WaveLockProofs
