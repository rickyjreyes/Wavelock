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
`b₁` is durably appended, no block sharing its `one_time_key_id`, its `public_key_fingerprint`
or its commitment table can even be durably consumed. -/
theorem python_one_winner (ad : Bool) (es₁ es₂ : List (Event CBlock)) {p q : ℕ} {o : WriteOutcome}
    {m₁ m₂ : List Atom} {b₁ b₂ : CBlock} (hadds : o.effect.adds = true)
    (h₁ : (run Discipline.implemented (pySpec .current) ad es₁).procs p = .recorded m₁ b₁)
    (h₂ : (run Discipline.implemented (pySpec .current) ad (es₁ ++ .writeBlock p o :: es₂)).procs q =
      .recorded m₂ b₂) :
    b₁.kid.render ≠ b₂.kid.render ∧ b₁.fp ≠ b₂.fp ∧ b₁.material ≠ b₂.material := by
  have h := at_most_one_acceptance (pySpec .current) (pySpec_acceptSound rfl)
    (Or.inl (pySpec_losslessRecon rfl)) es₁ es₂ hadds h₁ h₂
  have mem : ∀ (b : CBlock) (a : Atom), (a = .kid b.kid.render ∨ a = .fp b.fp ∨ a = .mat b.material) →
      a ∈ (pySpec .current).atoms b := by
    intro b a ha
    rcases ha with rfl | rfl | rfl <;> simp [pySpec, pyAtoms, PyRev.current]
  refine ⟨fun e => h _ (mem b₂ _ (Or.inl rfl)) (mem b₁ _ (Or.inl (by rw [e]))),
    fun e => h _ (mem b₂ _ (Or.inr (Or.inl rfl))) (mem b₁ _ (Or.inr (Or.inl (by rw [e])))),
    fun e => h _ (mem b₂ _ (Or.inr (Or.inr rfl))) (mem b₁ _ (Or.inr (Or.inr (by rw [e]))))⟩

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
    let r := runLog noLocks (pySpec .current) false Sys.init raceSchedule
    r.2 = [race1, race2] ∧ (pySpec .current).chainValid r.1.disk.chain = false := by
  decide

/-- The identical event list under the implemented discipline: process 1 cannot take the chain
lock while process 0 holds it, so only process 0 accepts. -/
theorem locked_same_schedule_single_accept :
    let r := runLog Discipline.implemented (pySpec .current) false Sys.init raceSchedule
    r.2 = [race1] ∧ r.1.disk.chain = [race1] := by
  decide

/-- Process 1 retrying after process 0 finished is rejected by the reloaded chain. -/
theorem locked_retry_rejected :
    let r := runLog Discipline.implemented (pySpec .current) false Sys.init
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
    let r := runLog { chainLock := false } (pySpec .current) false Sys.init
      [.begin 0 sib1, .begin 1 sib2, .ledger 0, .writeRecord 0 .ok, .ledger 1, .writeRecord 1 .ok,
       .writeBlock 0 .ok, .writeBlock 1 .ok]
    r.2 = [sib1, sib2] ∧ (pySpec .current).chainValid r.1.disk.chain = false ∧
      (kidCodes r.1.disk.chain).Nodup := by
  decide

/-- A payload signature consumed through `verify_ots_payload` (no chain). -/
def pay1 : CBlock := blk 0 0 0 (.str 43) 143 1430

/-- **Historical (Mythos M3, fixed before this change).**  With the locks but without re-reading
the replay file under the lock, a second ledger instance whose memory predates the first append
consumes the same identity again. -/
theorem historical_m3_no_rescan_double_consumes :
    let s := run { rescan := false } (pySpec .current) false
      [.payBegin 0 pay1, .payWrite 0 .ok, .payBegin 1 pay1, .payWrite 1 .ok]
    s.disk.replay.length = 2 ∧ replayUnique s.disk.replay = false := by
  decide

theorem current_rescan_rejects_second_consumer :
    let s := run Discipline.implemented (pySpec .current) false
      [.payBegin 0 pay1, .payWrite 0 .ok, .payBegin 1 pay1, .payWrite 1 .ok]
    s.disk.replay = [[.kid 43, .fp 143, .mat 1430]] := by
  decide

/-- The standalone payload path and block acceptance share one ledger (`OTS_LEDGER is
CONSENSUS_OTS_LEDGER`): an identity consumed as a payload cannot then authorise a block. -/
theorem payload_consumption_blocks_later_block :
    (run Discipline.implemented (pySpec .current) false
      ([.payBegin 0 pay1, .payWrite 0 .ok] ++ acceptAll 1 (blk 1 0 120 (.str 43) 143 1430))).disk.chain
      = [] := by
  decide

/-- The payload `pay1` relabelled by a third party: new key id and the fingerprint that recomputes,
same commitment table (`OTSVerification.kid_relabel_verifies` shows the relabelled signature
verifies). -/
def pay1Relabelled : CBlock := blk 0 0 0 (.str 44) 144 1430

/-- **PR #47 boundary.**  The standalone path consumed labels only: a relabelled replay of a
consumed signature was consumed again. -/
theorem pr47_payload_relabel_replayed :
    (run Discipline.implemented (pySpec .pr47) false
      [.payBegin 0 pay1, .payWrite 0 .ok, .payBegin 1 pay1Relabelled, .payWrite 1 .ok]).disk.replay.length
      = 2 := by
  decide

/-- **This change.**  The commitment table is consumed, so the relabelled replay is rejected. -/
theorem current_payload_relabel_rejected :
    (run Discipline.implemented (pySpec .current) false
      [.payBegin 0 pay1, .payWrite 0 .ok, .payBegin 1 pay1Relabelled, .payWrite 1 .ok]).disk.replay
      = [[.kid 43, .fp 143, .mat 1430]] := by
  decide

end OTS
end WaveLockProofs
