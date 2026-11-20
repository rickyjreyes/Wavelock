import Mathlib.Tactic

/-!
# WaveLock-OTS identity consumption: same-host block acceptance as a transition system

Frozen source: `rickyjreyes/Wavelock` master `b310ea5`, with the corrections of PR #47
(identity typing; the `index_signature` sentinel) and of this change (consumption of the
commitment table, `wavelock_ots.ots_material_id`).  Protocol constants: scheme
`WaveLock-OTS-v1`, signature/public-key `version = 1`, replay `LEDGER_VERSION = 2` (version 1
records still load), block transcript `OTS_BLOCK_TRANSCRIPT_VERSION = 1`, block header version 2
required on the OTS path.

## Identities as atoms

Acceptance consumes a finite list of *identity atoms* per signature.  At `b310ea5` and in PR #47
these are the two labels `one_time_key_id` and `public_key_fingerprint`; in this change also the
commitment table's Merkle root.  The model is generic in the atoms: every theorem below says that
no atom is consumed twice, for whatever atoms a revision consumes.  `pySpec` fixes the atoms of
each revision (`PyRev.b310ea5`, `PyRev.pr47`, `PyRev.current`).

## What is modelled (Python reference)

* `wavelock/network/server.py :: try_accept_block` under `ChainState.acceptance_lock`
  (`chain-writer.lock`, `fcntl.flock` / SQLite): reload `load_all_blocks`, `verify_ots_chain`,
  `_reconstruct_consumed_ots`, `_verify_pow_and_linkage`, `_verify_ots_block`, then
  `CHAIN.append` → `save_block_to_disk`.  Event `begin` (lock + reload + pure checks),
  `ledger`, `writeRecord`, `writeBlock`.
* `wavelock/crypto/ots_ledger.py :: PersistentOTSReplayLedger.accept`: under the ledger lock
  (`<path>.lock`), `_scan_file_into_sets` (additive re-scan, fails closed on a malformed record),
  the duplicate test on every consumed atom, `_append` (`write; flush; fsync; fsync(dir)`), then
  the in-memory update.  Event `ledger` and `writeRecord`; the standalone
  `server.verify_ots_payload` path is `payBegin`/`payWrite`.
* Process death (the kernel releases `flock`s), process start (`PersistentOTSReplayLedger()`
  load + `CHAIN.load_from_disk`), and loss of the side replay file (the Mythos M2 recovery
  model).

Pure checks (signature verification, transcript binding, linkage, proof of work) enter as the
parameters of `AcceptSpec`; the theorems below hold for *every* choice of them that satisfies the
named hypotheses.  The cryptographic layer is treated in `OTSVerification` and
`AuthenticatedBlockBinding`.

## Named omissions (and their effect)

* A blocked lock acquisition is an event that leaves the state unchanged (the scheduler retries).
* Acquiring a free lock and immediately reading the protected file is one event: no other
  process can write that file in between, and an external deletion commutes with the read.
* Threads of one process are serialized by `RLock` before the `flock`; they are not separated.
* `fsync` is assumed honest (a completed write survives).  Storage that loses acknowledged writes,
  or is edited by a hostile party, is *outside* this system; see `KeyReuseCounterexample`.
* Atom values are the strings the replay ledger stores (`str(...)` of the signature fields and
  the recomputed root); the concrete instance records where Python's raw-value and string views
  differ.
-/

namespace WaveLockProofs
namespace OTS

/-! ## Durable append-only files -/

/-- State of the end of an append-only JSONL file. -/
inductive Tail where
  /-- every line is complete and newline-terminated -/
  | clean
  /-- the last record is complete JSON but lacks its newline (it still parses) -/
  | pending
  /-- an unparseable partial line exists: every later load raises -/
  | garbage
  deriving DecidableEq, Repr

/-- Bytes that reached the file during one append. -/
inductive Effect where
  /-- nothing reached the file -/
  | nothing
  /-- one complete, newline-terminated line -/
  | line
  /-- an unparseable partial line (torn write) -/
  | torn
  /-- the complete record without its newline -/
  | noNewline
  deriving DecidableEq, Repr

/-- Outcome of one `write; flush; fsync; fsync(dir)` append (`ots_ledger._append`,
`chain_utils.save_block_to_disk`).  `fail` raises into the caller, which reports rejection;
`crash` kills the process. -/
inductive WriteOutcome where
  | ok
  | fail (e : Effect)
  | crash (e : Effect)
  deriving DecidableEq, Repr

namespace Effect

/-- The appended object is part of the file's logical content (the loader sees it, or the file
no longer loads at all). -/
def adds : Effect → Bool
  | line | noNewline => true
  | nothing | torn => false

/-- Appending a complete line right after a `pending` record merges both into one bad line. -/
def tail : Effect → Tail → Tail
  | nothing, t => t
  | torn, _ => .garbage
  | line, .clean => .clean
  | line, _ => .garbage
  | noNewline, .clean => .pending
  | noNewline, _ => .garbage

end Effect

/-- The effect of an outcome. -/
def WriteOutcome.effect : WriteOutcome → Effect
  | ok => .line
  | fail e => e
  | crash e => e

/-- Durable state shared by every process on the host.  Lists are newest-first; a replay record
is the list of identity atoms it consumes. -/
structure Disk (B A : Type) where
  chain : List B
  chainTail : Tail
  replay : List (List A)
  replayTail : Tail

/-- Atoms consumed by the records of the replay file (`_scan_file_into_sets`). -/
def fileAtoms {A : Type} (rs : List (List A)) : List A := rs.flatten

/-! ## Pure parameters of the protocol -/

/-- The stateless part of acceptance.  `atoms b` are the identities a signature consumes (as the
replay ledger stores them); `linkOK`/`blockOK` are the per-block conditions of
`verify_ots_chain`; `acceptOK` is every pure check performed by `try_accept_block` and by the
ledger before its duplicate test; `recon` is `index_signature`; `reconFail` is the
`_reconstruct_consumed_ots` raise condition; `recordOK` is the per-record schema of
`_scan_file_into_sets`. -/
structure AcceptSpec (B A : Type) where
  atoms : B → List A
  blockOK : B → Bool
  linkOK : List B → B → Bool
  acceptOK : List B → B → Bool
  payOK : B → Bool
  recon : B → List A
  reconFail : B → Bool
  recordOK : List A → Bool

namespace AcceptSpec

variable {B A : Type} [DecidableEq A] (S : AcceptSpec B A)

/-- Every atom of every block of a chain. -/
def chainAtoms (c : List B) : List A := c.flatMap S.atoms

/-- No atom of `b` occurs in the chain `c`. -/
def fresh (c : List B) (b : B) : Bool := (S.atoms b).all fun a => !decide (a ∈ S.chainAtoms c)

/-- Some atom of `b` is in the consumed set `m` (the ledger's duplicate test). -/
def hits (m : List A) (b : B) : Bool := (S.atoms b).any fun a => decide (a ∈ m)

/-- `verify_ots_chain` on a newest-first chain: linkage, per-block verification and no atom
shared between two blocks. -/
def chainValid : List B → Bool
  | [] => true
  | b :: c => chainValid c && S.linkOK c b && S.blockOK b && S.fresh c b

/-- `_reconstruct_consumed_ots`: fold `index_signature` over the chain. -/
def reconAll (c : List B) : List A := c.flatMap S.recon

/-- `CHAIN.load_from_disk(require_ots=True)` succeeds. -/
def chainLoads (d : Disk B A) : Bool :=
  d.chainTail != .garbage && S.chainValid d.chain && d.chain.all fun b => !S.reconFail b

/-- `_scan_file_into_sets` succeeds. -/
def replayLoads (d : Disk B A) : Bool :=
  d.replayTail != .garbage && d.replay.all S.recordOK

/-- Authoritative consumed atoms of a durable state: accepted chain plus replay records. -/
def consumed (d : Disk B A) : List A := S.chainAtoms d.chain ++ fileAtoms d.replay

/-- Every pure acceptance check implies the chain validator's per-block conditions. -/
def AcceptSound : Prop := ∀ c b, S.acceptOK c b = true → S.linkOK c b = true ∧ S.blockOK b = true

/-- `index_signature` folds back every atom of every block that acceptance admits. -/
def LosslessRecon : Prop := ∀ c b, S.acceptOK c b = true → ∀ a ∈ S.atoms b, a ∈ S.recon b

/-- Acceptance never writes a replay record that the loader rejects. -/
def RecordAdmissible : Prop :=
  (∀ c b, S.acceptOK c b = true → S.recordOK (S.atoms b) = true) ∧
    ∀ b, S.payOK b = true → S.recordOK (S.atoms b) = true

/-- Reconstruction never raises on a block that acceptance admits. -/
def ReconTotal : Prop := ∀ c b, S.acceptOK c b = true → S.reconFail b = false

end AcceptSpec

/-- No record of the replay file repeats an atom of an older record. -/
def replayUnique {A : Type} [DecidableEq A] : List (List A) → Bool
  | [] => true
  | r :: rs => replayUnique rs && r.all fun a => !decide (a ∈ fileAtoms rs)

/-! ## Processes, locks and events -/

/-- Control state of one process; `m` is its in-memory consumed set.  `loaded`/`checked`/
`recorded` hold the chain-writer lock; `checked`/`payChecked` hold the ledger lock. -/
inductive Phase (B A : Type) where
  | dead
  | idle (m : List A)
  | loaded (m : List A) (b : B)
  | checked (m : List A) (b : B)
  | recorded (m : List A) (b : B)
  | payChecked (m : List A) (b : B)

namespace Phase
variable {B A : Type}

def holdsChain : Phase B A → Bool
  | loaded .. | checked .. | recorded .. => true
  | _ => false

def holdsLedger : Phase B A → Bool
  | checked .. | payChecked .. => true
  | _ => false

end Phase

structure Sys (B A : Type) where
  disk : Disk B A
  procs : ℕ → Phase B A
  chainLock : Option ℕ
  ledgerLock : Option ℕ

/-- Which synchronisation the implementation performs.  `implemented` is the frozen code; the
other settings exist only to show that each element is necessary. -/
structure Discipline where
  chainLock : Bool := true
  ledgerLock : Bool := true
  rescan : Bool := true

def Discipline.implemented : Discipline := {}

/-- Steps of one process, or of the environment. -/
inductive Event (B : Type) where
  /-- `try_accept_block(b)`: take `chain-writer.lock`, reload, reconstruct, pure checks -/
  | begin (p : ℕ) (b : B)
  /-- ledger `accept`: take the ledger lock, re-scan the replay file, duplicate test -/
  | ledger (p : ℕ)
  /-- `_append` of the replay record -/
  | writeRecord (p : ℕ) (o : WriteOutcome)
  /-- `save_block_to_disk` -/
  | writeBlock (p : ℕ) (o : WriteOutcome)
  /-- `verify_ots_payload(...)` on the shared ledger (no chain) -/
  | payBegin (p : ℕ) (b : B)
  | payWrite (p : ℕ) (o : WriteOutcome)
  /-- process death; the kernel drops its `flock`s -/
  | crash (p : ℕ)
  /-- process start: ledger load, `load_from_disk`, reconstruction -/
  | restart (p : ℕ)
  /-- the side replay file is deleted or lost (Mythos M2 recovery model) -/
  | deleteReplay

def canTake (enabled : Bool) (l : Option ℕ) : Bool := !enabled || l.isNone

def take (enabled : Bool) (p : ℕ) (l : Option ℕ) : Option ℕ := if enabled then some p else l

def release (p : ℕ) (l : Option ℕ) : Option ℕ := if l = some p then none else l

namespace Sys
variable {B A : Type}

def set (s : Sys B A) (p : ℕ) (v : Phase B A) : ℕ → Phase B A :=
  fun q => if q = p then v else s.procs q

def init : Sys B A := ⟨⟨[], .clean, [], .clean⟩, fun _ => .idle [], none, none⟩

end Sys

section Step
variable {B A : Type} [DecidableEq A]

/-- One step.  Events whose process is not in the matching phase, or whose lock is busy, leave
the state unchanged. -/
def step (D : Discipline) (S : AcceptSpec B A) (allowDelete : Bool) (s : Sys B A) :
    Event B → Sys B A
  | .begin p b =>
    match s.procs p with
    | .idle m =>
      if canTake D.chainLock s.chainLock then
        if S.chainLoads s.disk then
          let m' := m ++ S.reconAll s.disk.chain
          if S.acceptOK s.disk.chain b then
            { s with procs := s.set p (.loaded m' b), chainLock := take D.chainLock p s.chainLock }
          else { s with procs := s.set p (.idle m') }
        else s
      else s
    | _ => s
  | .ledger p =>
    match s.procs p with
    | .loaded m b =>
      if canTake D.ledgerLock s.ledgerLock then
        if D.rescan && !S.replayLoads s.disk then
          { s with procs := s.set p (.idle m), chainLock := release p s.chainLock }
        else
          let m' := if D.rescan then m ++ fileAtoms s.disk.replay else m
          if S.hits m' b then
            { s with procs := s.set p (.idle m'), chainLock := release p s.chainLock }
          else
            { s with procs := s.set p (.checked m' b), ledgerLock := take D.ledgerLock p s.ledgerLock }
      else s
    | _ => s
  | .writeRecord p o =>
    match s.procs p with
    | .checked m b =>
      let d' : Disk B A :=
        { s.disk with
          replay := if o.effect.adds then S.atoms b :: s.disk.replay else s.disk.replay
          replayTail := o.effect.tail s.disk.replayTail }
      match o with
      | .ok =>
        { s with disk := d', procs := s.set p (.recorded (S.atoms b ++ m) b),
                 ledgerLock := release p s.ledgerLock }
      | .fail _ =>
        { s with disk := d', procs := s.set p (.idle m), ledgerLock := release p s.ledgerLock,
                 chainLock := release p s.chainLock }
      | .crash _ =>
        { s with disk := d', procs := s.set p .dead, ledgerLock := release p s.ledgerLock,
                 chainLock := release p s.chainLock }
    | _ => s
  | .writeBlock p o =>
    match s.procs p with
    | .recorded m b =>
      let d' : Disk B A :=
        { s.disk with
          chain := if o.effect.adds then b :: s.disk.chain else s.disk.chain
          chainTail := o.effect.tail s.disk.chainTail }
      match o with
      | .ok => { s with disk := d', procs := s.set p (.idle m), chainLock := release p s.chainLock }
      | .fail _ => { s with disk := d', procs := s.set p (.idle m), chainLock := release p s.chainLock }
      | .crash _ => { s with disk := d', procs := s.set p .dead, chainLock := release p s.chainLock }
    | _ => s
  | .payBegin p b =>
    match s.procs p with
    | .idle m =>
      if S.payOK b then
        if canTake D.ledgerLock s.ledgerLock then
          if D.rescan && !S.replayLoads s.disk then s
          else
            let m' := if D.rescan then m ++ fileAtoms s.disk.replay else m
            if S.hits m' b then { s with procs := s.set p (.idle m') }
            else
              { s with procs := s.set p (.payChecked m' b),
                       ledgerLock := take D.ledgerLock p s.ledgerLock }
        else s
      else s
    | _ => s
  | .payWrite p o =>
    match s.procs p with
    | .payChecked m b =>
      let d' : Disk B A :=
        { s.disk with
          replay := if o.effect.adds then S.atoms b :: s.disk.replay else s.disk.replay
          replayTail := o.effect.tail s.disk.replayTail }
      match o with
      | .ok => { s with disk := d', procs := s.set p (.idle (S.atoms b ++ m)),
                        ledgerLock := release p s.ledgerLock }
      | .fail _ => { s with disk := d', procs := s.set p (.idle m), ledgerLock := release p s.ledgerLock }
      | .crash _ => { s with disk := d', procs := s.set p .dead, ledgerLock := release p s.ledgerLock }
    | _ => s
  | .crash p =>
    { s with procs := s.set p .dead, chainLock := release p s.chainLock,
             ledgerLock := release p s.ledgerLock }
  | .restart p =>
    match s.procs p with
    | .dead =>
      if S.replayLoads s.disk && S.chainLoads s.disk then
        { s with procs := s.set p (.idle (fileAtoms s.disk.replay ++ S.reconAll s.disk.chain)) }
      else s
    | _ => s
  | .deleteReplay =>
    if allowDelete then { s with disk := { s.disk with replay := [], replayTail := .clean } } else s

/-- Execute a finite trace from the empty host. -/
def run (D : Discipline) (S : AcceptSpec B A) (allowDelete : Bool) (es : List (Event B)) :
    Sys B A :=
  es.foldl (step D S allowDelete) Sys.init

end Step

/-! ## The safety invariant -/

section Invariant
variable {B A : Type} [DecidableEq A] (S : AcceptSpec B A)

/-- Every chain atom is in the process's memory, or — when the side replay file cannot be lost
(`ad = false`) — in the replay file. -/
def Covers (ad : Bool) (m : List A) (d : Disk B A) : Prop :=
  ∀ a ∈ S.chainAtoms d.chain, a ∈ m ∨ (ad = false ∧ a ∈ fileAtoms d.replay)

/-- What a process in each phase knows about the current durable state. -/
def ProcOK (ad : Bool) (d : Disk B A) : Phase B A → Prop
  | .dead => True
  | .idle _ => True
  | .loaded m b => S.acceptOK d.chain b = true ∧ Covers S ad m d
  | .checked _ b => S.acceptOK d.chain b = true ∧ S.fresh d.chain b = true ∧
      ∀ a ∈ S.atoms b, a ∉ fileAtoms d.replay
  | .recorded _ b => S.acceptOK d.chain b = true ∧ S.fresh d.chain b = true ∧
      (ad = false → ∀ a ∈ S.atoms b, a ∈ fileAtoms d.replay)
  | .payChecked _ b => ∀ a ∈ S.atoms b, a ∉ fileAtoms d.replay

/-- The inductive invariant of the implemented system. -/
structure Inv (ad : Bool) (s : Sys B A) : Prop where
  chain_valid : S.chainValid s.disk.chain = true
  replay_unique : replayUnique s.disk.replay = true
  chain_admitted : ∀ x ∈ s.disk.chain, ∃ c, S.acceptOK c x = true
  chain_in_replay : ad = false → ∀ a ∈ S.chainAtoms s.disk.chain, a ∈ fileAtoms s.disk.replay
  chain_lock : ∀ p, s.chainLock = some p ↔ (s.procs p).holdsChain = true
  ledger_lock : ∀ p, s.ledgerLock = some p ↔ (s.procs p).holdsLedger = true
  procs_ok : ∀ p, ProcOK S ad s.disk (s.procs p)

/-! ### Small lemmas -/

omit [DecidableEq A] in
theorem release_eq_some {p q : ℕ} {l : Option ℕ} : release p l = some q ↔ l = some q ∧ q ≠ p := by
  unfold release
  split_ifs with h
  · subst h
    simp only [false_iff, not_and, not_not, Option.some.injEq]
    exact fun h => h.symm
  · constructor
    · intro hq
      refine ⟨hq, fun hqp => h ?_⟩
      rw [hq, hqp]
    · exact fun h => h.1

omit [DecidableEq A] in
theorem Sys.set_self {s : Sys B A} {p : ℕ} {v : Phase B A} : s.set p v p = v := by
  simp [Sys.set]

omit [DecidableEq A] in
theorem Sys.set_ne {s : Sys B A} {p q : ℕ} (h : q ≠ p) {v : Phase B A} :
    s.set p v q = s.procs q := by
  simp [Sys.set, h]

omit [DecidableEq A] in
/-- Lock bookkeeping after one process changes phase. -/
theorem lock_after_set {procs : ℕ → Phase B A} {f : Phase B A → Bool} {l l' : Option ℕ}
    {p : ℕ} {v : Phase B A}
    (hl : ∀ q, l = some q ↔ f (procs q) = true)
    (hp : l' = some p ↔ f v = true)
    (hother : ∀ q, q ≠ p → (l' = some q ↔ l = some q)) :
    ∀ q, l' = some q ↔ f (if q = p then v else procs q) = true := by
  intro q
  by_cases hq : q = p
  · subst hq; simpa using hp
  · simp only [hq, if_false]; rw [hother q hq]; exact hl q

omit [DecidableEq A] in
theorem mem_chainAtoms {c : List B} {a : A} : a ∈ S.chainAtoms c ↔ ∃ x ∈ c, a ∈ S.atoms x := by
  simp [AcceptSpec.chainAtoms]

omit [DecidableEq A] in
theorem mem_fileAtoms {rs : List (List A)} {a : A} : a ∈ fileAtoms rs ↔ ∃ r ∈ rs, a ∈ r := by
  simp [fileAtoms]

theorem fresh_iff {c : List B} {b : B} :
    S.fresh c b = true ↔ ∀ a ∈ S.atoms b, a ∉ S.chainAtoms c := by
  simp [AcceptSpec.fresh]

theorem hits_eq_false {m : List A} {b : B} : S.hits m b = false ↔ ∀ a ∈ S.atoms b, a ∉ m := by
  simp [AcceptSpec.hits]

theorem chainValid_cons {c : List B} {b : B} :
    S.chainValid (b :: c) = true ↔
      S.chainValid c = true ∧ S.linkOK c b = true ∧ S.blockOK b = true ∧ S.fresh c b = true := by
  simp [AcceptSpec.chainValid, Bool.and_eq_true, and_assoc]

theorem replayUnique_cons {r : List A} {rs : List (List A)} :
    replayUnique (r :: rs) = true ↔ replayUnique rs = true ∧ ∀ a ∈ r, a ∉ fileAtoms rs := by
  simp [replayUnique, Bool.and_eq_true]

omit [DecidableEq A] in
theorem mem_reconAll {c : List B} {x : B} (hx : x ∈ c) : ∀ a ∈ S.recon x, a ∈ S.reconAll c := by
  intro a ha; simp only [AcceptSpec.reconAll, List.mem_flatMap]; exact ⟨x, hx, ha⟩

omit [DecidableEq A] in
theorem fileAtoms_super {rs rs' : List (List A)} (hsub : ∀ r ∈ rs, r ∈ rs') :
    ∀ a ∈ fileAtoms rs, a ∈ fileAtoms rs' := by
  intro a ha
  obtain ⟨r, hr, har⟩ := (mem_fileAtoms).1 ha
  exact (mem_fileAtoms).2 ⟨r, hsub r hr, har⟩

/-! ### Process facts under changes made by another process -/

theorem procOK_replay_super {ad : Bool} {d : Disk B A} {ph : Phase B A}
    (rs : List (List A)) (t : Tail) (hsub : ∀ r ∈ d.replay, r ∈ rs)
    (hph : ph.holdsLedger = false) (h : ProcOK S ad d ph) :
    ProcOK S ad { d with replay := rs, replayTail := t } ph := by
  have hs := fileAtoms_super hsub
  cases ph with
  | dead => trivial
  | idle m => trivial
  | loaded m b =>
    obtain ⟨hacc, hcov⟩ := h
    refine ⟨hacc, fun a ha => ?_⟩
    rcases hcov a ha with h1 | ⟨h2, h3⟩
    · exact Or.inl h1
    · exact Or.inr ⟨h2, hs a h3⟩
  | checked m b => simp [Phase.holdsLedger] at hph
  | recorded m b =>
    obtain ⟨hacc, hfr, hin⟩ := h
    exact ⟨hacc, hfr, fun had a ha => hs a (hin had a ha)⟩
  | payChecked m b => simp [Phase.holdsLedger] at hph

theorem procOK_replay_delete {d : Disk B A} {ph : Phase B A} (h : ProcOK S true d ph) :
    ProcOK S true { d with replay := [], replayTail := .clean } ph := by
  cases ph with
  | dead => trivial
  | idle m => trivial
  | loaded m b =>
    obtain ⟨hacc, hcov⟩ := h
    refine ⟨hacc, fun a ha => ?_⟩
    rcases hcov a ha with h1 | ⟨h2, _⟩
    · exact Or.inl h1
    · exact absurd h2 (by decide)
  | checked m b =>
    obtain ⟨hacc, hfr, _⟩ := h
    exact ⟨hacc, hfr, by simp [fileAtoms]⟩
  | recorded m b =>
    obtain ⟨hacc, hfr, _⟩ := h
    exact ⟨hacc, hfr, fun h => absurd h (by decide)⟩
  | payChecked m b => simp [ProcOK, fileAtoms]

theorem procOK_chain_change {ad : Bool} {d : Disk B A} {ph : Phase B A}
    (c : List B) (t : Tail) (hph : ph.holdsChain = false) (h : ProcOK S ad d ph) :
    ProcOK S ad { d with chain := c, chainTail := t } ph := by
  cases ph with
  | dead => trivial
  | idle m => trivial
  | loaded m b => simp [Phase.holdsChain] at hph
  | checked m b => simp [Phase.holdsChain] at hph
  | recorded m b => simp [Phase.holdsChain] at hph
  | payChecked m b => exact h

end Invariant

/-! ## Preservation of the invariant by every event -/

section Preservation
variable {B A : Type} [DecidableEq A] (S : AcceptSpec B A) {ad : Bool}

theorem procs_after_set {d : Disk B A} {procs : ℕ → Phase B A} {p : ℕ} {v : Phase B A}
    (hall : ∀ q, q ≠ p → ProcOK S ad d (procs q)) (hv : ProcOK S ad d v) :
    ∀ q, ProcOK S ad d (if q = p then v else procs q) := by
  intro q
  by_cases hq : q = p
  · simp only [hq, if_true]; exact hv
  · simp only [hq, if_false]; exact hall q hq

omit [DecidableEq A] in
/-- `release` never leaves the releasing process as owner. -/
theorem release_ne_self {p : ℕ} {l : Option ℕ} : ¬ release p l = some p := by
  rw [release_eq_some]; exact fun h => h.2 rfl

omit [DecidableEq A] in
theorem release_other {p q : ℕ} {l : Option ℕ} (hq : q ≠ p) : release p l = some q ↔ l = some q := by
  rw [release_eq_some]; exact ⟨fun h => h.1, fun h => ⟨h, hq⟩⟩

omit [DecidableEq A] in
theorem not_holds_of_owner {procs : ℕ → Phase B A} {f : Phase B A → Bool} {l : Option ℕ}
    (hl : ∀ q, l = some q ↔ f (procs q) = true) {p q : ℕ} (hp : l = some p) (hq : q ≠ p) :
    f (procs q) = false := by
  by_contra hc
  have : l = some q := (hl q).2 (by simpa using hc)
  rw [hp] at this; exact hq (Option.some.inj this).symm

theorem covers_of_inv (hrecon : S.LosslessRecon ∨ ad = false) {s : Sys B A} (h : Inv S ad s)
    (m : List A) : Covers S ad (m ++ S.reconAll s.disk.chain) s.disk := by
  intro a ha
  rcases hrecon with hl | had
  · left
    obtain ⟨x, hx, hax⟩ := (mem_chainAtoms S).1 ha
    obtain ⟨c, hc⟩ := h.chain_admitted x hx
    exact List.mem_append_right _ (mem_reconAll S hx a (hl c x hc a hax))
  · exact Or.inr ⟨had, h.chain_in_replay had a ha⟩

theorem inv_begin (hrecon : S.LosslessRecon ∨ ad = false) {s : Sys B A} (h : Inv S ad s)
    (p : ℕ) (b : B) : Inv S ad (step Discipline.implemented S ad s (.begin p b)) := by
  simp only [step]
  split
  · rename_i m hp
    simp only [Discipline.implemented, canTake, take, Bool.not_true, Bool.false_or, if_true]
    split_ifs with hfree hload hacc
    · have hnone : s.chainLock = none := by simpa using hfree
      refine ⟨h.chain_valid, h.replay_unique, h.chain_admitted, h.chain_in_replay, ?_, ?_, ?_⟩
      · refine lock_after_set h.chain_lock (by simp [Phase.holdsChain]) ?_
        intro q hq; rw [hnone]; simp [Ne.symm hq]
      · refine lock_after_set h.ledger_lock ?_ (fun _ _ => Iff.rfl)
        have := h.ledger_lock p; rw [hp] at this; simpa [Phase.holdsLedger] using this
      · refine procs_after_set S (fun q _ => h.procs_ok q) ⟨hacc, covers_of_inv S hrecon h m⟩
    · refine ⟨h.chain_valid, h.replay_unique, h.chain_admitted, h.chain_in_replay, ?_, ?_, ?_⟩
      · refine lock_after_set h.chain_lock ?_ (fun _ _ => Iff.rfl)
        have := h.chain_lock p; rw [hp] at this; simpa [Phase.holdsChain] using this
      · refine lock_after_set h.ledger_lock ?_ (fun _ _ => Iff.rfl)
        have := h.ledger_lock p; rw [hp] at this; simpa [Phase.holdsLedger] using this
      · exact procs_after_set S (fun q _ => h.procs_ok q) trivial
    · exact h
    · exact h
  · exact h

theorem inv_ledger {s : Sys B A} (h : Inv S ad s) (p : ℕ) :
    Inv S ad (step Discipline.implemented S ad s (.ledger p)) := by
  simp only [step]
  split
  · rename_i m b hp
    have hown : s.chainLock = some p := (h.chain_lock p).2 (by rw [hp]; rfl)
    have hnl : s.ledgerLock ≠ some p := by
      intro hc; have := (h.ledger_lock p).1 hc; rw [hp] at this; simp [Phase.holdsLedger] at this
    have hpk : ProcOK S ad s.disk (s.procs p) := h.procs_ok p
    rw [hp] at hpk
    obtain ⟨hacc, hcov⟩ := hpk
    simp only [Discipline.implemented, canTake, take, Bool.not_true, Bool.false_or, if_true,
      Bool.true_and]
    have reject : ∀ m', Inv S ad { s with procs := s.set p (.idle m'), chainLock := release p s.chainLock } := by
      intro m'
      refine ⟨h.chain_valid, h.replay_unique, h.chain_admitted, h.chain_in_replay, ?_, ?_, ?_⟩
      · exact lock_after_set h.chain_lock (by simp [release_ne_self, Phase.holdsChain])
          (fun q hq => release_other hq)
      · exact lock_after_set h.ledger_lock (by simp [hnl, Phase.holdsLedger]) (fun _ _ => Iff.rfl)
      · exact procs_after_set S (fun q _ => h.procs_ok q) trivial
    split_ifs with hfree hscan hhit
    · exact reject m
    · exact reject _
    · have hnone : s.ledgerLock = none := by simpa using hfree
      refine ⟨h.chain_valid, h.replay_unique, h.chain_admitted, h.chain_in_replay, ?_, ?_, ?_⟩
      · refine lock_after_set h.chain_lock ?_ (fun _ _ => Iff.rfl)
        simp [hown, Phase.holdsChain]
      · refine lock_after_set h.ledger_lock (by simp [Phase.holdsLedger]) ?_
        intro q hq; rw [hnone]; simp [Ne.symm hq]
      · refine procs_after_set S (fun q _ => h.procs_ok q) ?_
        have hnot := (hits_eq_false S).1 (by simpa using hhit)
        refine ⟨hacc, (fresh_iff S).2 fun a ha hc => ?_, fun a ha hf => ?_⟩
        · rcases hcov a hc with h1 | ⟨_, h3⟩
          · exact hnot a ha (List.mem_append_left _ h1)
          · exact hnot a ha (List.mem_append_right _ h3)
        · exact hnot a ha (List.mem_append_right _ hf)
    · exact h
  · exact h

omit [DecidableEq A] in
theorem replay_super_of {rs : List (List A)} (r : List A) (add : Bool) :
    ∀ x ∈ rs, x ∈ (if add then r :: rs else rs) := by
  intro x hx; split_ifs
  · exact List.mem_cons_of_mem _ hx
  · exact hx

theorem inv_writeRecord {s : Sys B A} (h : Inv S ad s) (p : ℕ) (o : WriteOutcome) :
    Inv S ad (step Discipline.implemented S ad s (.writeRecord p o)) := by
  simp only [step]
  split
  · rename_i m b hp
    have hownC : s.chainLock = some p := (h.chain_lock p).2 (by rw [hp]; rfl)
    have hownL : s.ledgerLock = some p := (h.ledger_lock p).2 (by rw [hp]; rfl)
    have hpk : ProcOK S ad s.disk (s.procs p) := h.procs_ok p
    rw [hp] at hpk
    obtain ⟨hacc, hfr, hnot⟩ := hpk
    set d' : Disk B A :=
      { s.disk with
        replay := if o.effect.adds then S.atoms b :: s.disk.replay else s.disk.replay
        replayTail := o.effect.tail s.disk.replayTail } with hd'
    have hsup : ∀ r ∈ s.disk.replay, r ∈ d'.replay := replay_super_of _ _
    have huniq : replayUnique d'.replay = true := by
      simp only [hd']
      split_ifs
      · exact replayUnique_cons.2 ⟨h.replay_unique, hnot⟩
      · exact h.replay_unique
    have hcir : ad = false → ∀ a ∈ S.chainAtoms d'.chain, a ∈ fileAtoms d'.replay :=
      fun had a ha => fileAtoms_super hsup a (h.chain_in_replay had a ha)
    have hothers : ∀ q, q ≠ p → ProcOK S ad d' (s.procs q) := by
      intro q hq
      exact procOK_replay_super S _ _ hsup (not_holds_of_owner h.ledger_lock hownL hq) (h.procs_ok q)
    cases o with
    | ok =>
      refine ⟨h.chain_valid, huniq, h.chain_admitted, hcir, ?_, ?_, ?_⟩
      · exact lock_after_set h.chain_lock (by simp [hownC, Phase.holdsChain]) (fun _ _ => Iff.rfl)
      · exact lock_after_set h.ledger_lock (by simp [release_ne_self, Phase.holdsLedger])
          (fun q hq => release_other hq)
      · refine procs_after_set S hothers ⟨hacc, hfr, fun _ a ha => ?_⟩
        simp only [hd', WriteOutcome.effect, Effect.adds, if_true]
        exact (mem_fileAtoms).2 ⟨_, List.mem_cons_self, ha⟩
    | fail e =>
      refine ⟨h.chain_valid, huniq, h.chain_admitted, hcir, ?_, ?_, ?_⟩
      · exact lock_after_set h.chain_lock (by simp [release_ne_self, Phase.holdsChain])
          (fun q hq => release_other hq)
      · exact lock_after_set h.ledger_lock (by simp [release_ne_self, Phase.holdsLedger])
          (fun q hq => release_other hq)
      · exact procs_after_set S hothers trivial
    | crash e =>
      refine ⟨h.chain_valid, huniq, h.chain_admitted, hcir, ?_, ?_, ?_⟩
      · exact lock_after_set h.chain_lock (by simp [release_ne_self, Phase.holdsChain])
          (fun q hq => release_other hq)
      · exact lock_after_set h.ledger_lock (by simp [release_ne_self, Phase.holdsLedger])
          (fun q hq => release_other hq)
      · exact procs_after_set S hothers trivial
  · exact h

theorem inv_writeBlock (hsound : S.AcceptSound) {s : Sys B A} (h : Inv S ad s) (p : ℕ)
    (o : WriteOutcome) : Inv S ad (step Discipline.implemented S ad s (.writeBlock p o)) := by
  simp only [step]
  split
  · rename_i m b hp
    have hownC : s.chainLock = some p := (h.chain_lock p).2 (by rw [hp]; rfl)
    have hnl : s.ledgerLock ≠ some p := by
      intro hc; have := (h.ledger_lock p).1 hc; rw [hp] at this; simp [Phase.holdsLedger] at this
    have hpk : ProcOK S ad s.disk (s.procs p) := h.procs_ok p
    rw [hp] at hpk
    obtain ⟨hacc, hfr, hin⟩ := hpk
    obtain ⟨hlink, hblk⟩ := hsound _ _ hacc
    set d' : Disk B A :=
      { s.disk with
        chain := if o.effect.adds then b :: s.disk.chain else s.disk.chain
        chainTail := o.effect.tail s.disk.chainTail } with hd'
    have hvalid : S.chainValid d'.chain = true := by
      simp only [hd']
      split_ifs
      · exact (chainValid_cons S).2 ⟨h.chain_valid, hlink, hblk, hfr⟩
      · exact h.chain_valid
    have hadm : ∀ x ∈ d'.chain, ∃ c, S.acceptOK c x = true := by
      intro x hx
      simp only [hd'] at hx
      split_ifs at hx
      · rcases List.mem_cons.1 hx with rfl | hx
        · exact ⟨_, hacc⟩
        · exact h.chain_admitted x hx
      · exact h.chain_admitted x hx
    have hcir : ad = false → ∀ a ∈ S.chainAtoms d'.chain, a ∈ fileAtoms d'.replay := by
      intro had a ha
      simp only [hd'] at ha ⊢
      split_ifs at ha
      · simp only [AcceptSpec.chainAtoms, List.flatMap_cons, List.mem_append] at ha
        rcases ha with ha | ha
        · exact hin had a ha
        · exact h.chain_in_replay had a ha
      · exact h.chain_in_replay had a ha
    have hothers : ∀ q, q ≠ p → ProcOK S ad d' (s.procs q) := by
      intro q hq
      exact procOK_chain_change S _ _ (not_holds_of_owner h.chain_lock hownC hq) (h.procs_ok q)
    have hlockL : ∀ v : Phase B A, v.holdsLedger = false →
        ∀ q, s.ledgerLock = some q ↔ (if q = p then v else s.procs q).holdsLedger = true :=
      fun v hv => lock_after_set h.ledger_lock (by simp [hnl, hv]) (fun _ _ => Iff.rfl)
    have hlockC : ∀ v : Phase B A, v.holdsChain = false →
        ∀ q, release p s.chainLock = some q ↔ (if q = p then v else s.procs q).holdsChain = true :=
      fun v hv => lock_after_set h.chain_lock (by simp [release_ne_self, hv])
        (fun q hq => release_other hq)
    cases o with
    | ok => exact ⟨hvalid, h.replay_unique, hadm, hcir, hlockC _ rfl, hlockL _ rfl,
        procs_after_set S hothers trivial⟩
    | fail e => exact ⟨hvalid, h.replay_unique, hadm, hcir, hlockC _ rfl, hlockL _ rfl,
        procs_after_set S hothers trivial⟩
    | crash e => exact ⟨hvalid, h.replay_unique, hadm, hcir, hlockC _ rfl, hlockL _ rfl,
        procs_after_set S hothers trivial⟩
  · exact h

theorem inv_payBegin {s : Sys B A} (h : Inv S ad s) (p : ℕ) (b : B) :
    Inv S ad (step Discipline.implemented S ad s (.payBegin p b)) := by
  simp only [step]
  split
  · rename_i m hp
    have hnc : s.chainLock ≠ some p := by
      intro hc; have := (h.chain_lock p).1 hc; rw [hp] at this; simp [Phase.holdsChain] at this
    have hnl : s.ledgerLock ≠ some p := by
      intro hc; have := (h.ledger_lock p).1 hc; rw [hp] at this; simp [Phase.holdsLedger] at this
    simp only [Discipline.implemented, canTake, take, Bool.not_true, Bool.false_or, if_true,
      Bool.true_and]
    split_ifs with hpay hfree hscan hhit
    · exact h
    · refine ⟨h.chain_valid, h.replay_unique, h.chain_admitted, h.chain_in_replay, ?_, ?_, ?_⟩
      · exact lock_after_set h.chain_lock (by simp [hnc, Phase.holdsChain]) (fun _ _ => Iff.rfl)
      · exact lock_after_set h.ledger_lock (by simp [hnl, Phase.holdsLedger]) (fun _ _ => Iff.rfl)
      · exact procs_after_set S (fun q _ => h.procs_ok q) trivial
    · have hnone : s.ledgerLock = none := by simpa using hfree
      refine ⟨h.chain_valid, h.replay_unique, h.chain_admitted, h.chain_in_replay, ?_, ?_, ?_⟩
      · exact lock_after_set h.chain_lock (by simp [hnc, Phase.holdsChain]) (fun _ _ => Iff.rfl)
      · refine lock_after_set h.ledger_lock (by simp [Phase.holdsLedger]) ?_
        intro q hq; rw [hnone]; simp [Ne.symm hq]
      · refine procs_after_set S (fun q _ => h.procs_ok q) ?_
        have hnot := (hits_eq_false S).1 (by simpa using hhit)
        exact fun a ha hf => hnot a ha (List.mem_append_right _ hf)
    · exact h
    · exact h
  · exact h

theorem inv_payWrite {s : Sys B A} (h : Inv S ad s) (p : ℕ) (o : WriteOutcome) :
    Inv S ad (step Discipline.implemented S ad s (.payWrite p o)) := by
  simp only [step]
  split
  · rename_i m b hp
    have hownL : s.ledgerLock = some p := (h.ledger_lock p).2 (by rw [hp]; rfl)
    have hnc : s.chainLock ≠ some p := by
      intro hc; have := (h.chain_lock p).1 hc; rw [hp] at this; simp [Phase.holdsChain] at this
    have hpk : ProcOK S ad s.disk (s.procs p) := h.procs_ok p
    rw [hp] at hpk
    set d' : Disk B A :=
      { s.disk with
        replay := if o.effect.adds then S.atoms b :: s.disk.replay else s.disk.replay
        replayTail := o.effect.tail s.disk.replayTail } with hd'
    have hsup : ∀ r ∈ s.disk.replay, r ∈ d'.replay := replay_super_of _ _
    have huniq : replayUnique d'.replay = true := by
      simp only [hd']
      split_ifs
      · exact replayUnique_cons.2 ⟨h.replay_unique, hpk⟩
      · exact h.replay_unique
    have hcir : ad = false → ∀ a ∈ S.chainAtoms d'.chain, a ∈ fileAtoms d'.replay :=
      fun had a ha => fileAtoms_super hsup a (h.chain_in_replay had a ha)
    have hothers : ∀ q, q ≠ p → ProcOK S ad d' (s.procs q) := by
      intro q hq
      exact procOK_replay_super S _ _ hsup (not_holds_of_owner h.ledger_lock hownL hq) (h.procs_ok q)
    have hlockC : ∀ v : Phase B A, v.holdsChain = false →
        ∀ q, s.chainLock = some q ↔ (if q = p then v else s.procs q).holdsChain = true :=
      fun v hv => lock_after_set h.chain_lock (by simp [hnc, hv]) (fun _ _ => Iff.rfl)
    have hlockL : ∀ v : Phase B A, v.holdsLedger = false →
        ∀ q, release p s.ledgerLock = some q ↔ (if q = p then v else s.procs q).holdsLedger = true :=
      fun v hv => lock_after_set h.ledger_lock (by simp [release_ne_self, hv])
        (fun q hq => release_other hq)
    cases o with
    | ok => exact ⟨h.chain_valid, huniq, h.chain_admitted, hcir, hlockC _ rfl, hlockL _ rfl,
        procs_after_set S hothers trivial⟩
    | fail e => exact ⟨h.chain_valid, huniq, h.chain_admitted, hcir, hlockC _ rfl, hlockL _ rfl,
        procs_after_set S hothers trivial⟩
    | crash e => exact ⟨h.chain_valid, huniq, h.chain_admitted, hcir, hlockC _ rfl, hlockL _ rfl,
        procs_after_set S hothers trivial⟩
  · exact h

theorem inv_crash {s : Sys B A} (h : Inv S ad s) (p : ℕ) :
    Inv S ad (step Discipline.implemented S ad s (.crash p)) := by
  simp only [step]
  refine ⟨h.chain_valid, h.replay_unique, h.chain_admitted, h.chain_in_replay, ?_, ?_, ?_⟩
  · exact lock_after_set h.chain_lock (by simp [release_ne_self, Phase.holdsChain])
      (fun q hq => release_other hq)
  · exact lock_after_set h.ledger_lock (by simp [release_ne_self, Phase.holdsLedger])
      (fun q hq => release_other hq)
  · exact procs_after_set S (fun q _ => h.procs_ok q) trivial

theorem inv_restart {s : Sys B A} (h : Inv S ad s) (p : ℕ) :
    Inv S ad (step Discipline.implemented S ad s (.restart p)) := by
  simp only [step]
  split
  · rename_i hp
    have hnc : s.chainLock ≠ some p := by
      intro hc; have := (h.chain_lock p).1 hc; rw [hp] at this; simp [Phase.holdsChain] at this
    have hnl : s.ledgerLock ≠ some p := by
      intro hc; have := (h.ledger_lock p).1 hc; rw [hp] at this; simp [Phase.holdsLedger] at this
    split_ifs
    · refine ⟨h.chain_valid, h.replay_unique, h.chain_admitted, h.chain_in_replay, ?_, ?_, ?_⟩
      · exact lock_after_set h.chain_lock (by simp [hnc, Phase.holdsChain]) (fun _ _ => Iff.rfl)
      · exact lock_after_set h.ledger_lock (by simp [hnl, Phase.holdsLedger]) (fun _ _ => Iff.rfl)
      · exact procs_after_set S (fun q _ => h.procs_ok q) trivial
    · exact h
  · exact h

theorem inv_deleteReplay {s : Sys B A} (h : Inv S ad s) :
    Inv S ad (step Discipline.implemented S ad s .deleteReplay) := by
  simp only [step]
  split_ifs with had
  · subst had
    refine ⟨h.chain_valid, rfl, h.chain_admitted, fun h' => absurd h' (by decide),
      h.chain_lock, h.ledger_lock, fun q => procOK_replay_delete S (h.procs_ok q)⟩
  · exact h

/-- **Preservation.**  Every event of the implemented discipline preserves the invariant, provided
acceptance entails the chain validator's per-block checks, and either reconstruction from the
chain is lossless or the side replay file is never lost. -/
theorem inv_step (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    {s : Sys B A} (h : Inv S ad s) (e : Event B) :
    Inv S ad (step Discipline.implemented S ad s e) := by
  cases e with
  | begin p b => exact inv_begin S hrecon h p b
  | ledger p => exact inv_ledger S h p
  | writeRecord p o => exact inv_writeRecord S h p o
  | writeBlock p o => exact inv_writeBlock S hsound h p o
  | payBegin p b => exact inv_payBegin S h p b
  | payWrite p o => exact inv_payWrite S h p o
  | crash p => exact inv_crash S h p
  | restart p => exact inv_restart S h p
  | deleteReplay => exact inv_deleteReplay S h

theorem inv_init : Inv S ad (Sys.init : Sys B A) := by
  refine ⟨rfl, rfl, by simp [Sys.init], fun _ => by simp [Sys.init, AcceptSpec.chainAtoms],
    ?_, ?_, fun _ => trivial⟩
  · intro p; simp [Sys.init, Phase.holdsChain]
  · intro p; simp [Sys.init, Phase.holdsLedger]

/-- The invariant holds after every finite trace from the empty host. -/
theorem inv_run (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) : Inv S ad (run Discipline.implemented S ad es) := by
  unfold run
  suffices ∀ (es : List (Event B)) (s : Sys B A), Inv S ad s →
      Inv S ad (es.foldl (step Discipline.implemented S ad) s) from this es _ (inv_init S)
  intro es
  induction es with
  | nil => intro s hs; exact hs
  | cons e es ih => intro s hs; exact ih _ (inv_step S hsound hrecon hs e)

end Preservation

/-! ## Consequences for arbitrary finite traces -/

section Consequences
variable {B A : Type} [DecidableEq A] (S : AcceptSpec B A) {ad : Bool}

/-- Two blocks share no atom. -/
def Disjoint (x y : B) : Prop := ∀ a ∈ S.atoms x, a ∉ S.atoms y

theorem chainValid_pairwise : ∀ c : List B, S.chainValid c = true → c.Pairwise (Disjoint S)
  | [], _ => List.Pairwise.nil
  | b :: c, h => by
    obtain ⟨hc, _, _, hfr⟩ := (chainValid_cons S).1 h
    refine List.Pairwise.cons (fun y hy a ha hay => ?_) (chainValid_pairwise c hc)
    exact (fresh_iff S).1 hfr a ha ((mem_chainAtoms S).2 ⟨y, hy, hay⟩)

/-- **Single use in the authoritative chain.**  After any finite trace — any interleaving of any
number of same-host processes, any write failures, crashes at any modelled point, restarts and
loss of the side replay file — no two stored blocks share an identity atom. -/
theorem chain_single_use (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) :
    (run Discipline.implemented S ad es).disk.chain.Pairwise (Disjoint S) :=
  chainValid_pairwise S _ (inv_run S hsound hrecon es).chain_valid

omit [DecidableEq A] in
/-- A per-component projection has no repeats on a chain of pairwise atom-disjoint blocks when
every block exposes that component as an atom. -/
theorem nodup_of_pairwise {c : List B} {V : Type} (f : B → V) (g : V → A)
    (hmem : ∀ x, g (f x) ∈ S.atoms x) (h : c.Pairwise (Disjoint S)) : (c.map f).Nodup := by
  rw [List.Nodup, List.pairwise_map]
  refine h.imp fun {x y} hxy heq => hxy _ (hmem x) ?_
  rw [heq]; exact hmem y

theorem run_append (D : Discipline) (es es' : List (Event B)) :
    run D S ad (es ++ es') = es'.foldl (step D S ad) (run D S ad es) := by
  simp [run, List.foldl_append]

/-- The durable chain is append-only under every discipline (no event removes a block). -/
theorem chain_suffix_step (D : Discipline) (s : Sys B A) (e : Event B) :
    s.disk.chain <:+ (step D S ad s e).disk.chain := by
  cases e with
  | writeBlock p o =>
    simp only [step]
    split
    · cases o <;> simp only <;> split_ifs <;>
        first | exact List.suffix_cons _ _ | exact List.suffix_refl _
    · exact List.suffix_refl _
  | begin p b =>
    simp only [step]; split
    · split_ifs <;> exact List.suffix_refl _
    · exact List.suffix_refl _
  | ledger p =>
    simp only [step]; split
    · split_ifs <;> exact List.suffix_refl _
    · exact List.suffix_refl _
  | writeRecord p o =>
    simp only [step]; split
    · cases o <;> exact List.suffix_refl _
    · exact List.suffix_refl _
  | payBegin p b =>
    simp only [step]; split
    · split_ifs <;> exact List.suffix_refl _
    · exact List.suffix_refl _
  | payWrite p o =>
    simp only [step]; split
    · cases o <;> exact List.suffix_refl _
    · exact List.suffix_refl _
  | crash p => exact List.suffix_refl _
  | restart p =>
    simp only [step]; split
    · split_ifs <;> exact List.suffix_refl _
    · exact List.suffix_refl _
  | deleteReplay =>
    simp only [step]; split_ifs <;> exact List.suffix_refl _

theorem chain_suffix_foldl (D : Discipline) (es : List (Event B)) (s : Sys B A) :
    s.disk.chain <:+ (es.foldl (step D S ad) s).disk.chain := by
  induction es generalizing s with
  | nil => exact List.suffix_refl _
  | cons e es ih => exact (chain_suffix_step S D s e).trans (ih _)

/-- **Consumption is fresh.**  Whenever a process is about to durably record a consumption
(`_append` under the ledger lock), none of the block's atoms is consumed in the authoritative
durable state — accepted chain *and* replay records — at that instant. -/
theorem consume_fresh (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) {p : ℕ} {m : List A} {b : B}
    (hp : (run Discipline.implemented S ad es).procs p = .checked m b) :
    ∀ a ∈ S.atoms b, a ∉ S.consumed (run Discipline.implemented S ad es).disk := by
  have h := (inv_run S hsound hrecon es).procs_ok p
  rw [hp] at h
  obtain ⟨_, hfr, hnot⟩ := h
  intro a ha hc
  rcases List.mem_append.1 hc with hc | hc
  · exact (fresh_iff S).1 hfr a ha hc
  · exact hnot a ha hc

/-- **Consumption records every atom.**  A record append whose bytes land (successful, or followed
by an I/O error) makes every atom of the block consumed. -/
theorem writeRecord_consumes_all (D : Discipline) {s : Sys B A} {p : ℕ} {m : List A} {b : B}
    (hp : s.procs p = .checked m b) {o : WriteOutcome} (ho : o.effect.adds = true) :
    ∀ a ∈ S.atoms b, a ∈ S.consumed (step D S ad s (.writeRecord p o)).disk := by
  intro a ha
  have hrec : S.atoms b ∈ (step D S ad s (.writeRecord p o)).disk.replay := by
    simp only [step, hp]
    cases o <;> simp_all [WriteOutcome.effect]
  exact List.mem_append_right _ ((mem_fileAtoms).2 ⟨_, hrec, ha⟩)

/-- A block append whose bytes land puts the block itself in the authoritative chain. -/
theorem writeBlock_appends (D : Discipline) {s : Sys B A} {p : ℕ} {m : List A} {b : B}
    (hp : s.procs p = .recorded m b) {o : WriteOutcome} (ho : o.effect.adds = true) :
    (step D S ad s (.writeBlock p o)).disk.chain = b :: s.disk.chain := by
  simp only [step, hp]
  cases o <;> simp_all [WriteOutcome.effect]

/-- **At most one acceptance per atom, over any interleaving.**  If some process durably appended
`b₁` (a reported acceptance, or an append whose bytes landed before an I/O error), no process can
later even reach the post-consumption phase for a block sharing any atom with it. -/
theorem at_most_one_acceptance (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es₁ es₂ : List (Event B)) {p q : ℕ} {o : WriteOutcome} {m₁ m₂ : List A} {b₁ b₂ : B}
    (hadds : o.effect.adds = true)
    (h₁ : (run Discipline.implemented S ad es₁).procs p = .recorded m₁ b₁)
    (h₂ : (run Discipline.implemented S ad (es₁ ++ .writeBlock p o :: es₂)).procs q =
      .recorded m₂ b₂) :
    Disjoint S b₂ b₁ := by
  set s₁ := run Discipline.implemented S ad es₁
  set s₂ := run Discipline.implemented S ad (es₁ ++ .writeBlock p o :: es₂)
  have hmem₁ : b₁ ∈ (step Discipline.implemented S ad s₁ (.writeBlock p o)).disk.chain := by
    rw [writeBlock_appends S _ h₁ hadds]; exact List.mem_cons_self
  have hsuf : (step Discipline.implemented S ad s₁ (.writeBlock p o)).disk.chain <:+ s₂.disk.chain := by
    have : s₂ = es₂.foldl (step Discipline.implemented S ad)
        (step Discipline.implemented S ad s₁ (.writeBlock p o)) := by
      simp only [s₂, s₁, run_append, List.foldl_cons]
    rw [this]; exact chain_suffix_foldl S _ es₂ _
  have hb₁ : b₁ ∈ s₂.disk.chain := hsuf.subset hmem₁
  have h := (inv_run S hsound hrecon (es₁ ++ .writeBlock p o :: es₂)).procs_ok q
  rw [h₂] at h
  obtain ⟨_, hfr, _⟩ := h
  intro a ha ha₁
  exact (fresh_iff S).1 hfr a ha ((mem_chainAtoms S).2 ⟨b₁, hb₁, ha₁⟩)

/-- Rejecting and control events never touch durable state. -/
theorem control_events_keep_disk (D : Discipline) (s : Sys B A) (e : Event B)
    (he : (∃ p b, e = .begin p b) ∨ (∃ p, e = .ledger p) ∨ (∃ p b, e = .payBegin p b) ∨
      (∃ p, e = .crash p) ∨ (∃ p, e = .restart p)) :
    (step D S ad s e).disk = s.disk := by
  rcases he with ⟨p, b, rfl⟩ | ⟨p, rfl⟩ | ⟨p, b, rfl⟩ | ⟨p, rfl⟩ | ⟨p, rfl⟩
  · simp only [step]; split
    · split_ifs <;> rfl
    · rfl
  · simp only [step]; split
    · split_ifs <;> rfl
    · rfl
  · simp only [step]; split
    · split_ifs <;> rfl
    · rfl
  · rfl
  · simp only [step]; split
    · split_ifs <;> rfl
    · rfl

/-- Consumption is monotone: apart from loss of the side replay file, no event removes a
consumed atom from the authoritative durable state. -/
theorem consumed_monotone (D : Discipline) (s : Sys B A) (e : Event B) (he : e ≠ .deleteReplay) :
    ∀ a ∈ S.consumed s.disk, a ∈ S.consumed (step D S ad s e).disk := by
  have hchain := chain_suffix_step S (ad := ad) D s e
  have hrep : ∀ r ∈ s.disk.replay, r ∈ (step D S ad s e).disk.replay := by
    intro r hr
    cases e with
    | writeRecord p o =>
      simp only [step]; split
      · cases o <;> simp only <;> exact replay_super_of _ _ r hr
      · exact hr
    | payWrite p o =>
      simp only [step]; split
      · cases o <;> simp only <;> exact replay_super_of _ _ r hr
      · exact hr
    | writeBlock p o =>
      simp only [step]; split
      · cases o <;> exact hr
      · exact hr
    | deleteReplay => exact absurd rfl he
    | begin p b => rw [control_events_keep_disk S D s _ (Or.inl ⟨p, b, rfl⟩)]; exact hr
    | ledger p => rw [control_events_keep_disk S D s _ (Or.inr (Or.inl ⟨p, rfl⟩))]; exact hr
    | payBegin p b =>
      rw [control_events_keep_disk S D s _ (Or.inr (Or.inr (Or.inl ⟨p, b, rfl⟩)))]; exact hr
    | crash p => exact hr
    | restart p =>
      rw [control_events_keep_disk S D s _ (Or.inr (Or.inr (Or.inr (Or.inr ⟨p, rfl⟩))))]; exact hr
  intro a ha
  rcases List.mem_append.1 ha with ha | ha
  · obtain ⟨x, hx, hax⟩ := (mem_chainAtoms S).1 ha
    exact List.mem_append_left _ ((mem_chainAtoms S).2 ⟨x, hchain.subset hx, hax⟩)
  · exact List.mem_append_right _ (fileAtoms_super hrep a ha)

/-- **Restart reconstruction.**  A restarted process holds every authoritative consumed atom in
memory when reconstruction is lossless. -/
theorem restart_recovers (hl : S.LosslessRecon) {s : Sys B A} (h : Inv S ad s) {p : ℕ}
    (hp : s.procs p = .dead) (hload : S.replayLoads s.disk = true ∧ S.chainLoads s.disk = true) :
    ∃ m, (step Discipline.implemented S ad s (.restart p)).procs p = .idle m ∧
      ∀ a ∈ S.consumed s.disk, a ∈ m := by
  refine ⟨fileAtoms s.disk.replay ++ S.reconAll s.disk.chain, ?_, ?_⟩
  · simp [step, hp, hload, Sys.set]
  · intro a ha
    rcases List.mem_append.1 ha with ha | ha
    · obtain ⟨x, hx, hax⟩ := (mem_chainAtoms S).1 ha
      obtain ⟨c, hc⟩ := h.chain_admitted x hx
      exact List.mem_append_right _ (mem_reconAll S hx a (hl c x hc a hax))
    · exact List.mem_append_left _ ha

/-- **Fail-closed under torn chain writes.**  A garbage chain tail is permanent and no
acceptance can start: every later `begin` is a no-op. -/
theorem chain_garbage_absorbing (D : Discipline) (s : Sys B A) (e : Event B)
    (h : s.disk.chainTail = .garbage) : (step D S ad s e).disk.chainTail = .garbage := by
  have ht : ∀ x : Effect, x.tail .garbage = .garbage := by intro x; cases x <;> rfl
  cases e with
  | writeBlock p o =>
    simp only [step]; split
    · cases o <;> simp [h, ht]
    · exact h
  | writeRecord p o =>
    simp only [step]; split
    · cases o <;> exact h
    · exact h
  | payWrite p o =>
    simp only [step]; split
    · cases o <;> exact h
    · exact h
  | deleteReplay => simp only [step]; split_ifs <;> exact h
  | begin p b => rw [control_events_keep_disk S D s _ (Or.inl ⟨p, b, rfl⟩)]; exact h
  | ledger p => rw [control_events_keep_disk S D s _ (Or.inr (Or.inl ⟨p, rfl⟩))]; exact h
  | payBegin p b =>
    rw [control_events_keep_disk S D s _ (Or.inr (Or.inr (Or.inl ⟨p, b, rfl⟩)))]; exact h
  | crash p => exact h
  | restart p =>
    rw [control_events_keep_disk S D s _ (Or.inr (Or.inr (Or.inr (Or.inr ⟨p, rfl⟩))))]; exact h

theorem begin_blocked_by_garbage (D : Discipline) (s : Sys B A) (p : ℕ) (b : B)
    (h : s.disk.chainTail = .garbage) : step D S ad s (.begin p b) = s := by
  simp only [step]
  split
  · have : S.chainLoads s.disk = false := by simp [AcceptSpec.chainLoads, h]
    simp [this]
  · rfl

/-- **Fail-closed under a torn or unreadable replay file.**  With the implemented discipline a
process that has passed the chain checks never reaches the consumption phase while the replay
file does not load. -/
theorem ledger_blocked_by_bad_replay (s : Sys B A) {p : ℕ} {m : List A} {b : B}
    (hp : s.procs p = .loaded m b) (h : S.replayLoads s.disk = false) (m' : List A) (b' : B) :
    (step Discipline.implemented S ad s (.ledger p)).procs p ≠ .checked m' b' := by
  simp only [step, hp, Discipline.implemented, canTake, take, Bool.not_true, Bool.false_or,
    if_true, Bool.true_and, h, Bool.not_false]
  split_ifs <;> simp [Sys.set, hp]

end Consequences

/-! ## Liveness: acceptance never makes the node's own state unloadable -/

section Liveness
variable {B A : Type} [DecidableEq A] (S : AcceptSpec B A) {ad : Bool}

/-- Load-relevant knowledge of a process about the block or record it is about to write. -/
def LoadOK : Phase B A → Prop
  | .checked _ b => S.recordOK (S.atoms b) = true ∧ S.reconFail b = false
  | .recorded _ b => S.reconFail b = false
  | .payChecked _ b => S.recordOK (S.atoms b) = true
  | _ => True

structure LoadInv (s : Sys B A) : Prop where
  chain_tail : s.disk.chainTail = .clean
  replay_tail : s.disk.replayTail = .clean
  records_ok : ∀ r ∈ s.disk.replay, S.recordOK r = true
  chain_total : ∀ x ∈ s.disk.chain, S.reconFail x = false
  procs : ∀ p, LoadOK S (s.procs p)

/-- No append tears: every write leaves either nothing or one complete line. -/
def CleanIO : Event B → Prop
  | .writeRecord _ o => o.effect = .line ∨ o.effect = .nothing
  | .writeBlock _ o => o.effect = .line ∨ o.effect = .nothing
  | .payWrite _ o => o.effect = .line ∨ o.effect = .nothing
  | _ => True

omit [DecidableEq A] in
theorem loadOK_after_set {procs : ℕ → Phase B A} {p : ℕ} {v : Phase B A}
    (hall : ∀ q, LoadOK S (procs q)) (hv : LoadOK S v) : ∀ q, LoadOK S (if q = p then v else procs q) := by
  intro q; by_cases hq : q = p
  · simp only [hq, if_true]; exact hv
  · simp only [hq, if_false]; exact hall q

theorem clean_tail {o : WriteOutcome} (ho : o.effect = .line ∨ o.effect = .nothing) :
    o.effect.tail .clean = .clean := by
  rcases ho with h | h <;> rw [h] <;> rfl

omit [DecidableEq A] in
theorem loadInv_set {s : Sys B A} {p : ℕ} {v : Phase B A} {c l : Option ℕ}
    (h : LoadInv S s) (hv : LoadOK S v) :
    LoadInv S { s with procs := s.set p v, chainLock := c, ledgerLock := l } :=
  ⟨h.chain_tail, h.replay_tail, h.records_ok, h.chain_total, loadOK_after_set S h.procs hv⟩

theorem loadInv_step (hadm : S.RecordAdmissible) (htot : S.ReconTotal) {s : Sys B A}
    (hi : Inv S ad s) (h : LoadInv S s) (e : Event B) (he : CleanIO e) :
    LoadInv S (step Discipline.implemented S ad s e) := by
  cases e with
  | begin p b =>
    simp only [step]; split
    · simp only [Discipline.implemented, canTake, take, Bool.not_true, Bool.false_or, if_true]
      split_ifs
      all_goals first | exact h | exact loadInv_set S h trivial
    · exact h
  | ledger p =>
    simp only [step]; split
    · rename_i m b hp
      have hpk := hi.procs_ok p
      rw [hp] at hpk
      have hacc := hpk.1
      simp only [Discipline.implemented, canTake, take, Bool.not_true, Bool.false_or, if_true,
        Bool.true_and]
      split_ifs
      all_goals first
        | exact h
        | exact loadInv_set S h trivial
        | exact loadInv_set S h ⟨hadm.1 _ _ hacc, htot _ _ hacc⟩
    · exact h
  | writeRecord p o =>
    simp only [step]; split
    · rename_i m b hp
      have hl := h.procs p
      rw [hp] at hl
      obtain ⟨hrec, htotb⟩ := hl
      have hrecs : ∀ r ∈ (if o.effect.adds then S.atoms b :: s.disk.replay else s.disk.replay),
          S.recordOK r = true := by
        intro r hr; split_ifs at hr
        · rcases List.mem_cons.1 hr with rfl | hr
          · exact hrec
          · exact h.records_ok r hr
        · exact h.records_ok r hr
      have ht : o.effect.tail s.disk.replayTail = .clean := by rw [h.replay_tail]; exact clean_tail he
      cases o with
      | ok => exact ⟨h.chain_tail, ht, hrecs, h.chain_total, loadOK_after_set S h.procs htotb⟩
      | fail e => exact ⟨h.chain_tail, ht, hrecs, h.chain_total, loadOK_after_set S h.procs trivial⟩
      | crash e => exact ⟨h.chain_tail, ht, hrecs, h.chain_total, loadOK_after_set S h.procs trivial⟩
    · exact h
  | writeBlock p o =>
    simp only [step]; split
    · rename_i m b hp
      have hl := h.procs p
      rw [hp] at hl
      have hchain : ∀ x ∈ (if o.effect.adds then b :: s.disk.chain else s.disk.chain),
          S.reconFail x = false := by
        intro x hx; split_ifs at hx
        · rcases List.mem_cons.1 hx with rfl | hx
          · exact hl
          · exact h.chain_total x hx
        · exact h.chain_total x hx
      have ht : o.effect.tail s.disk.chainTail = .clean := by rw [h.chain_tail]; exact clean_tail he
      cases o with
      | ok => exact ⟨ht, h.replay_tail, h.records_ok, hchain, loadOK_after_set S h.procs trivial⟩
      | fail e => exact ⟨ht, h.replay_tail, h.records_ok, hchain, loadOK_after_set S h.procs trivial⟩
      | crash e => exact ⟨ht, h.replay_tail, h.records_ok, hchain, loadOK_after_set S h.procs trivial⟩
    · exact h
  | payBegin p b =>
    simp only [step]; split
    · simp only [Discipline.implemented, canTake, take, Bool.not_true, Bool.false_or, if_true,
        Bool.true_and]
      split_ifs
      all_goals first
        | exact h
        | exact loadInv_set S h trivial
        | exact loadInv_set S h (hadm.2 b ‹_›)
    · exact h
  | payWrite p o =>
    simp only [step]; split
    · rename_i m b hp
      have hl := h.procs p
      rw [hp] at hl
      have hrecs : ∀ r ∈ (if o.effect.adds then S.atoms b :: s.disk.replay else s.disk.replay),
          S.recordOK r = true := by
        intro r hr; split_ifs at hr
        · rcases List.mem_cons.1 hr with rfl | hr
          · exact hl
          · exact h.records_ok r hr
        · exact h.records_ok r hr
      have ht : o.effect.tail s.disk.replayTail = .clean := by rw [h.replay_tail]; exact clean_tail he
      cases o with
      | ok => exact ⟨h.chain_tail, ht, hrecs, h.chain_total, loadOK_after_set S h.procs trivial⟩
      | fail e => exact ⟨h.chain_tail, ht, hrecs, h.chain_total, loadOK_after_set S h.procs trivial⟩
      | crash e => exact ⟨h.chain_tail, ht, hrecs, h.chain_total, loadOK_after_set S h.procs trivial⟩
    · exact h
  | crash p => exact loadInv_set S h trivial
  | restart p =>
    simp only [step]; split
    · split_ifs
      · exact loadInv_set S h trivial
      · exact h
    · exact h
  | deleteReplay =>
    simp only [step]; split_ifs
    · exact ⟨h.chain_tail, rfl, by simp, h.chain_total, h.procs⟩
    · exact h

/-- **No self-inflicted denial of service.**  If acceptance only admits blocks whose replay
records the loader accepts and whose reconstruction cannot raise, then after any trace without
torn writes both durable files still load: no sequence of adversarial blocks can make the node
unable to restart or to accept. -/
theorem loadable_run (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (hadm : S.RecordAdmissible) (htot : S.ReconTotal) (es : List (Event B))
    (hclean : ∀ e ∈ es, CleanIO e) :
    S.chainLoads (run Discipline.implemented S ad es).disk = true ∧
      S.replayLoads (run Discipline.implemented S ad es).disk = true := by
  have key : ∀ (es : List (Event B)) (s : Sys B A), (∀ e ∈ es, CleanIO e) →
      Inv S ad s → LoadInv S s →
      Inv S ad (es.foldl (step Discipline.implemented S ad) s) ∧
        LoadInv S (es.foldl (step Discipline.implemented S ad) s) := by
    intro es
    induction es with
    | nil => intro s _ hi hl; exact ⟨hi, hl⟩
    | cons e es ih =>
      intro s hc hi hl
      exact ih _ (fun e' he' => hc e' (List.mem_cons_of_mem _ he'))
        (inv_step S hsound hrecon hi e) (loadInv_step S hadm htot hi hl e (hc e List.mem_cons_self))
  have hinit : LoadInv S (Sys.init : Sys B A) :=
    ⟨rfl, rfl, by simp [Sys.init], by simp [Sys.init], fun _ => trivial⟩
  obtain ⟨hi, hl⟩ := key es _ hclean (inv_init S) hinit
  simp only [run]
  constructor
  · simp only [AcceptSpec.chainLoads, Bool.and_eq_true, List.all_eq_true, Bool.not_eq_true']
    refine ⟨⟨by simp [hl.chain_tail], hi.chain_valid⟩, fun x hx => hl.chain_total x hx⟩
  · simp only [AcceptSpec.replayLoads, Bool.and_eq_true, List.all_eq_true]
    exact ⟨by simp [hl.replay_tail], fun r hr => hl.records_ok r hr⟩

end Liveness

/-! ## Concrete instance: `ots_ledger.py` + `server.py`, three revisions -/

/-- The JSON value of a signature's `one_time_key_id`.  Strings are named by codes: code `0` is
`""` and code `1` is `"None"`. -/
inductive RawKid where
  | null
  | str (code : ℕ)
  | num (n : ℕ)
  | arr (n : ℕ)
  deriving DecidableEq, Repr

namespace RawKid

/-- Python `str(kid)`: the string the replay ledger stores.  `str(None) == "None"`, so `null`
and the string `"None"` render alike.  Numbers and containers render to texts assumed distinct
from the strings used in a trace. -/
def render : RawKid → ℕ
  | null => 1
  | str c => c
  | num n => 1000000 + n
  | arr n => 2000000 + n

/-- The raw value can be put in a Python `set` (`verify_ots_chain`). -/
def hashable : RawKid → Bool
  | arr _ => false
  | _ => true

/-- `isinstance(kid, str) and kid != ""`, the identity typing of replay records. -/
def nonEmptyString : RawKid → Bool
  | str c => c != 0
  | _ => false

end RawKid

/-- What acceptance reads from a received OTS block. -/
structure CBlock where
  /-- `Block.index` -/
  index : ℕ
  /-- `previous_hash`, named by the parent's hash code (`0` for sixty-four zeros) -/
  parent : ℕ
  /-- `Block.hash` -/
  hash : ℕ
  /-- `meta.ots_auth.signature.one_time_key_id` (raw JSON) -/
  kid : RawKid
  /-- `public_key_fingerprint` (64 lowercase hex after verification; never `"None"`) -/
  fp : ℕ
  /-- `ots_material_id(public_key)`: the commitment table's recomputed Merkle root (not the
  block's own message `merkle_root`) -/
  material : ℕ
  /-- `verify_block_integrity ∧ verify_ots_block` (signature, transcript binding, header context) -/
  authentic : Bool
  deriving DecidableEq, Repr

def cTip : List CBlock → ℕ
  | [] => 0
  | t :: _ => t.hash

/-- `_verify_pow_and_linkage` / `verify_ots_chain` linkage on a newest-first chain. -/
def cLink (c : List CBlock) (b : CBlock) : Bool := b.index == c.length + 1 && b.parent == cTip c

/-- One identity consumed by the replay ledger. -/
inductive Atom where
  /-- `str(one_time_key_id)` -/
  | kid (k : ℕ)
  /-- `public_key_fingerprint` -/
  | fp (f : ℕ)
  /-- `ots_material_id(public_key)` -/
  | mat (m : ℕ)
  deriving DecidableEq, Repr

/-- A revision of the acceptance code. -/
structure PyRev where
  /-- PR #47: replay identities must be strings; reconstruction keeps the key id `"None"` -/
  typed : Bool
  /-- this change: the commitment table (`ots_material_id`) is consumed too -/
  material : Bool
  deriving DecidableEq, Repr

/-- `master` before this work. -/
def PyRev.b310ea5 : PyRev := ⟨false, false⟩
/-- PR #47 before this change. -/
def PyRev.pr47 : PyRev := ⟨true, false⟩
/-- This change. -/
def PyRev.current : PyRev := ⟨true, true⟩

/-- The atoms a revision consumes for a signature. -/
def pyAtoms (rev : PyRev) (b : CBlock) : List Atom :=
  [.kid b.kid.render, .fp b.fp] ++ if rev.material then [.mat b.material] else []

/-- `index_signature`'s handling of the key id: at `b310ea5` it skipped the *string* `"None"`;
from PR #47 on it skips only an absent value. -/
def reconKid (rev : PyRev) (k : RawKid) : List Atom :=
  if rev.typed then
    match k with
    | .null => []
    | k => [.kid k.render]
  else if k.render = 1 then [] else [.kid k.render]

/-- `index_signature` (with the carried public key in the current revision). -/
def pyRecon (rev : PyRev) (b : CBlock) : List Atom :=
  reconKid rev b.kid ++ [.fp b.fp] ++ if rev.material then [.mat b.material] else []

/-- The replay loader's per-atom schema: a key id must be a non-empty string. -/
def recordAtomOK : Atom → Bool
  | .kid k => k != 0
  | _ => true

/-- The Python acceptance protocol of revision `rev`. -/
def pySpec (rev : PyRev) : AcceptSpec CBlock Atom where
  atoms := pyAtoms rev
  blockOK b := b.authentic && b.kid.hashable
  linkOK := cLink
  acceptOK c b := b.authentic && cLink c b && (!rev.typed || b.kid.nonEmptyString)
  payOK b := b.authentic && (!rev.typed || b.kid.nonEmptyString)
  recon := pyRecon rev
  reconFail b := b.kid == .null
  recordOK r := r.all recordAtomOK

theorem nonEmptyString_str {k : RawKid} (h : k.nonEmptyString = true) : ∃ c, k = .str c ∧ c ≠ 0 := by
  cases k with
  | str c => exact ⟨c, rfl, by simpa [RawKid.nonEmptyString] using h⟩
  | _ => simp [RawKid.nonEmptyString] at h

theorem typed_accept {rev : PyRev} (hrev : rev.typed = true) {c : List CBlock} {b : CBlock}
    (h : (pySpec rev).acceptOK c b = true) :
    b.authentic = true ∧ cLink c b = true ∧ ∃ k, b.kid = .str k ∧ k ≠ 0 := by
  simp only [pySpec, hrev, Bool.not_true, Bool.false_or, Bool.and_eq_true] at h
  exact ⟨h.1.1, h.1.2, nonEmptyString_str h.2⟩

theorem pySpec_acceptSound {rev : PyRev} (hrev : rev.typed = true) : (pySpec rev).AcceptSound := by
  intro c b h
  obtain ⟨ha, hl, k, hk, _⟩ := typed_accept hrev h
  exact ⟨hl, by simp [pySpec, ha, hk, RawKid.hashable]⟩

theorem pySpec_losslessRecon {rev : PyRev} (hrev : rev.typed = true) :
    (pySpec rev).LosslessRecon := by
  intro c b h a ha
  obtain ⟨_, _, k, hk, _⟩ := typed_accept hrev h
  simp only [pySpec, pyAtoms, List.mem_append, List.mem_cons, List.not_mem_nil, or_false] at ha
  simp only [pySpec, pyRecon, reconKid, hrev, if_true, hk, List.mem_append, List.mem_cons,
    List.not_mem_nil, or_false]
  rcases ha with (rfl | rfl) | ha
  · left; left; simp [hk]
  · left; right; rfl
  · right; exact ha

theorem pySpec_recordAdmissible {rev : PyRev} (hrev : rev.typed = true) :
    (pySpec rev).RecordAdmissible := by
  have key : ∀ b : CBlock, (∃ k, b.kid = .str k ∧ k ≠ 0) → (pySpec rev).recordOK (pyAtoms rev b) = true := by
    intro b ⟨k, hk, hk0⟩
    simp only [pySpec, pyAtoms, List.all_eq_true, List.mem_append, List.mem_cons, List.not_mem_nil,
      or_false]
    intro a ha
    rcases ha with (rfl | rfl) | ha
    · simp [recordAtomOK, hk, RawKid.render, hk0]
    · rfl
    · split_ifs at ha
      · simp only [List.mem_cons, List.not_mem_nil, or_false] at ha; subst ha; rfl
      · simp at ha
  constructor
  · intro c b h; exact key b (typed_accept hrev h).2.2
  · intro b h
    simp only [pySpec, hrev, Bool.not_true, Bool.false_or, Bool.and_eq_true] at h
    exact key b (nonEmptyString_str h.2)

theorem pySpec_reconTotal {rev : PyRev} (hrev : rev.typed = true) : (pySpec rev).ReconTotal := by
  intro c b h
  obtain ⟨_, _, k, hk, _⟩ := typed_accept hrev h
  simp [pySpec, hk]

/-- **Single use of every consumed identity (implementation model).**  For PR #47 and for this
change, every finite same-host trace — any interleaving of processes, write failures, crashes at
any modelled point, restarts, and (with `ad = true`) loss of the side replay file — leaves a
durable chain in which no two blocks share a consumed atom. -/
theorem wavelock_ots_single_use (rev : PyRev) (hrev : rev.typed = true) (ad : Bool)
    (es : List (Event CBlock)) :
    (run Discipline.implemented (pySpec rev) ad es).disk.chain.Pairwise (Disjoint (pySpec rev)) :=
  chain_single_use _ (pySpec_acceptSound hrev) (Or.inl (pySpec_losslessRecon hrev)) es

/-- The PR #47 statement: no `one_time_key_id` and no `public_key_fingerprint` twice. -/
theorem wavelock_ots_chain_single_use (rev : PyRev) (hrev : rev.typed = true) (ad : Bool)
    (es : List (Event CBlock)) :
    ((run Discipline.implemented (pySpec rev) ad es).disk.chain.map fun b => b.kid.render).Nodup ∧
      ((run Discipline.implemented (pySpec rev) ad es).disk.chain.map CBlock.fp).Nodup := by
  have h := wavelock_ots_single_use rev hrev ad es
  refine ⟨nodup_of_pairwise _ _ Atom.kid (fun x => ?_) h, nodup_of_pairwise _ _ Atom.fp (fun x => ?_) h⟩
  · simp [pySpec, pyAtoms]
  · simp [pySpec, pyAtoms]

/-- **Single use of the one-time key material (this change).**  No two blocks of the durable
chain carry the same commitment table (`ots_material_id`), whatever labels they carry. -/
theorem wavelock_ots_material_single_use (ad : Bool) (es : List (Event CBlock)) :
    ((run Discipline.implemented (pySpec .current) ad es).disk.chain.map CBlock.material).Nodup :=
  nodup_of_pairwise _ _ Atom.mat (fun x => by simp [pySpec, pyAtoms, PyRev.current])
    (wavelock_ots_single_use .current rfl ad es)

/-- For PR #47 and this change no sequence of blocks or payload signatures can make the node's
own files unloadable; only a torn write (or hostile storage) can. -/
theorem wavelock_ots_no_self_dos (rev : PyRev) (hrev : rev.typed = true) (ad : Bool)
    (es : List (Event CBlock)) (hclean : ∀ e ∈ es, CleanIO e) :
    (pySpec rev).chainLoads (run Discipline.implemented (pySpec rev) ad es).disk = true ∧
      (pySpec rev).replayLoads (run Discipline.implemented (pySpec rev) ad es).disk = true :=
  loadable_run _ (pySpec_acceptSound hrev) (Or.inl (pySpec_losslessRecon hrev))
    (pySpec_recordAdmissible hrev) (pySpec_reconTotal hrev) es hclean

/-! ### The three consumed sets `(K, F, M)` -/

/-- Projections of consumed atoms onto key ids, fingerprints and commitment tables. -/
def Atom.kid? : Atom → Option ℕ
  | .kid k => some k
  | _ => none

def Atom.fp? : Atom → Option ℕ
  | .fp f => some f
  | _ => none

def Atom.mat? : Atom → Option ℕ
  | .mat m => some m
  | _ => none

/-- `K_s`, `F_s`, `M_s`: the consumed key ids, fingerprints and commitment tables of a durable
state (accepted chain and replay records). -/
def consumedK (d : Disk CBlock Atom) : List ℕ := ((pySpec .current).consumed d).filterMap Atom.kid?
def consumedF (d : Disk CBlock Atom) : List ℕ := ((pySpec .current).consumed d).filterMap Atom.fp?
def consumedM (d : Disk CBlock Atom) : List ℕ := ((pySpec .current).consumed d).filterMap Atom.mat?

theorem atoms_current (b : CBlock) :
    (pySpec .current).atoms b = [.kid b.kid.render, .fp b.fp, .mat b.material] := by
  simp [pySpec, pyAtoms, PyRev.current]

/-- **`Fresh(s, b)` at consumption.**  In every trace of the current protocol, a process about to
durably consume `b` does so only when `K(b) ∉ K_s`, `F(b) ∉ F_s` and `M(b) ∉ M_s` for the
authoritative state `s` at that instant. -/
theorem wavelock_ots_fresh_triple (ad : Bool) (es : List (Event CBlock)) {p : ℕ} {m : List Atom}
    {b : CBlock} (hp : (run Discipline.implemented (pySpec .current) ad es).procs p = .checked m b) :
    b.kid.render ∉ consumedK (run Discipline.implemented (pySpec .current) ad es).disk ∧
      b.fp ∉ consumedF (run Discipline.implemented (pySpec .current) ad es).disk ∧
      b.material ∉ consumedM (run Discipline.implemented (pySpec .current) ad es).disk := by
  have h := consume_fresh (pySpec .current) (pySpec_acceptSound rfl)
    (Or.inl (pySpec_losslessRecon rfl)) es hp
  rw [atoms_current] at h
  refine ⟨fun hk => ?_, fun hf => ?_, fun hm => ?_⟩
  · obtain ⟨a, ha, hak⟩ := List.mem_filterMap.1 hk
    cases a <;> simp only [Atom.kid?, Option.some.injEq, reduceCtorEq] at hak
    subst hak; exact h _ (by simp) ha
  · obtain ⟨a, ha, haf⟩ := List.mem_filterMap.1 hf
    cases a <;> simp only [Atom.fp?, Option.some.injEq, reduceCtorEq] at haf
    subst haf; exact h _ (by simp) ha
  · obtain ⟨a, ha, ham⟩ := List.mem_filterMap.1 hm
    cases a <;> simp only [Atom.mat?, Option.some.injEq, reduceCtorEq] at ham
    subst ham; exact h _ (by simp) ha

/-- **The consumption transition records all three at once.**  One record append whose bytes land
puts `K(b)`, `F(b)` and `M(b)` into the consumed sets. -/
theorem wavelock_ots_records_triple (D : Discipline) (ad : Bool) {s : Sys CBlock Atom} {p : ℕ}
    {m : List Atom} {b : CBlock} (hp : s.procs p = .checked m b) {o : WriteOutcome}
    (ho : o.effect.adds = true) :
    b.kid.render ∈ consumedK (step D (pySpec .current) ad s (.writeRecord p o)).disk ∧
      b.fp ∈ consumedF (step D (pySpec .current) ad s (.writeRecord p o)).disk ∧
      b.material ∈ consumedM (step D (pySpec .current) ad s (.writeRecord p o)).disk := by
  have h := writeRecord_consumes_all (pySpec .current) (ad := ad) D hp ho
  rw [atoms_current] at h
  exact ⟨List.mem_filterMap.2 ⟨.kid b.kid.render, h _ (by simp), rfl⟩,
    List.mem_filterMap.2 ⟨.fp b.fp, h _ (by simp), rfl⟩,
    List.mem_filterMap.2 ⟨.mat b.material, h _ (by simp), rfl⟩⟩

/-! ### Trace helpers -/

/-- An authentic block at chain position `i` on parent hash `par`, with hash `h`, key id `k`,
fingerprint `f` and key material `mat`. -/
def blk (i par h : ℕ) (k : RawKid) (f mat : ℕ) : CBlock := ⟨i, par, h, k, f, mat, true⟩

/-- One complete, fault-free `try_accept_block` by process `p`. -/
def acceptAll (p : ℕ) (b : CBlock) : List (Event CBlock) :=
  [.begin p b, .ledger p, .writeRecord p .ok, .writeBlock p .ok]

/-- `try_accept_block` with explicit outcomes for the record and block appends. -/
def acceptWith (p : ℕ) (b : CBlock) (rec blkOut : WriteOutcome) : List (Event CBlock) :=
  [.begin p b, .ledger p, .writeRecord p rec, .writeBlock p blkOut]

/-- The reported acceptance of an event, if any (`try_accept_block` returned `True`). -/
def acceptedBlock? {B A : Type} (s : Sys B A) : Event B → Option B
  | .writeBlock p .ok => match s.procs p with
    | .recorded _ b => some b
    | _ => none
  | _ => none

/-- Run a trace and collect every reported acceptance, in order. -/
def runLog {B A : Type} [DecidableEq A] (D : Discipline) (S : AcceptSpec B A)
    (ad : Bool) (s : Sys B A) (es : List (Event B)) : Sys B A × List B :=
  es.foldl (fun (acc : Sys B A × List B) e =>
    (step D S ad acc.1 e, acc.2 ++ (acceptedBlock? acc.1 e).toList)) (s, [])

/-- Key-id, fingerprint and material codes of a chain, oldest last. -/
def kidCodes (c : List CBlock) : List ℕ := c.map fun b => b.kid.render
def fpCodes (c : List CBlock) : List ℕ := c.map CBlock.fp
def materialCodes (c : List CBlock) : List ℕ := c.map CBlock.material

end OTS
end WaveLockProofs
