import Mathlib.Tactic

/-!
# WaveLock-OTS identity consumption: same-host block acceptance as a transition system

Frozen source: `rickyjreyes/Wavelock` master `b310ea5`, plus the narrowly scoped ledger
correction in this change (`PersistentOTSReplayLedger.accept` identity typing and the
`index_signature` sentinel). Protocol constants: scheme `WaveLock-OTS-v1`, signature/public-key
`version = 1`, replay `LEDGER_VERSION = 1`, block transcript `OTS_BLOCK_TRANSCRIPT_VERSION = 1`,
block header version 2 required on the OTS path.

## What is modelled (Python reference)

* `wavelock/network/server.py :: try_accept_block` under `ChainState.acceptance_lock`
  (`chain-writer.lock`, `fcntl.flock` / SQLite): reload `load_all_blocks`, `verify_ots_chain`,
  `_reconstruct_consumed_ots`, `_verify_pow_and_linkage`, `_verify_ots_block`, then
  `CHAIN.append` → `save_block_to_disk`.  Event `begin` (lock + reload + pure checks),
  `ledger`, `writeRecord`, `writeBlock`.
* `wavelock/crypto/ots_ledger.py :: PersistentOTSReplayLedger.accept`: under the ledger lock
  (`<path>.lock`), `_scan_file_into_sets` (additive re-scan, fails closed on a malformed record),
  duplicate test on `one_time_key_id` **or** `public_key_fingerprint`, `_append`
  (`write; flush; fsync; fsync(dir)`), then the in-memory update.  Event `ledger` and
  `writeRecord`; the standalone `server.verify_ots_payload` path is `payBegin`/`payWrite`.
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
* Identity values are the strings the replay ledger stores (`str(...)` of the signature fields);
  the concrete instance below records where Python's raw-value and string views differ.
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

/-- Durable state shared by every process on the host.  Lists are newest-first. -/
structure Disk (B K F : Type) where
  chain : List B
  chainTail : Tail
  replay : List (K × F)
  replayTail : Tail

/-- The two in-memory consumed sets of one `PersistentOTSReplayLedger` instance. -/
structure Mem (K F : Type) where
  keys : List K
  leaves : List F

namespace Mem
variable {K F : Type}

def empty : Mem K F := ⟨[], []⟩

def union (m n : Mem K F) : Mem K F := ⟨m.keys ++ n.keys, m.leaves ++ n.leaves⟩

def add (m : Mem K F) (i : K × F) : Mem K F := ⟨i.1 :: m.keys, i.2 :: m.leaves⟩

/-- `kid in self._key_ids or leaf in self._leaf_ids`. -/
def hits [DecidableEq K] [DecidableEq F] (m : Mem K F) (i : K × F) : Bool :=
  decide (i.1 ∈ m.keys) || decide (i.2 ∈ m.leaves)

end Mem

/-- Records of the replay file as consumed sets (`_scan_file_into_sets`). -/
def fileMem {K F : Type} (rs : List (K × F)) : Mem K F := ⟨rs.map Prod.fst, rs.map Prod.snd⟩

/-! ## Pure parameters of the protocol -/

/-- The stateless part of acceptance.  `ident b` is `(one_time_key_id, public_key_fingerprint)`
as the replay ledger stores them; `linkOK`/`blockOK` are the per-block conditions of
`verify_ots_chain`; `acceptOK` is every pure check performed by `try_accept_block` and by the
ledger before its duplicate test; `recon` is `index_signature`; `reconFail` is the
`_reconstruct_consumed_ots` raise condition; `recordOK` is the per-record schema of
`_scan_file_into_sets`. -/
structure AcceptSpec (B K F : Type) where
  ident : B → K × F
  blockOK : B → Bool
  linkOK : List B → B → Bool
  acceptOK : List B → B → Bool
  payOK : B → Bool
  recon : B → Mem K F
  reconFail : B → Bool
  recordOK : K × F → Bool

namespace AcceptSpec

variable {B K F : Type} [DecidableEq K] [DecidableEq F] (S : AcceptSpec B K F)

def kids (c : List B) : List K := c.map fun b => (S.ident b).1

def fps (c : List B) : List F := c.map fun b => (S.ident b).2

/-- Neither identity of `b` occurs in the chain `c`. -/
def fresh (c : List B) (b : B) : Bool :=
  !decide ((S.ident b).1 ∈ S.kids c) && !decide ((S.ident b).2 ∈ S.fps c)

/-- `verify_ots_chain` on a newest-first chain: linkage, per-block verification and pairwise
distinct `one_time_key_id` and `public_key_fingerprint`. -/
def chainValid : List B → Bool
  | [] => true
  | b :: c => chainValid c && S.linkOK c b && S.blockOK b && S.fresh c b

/-- `_reconstruct_consumed_ots`: fold `index_signature` over the chain. -/
def reconAll (c : List B) : Mem K F :=
  ⟨c.flatMap fun b => (S.recon b).keys, c.flatMap fun b => (S.recon b).leaves⟩

/-- `CHAIN.load_from_disk(require_ots=True)` succeeds. -/
def chainLoads (d : Disk B K F) : Bool :=
  d.chainTail != .garbage && S.chainValid d.chain && d.chain.all fun b => !S.reconFail b

/-- `_scan_file_into_sets` succeeds. -/
def replayLoads (d : Disk B K F) : Bool :=
  d.replayTail != .garbage && d.replay.all S.recordOK

/-- Authoritative consumed identities of a durable state: accepted chain plus replay records. -/
def consumedKeys (d : Disk B K F) : List K := S.kids d.chain ++ d.replay.map Prod.fst

def consumedLeaves (d : Disk B K F) : List F := S.fps d.chain ++ d.replay.map Prod.snd

/-- Every pure acceptance check implies the chain validator's per-block conditions. -/
def AcceptSound : Prop := ∀ c b, S.acceptOK c b = true → S.linkOK c b = true ∧ S.blockOK b = true

/-- `index_signature` folds back both identities of every block that acceptance admits. -/
def LosslessRecon : Prop :=
  ∀ c b, S.acceptOK c b = true → (S.ident b).1 ∈ (S.recon b).keys ∧ (S.ident b).2 ∈ (S.recon b).leaves

/-- Acceptance never writes a replay record that the loader rejects. -/
def RecordAdmissible : Prop :=
  (∀ c b, S.acceptOK c b = true → S.recordOK (S.ident b) = true) ∧
    ∀ b, S.payOK b = true → S.recordOK (S.ident b) = true

/-- Reconstruction never raises on a block that acceptance admits. -/
def ReconTotal : Prop := ∀ c b, S.acceptOK c b = true → S.reconFail b = false

end AcceptSpec

/-- No record of the replay file repeats a key id or a fingerprint of an older record. -/
def replayUnique {K F : Type} [DecidableEq K] [DecidableEq F] : List (K × F) → Bool
  | [] => true
  | r :: rs => replayUnique rs && !decide (r.1 ∈ rs.map Prod.fst) && !decide (r.2 ∈ rs.map Prod.snd)

/-! ## Processes, locks and events -/

/-- Control state of one process.  `loaded`/`checked`/`recorded` hold the chain-writer lock;
`checked`/`payChecked` hold the ledger lock. -/
inductive Phase (B K F : Type) where
  | dead
  | idle (m : Mem K F)
  | loaded (m : Mem K F) (b : B)
  | checked (m : Mem K F) (b : B)
  | recorded (m : Mem K F) (b : B)
  | payChecked (m : Mem K F) (b : B)

namespace Phase
variable {B K F : Type}

def holdsChain : Phase B K F → Bool
  | loaded .. | checked .. | recorded .. => true
  | _ => false

def holdsLedger : Phase B K F → Bool
  | checked .. | payChecked .. => true
  | _ => false

end Phase

structure Sys (B K F : Type) where
  disk : Disk B K F
  procs : ℕ → Phase B K F
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
variable {B K F : Type}

def set (s : Sys B K F) (p : ℕ) (v : Phase B K F) : ℕ → Phase B K F :=
  fun q => if q = p then v else s.procs q

def init : Sys B K F := ⟨⟨[], .clean, [], .clean⟩, fun _ => .idle Mem.empty, none, none⟩

end Sys

section Step
variable {B K F : Type} [DecidableEq K] [DecidableEq F]

/-- One step.  Events whose process is not in the matching phase, or whose lock is busy, leave
the state unchanged. -/
def step (D : Discipline) (S : AcceptSpec B K F) (allowDelete : Bool) (s : Sys B K F) :
    Event B → Sys B K F
  | .begin p b =>
    match s.procs p with
    | .idle m =>
      if canTake D.chainLock s.chainLock then
        if S.chainLoads s.disk then
          let m' := m.union (S.reconAll s.disk.chain)
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
          let m' := if D.rescan then m.union (fileMem s.disk.replay) else m
          if m'.hits (S.ident b) then
            { s with procs := s.set p (.idle m'), chainLock := release p s.chainLock }
          else
            { s with procs := s.set p (.checked m' b), ledgerLock := take D.ledgerLock p s.ledgerLock }
      else s
    | _ => s
  | .writeRecord p o =>
    match s.procs p with
    | .checked m b =>
      let d' : Disk B K F :=
        { s.disk with
          replay := if o.effect.adds then S.ident b :: s.disk.replay else s.disk.replay
          replayTail := o.effect.tail s.disk.replayTail }
      match o with
      | .ok =>
        { s with disk := d', procs := s.set p (.recorded (m.add (S.ident b)) b),
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
      let d' : Disk B K F :=
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
            let m' := if D.rescan then m.union (fileMem s.disk.replay) else m
            if m'.hits (S.ident b) then { s with procs := s.set p (.idle m') }
            else
              { s with procs := s.set p (.payChecked m' b),
                       ledgerLock := take D.ledgerLock p s.ledgerLock }
        else s
      else s
    | _ => s
  | .payWrite p o =>
    match s.procs p with
    | .payChecked m b =>
      let d' : Disk B K F :=
        { s.disk with
          replay := if o.effect.adds then S.ident b :: s.disk.replay else s.disk.replay
          replayTail := o.effect.tail s.disk.replayTail }
      match o with
      | .ok => { s with disk := d', procs := s.set p (.idle (m.add (S.ident b))),
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
        { s with procs := s.set p (.idle ((fileMem s.disk.replay).union (S.reconAll s.disk.chain))) }
      else s
    | _ => s
  | .deleteReplay =>
    if allowDelete then { s with disk := { s.disk with replay := [], replayTail := .clean } } else s

/-- Execute a finite trace from the empty host. -/
def run (D : Discipline) (S : AcceptSpec B K F) (allowDelete : Bool) (es : List (Event B)) :
    Sys B K F :=
  es.foldl (step D S allowDelete) Sys.init

end Step


/-! ## The safety invariant -/

section Invariant
variable {B K F : Type} [DecidableEq K] [DecidableEq F] (S : AcceptSpec B K F)

/-- Every chain identity is in the process's memory, or — when the side replay file cannot be
lost (`ad = false`) — in the replay file. -/
def Covers (ad : Bool) (m : Mem K F) (d : Disk B K F) : Prop :=
  (∀ k ∈ S.kids d.chain, k ∈ m.keys ∨ (ad = false ∧ k ∈ d.replay.map Prod.fst)) ∧
  (∀ f ∈ S.fps d.chain, f ∈ m.leaves ∨ (ad = false ∧ f ∈ d.replay.map Prod.snd))

/-- What a process in each phase knows about the current durable state. -/
def ProcOK (ad : Bool) (d : Disk B K F) : Phase B K F → Prop
  | .dead => True
  | .idle _ => True
  | .loaded m b => S.acceptOK d.chain b = true ∧ Covers S ad m d
  | .checked _ b => S.acceptOK d.chain b = true ∧ S.fresh d.chain b = true ∧
      (S.ident b).1 ∉ d.replay.map Prod.fst ∧ (S.ident b).2 ∉ d.replay.map Prod.snd
  | .recorded _ b => S.acceptOK d.chain b = true ∧ S.fresh d.chain b = true ∧
      (ad = false → (S.ident b).1 ∈ d.replay.map Prod.fst ∧ (S.ident b).2 ∈ d.replay.map Prod.snd)
  | .payChecked _ b =>
      (S.ident b).1 ∉ d.replay.map Prod.fst ∧ (S.ident b).2 ∉ d.replay.map Prod.snd

/-- The inductive invariant of the implemented system. -/
structure Inv (ad : Bool) (s : Sys B K F) : Prop where
  chain_valid : S.chainValid s.disk.chain = true
  replay_unique : replayUnique s.disk.replay = true
  chain_admitted : ∀ x ∈ s.disk.chain, ∃ c, S.acceptOK c x = true
  chain_in_replay : ad = false →
    (∀ k ∈ S.kids s.disk.chain, k ∈ s.disk.replay.map Prod.fst) ∧
    (∀ f ∈ S.fps s.disk.chain, f ∈ s.disk.replay.map Prod.snd)
  chain_lock : ∀ p, s.chainLock = some p ↔ (s.procs p).holdsChain = true
  ledger_lock : ∀ p, s.ledgerLock = some p ↔ (s.procs p).holdsLedger = true
  procs_ok : ∀ p, ProcOK S ad s.disk (s.procs p)

/-! ### Small lemmas -/

omit [DecidableEq K] [DecidableEq F] in
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

omit [DecidableEq K] [DecidableEq F] in
theorem Sys.set_self {s : Sys B K F} {p : ℕ} {v : Phase B K F} : s.set p v p = v := by
  simp [Sys.set]

omit [DecidableEq K] [DecidableEq F] in
theorem Sys.set_ne {s : Sys B K F} {p q : ℕ} (h : q ≠ p) {v : Phase B K F} :
    s.set p v q = s.procs q := by
  simp [Sys.set, h]

omit [DecidableEq K] [DecidableEq F] in
/-- Lock bookkeeping after one process changes phase. -/
theorem lock_after_set {procs : ℕ → Phase B K F} {f : Phase B K F → Bool} {l l' : Option ℕ}
    {p : ℕ} {v : Phase B K F}
    (hl : ∀ q, l = some q ↔ f (procs q) = true)
    (hp : l' = some p ↔ f v = true)
    (hother : ∀ q, q ≠ p → (l' = some q ↔ l = some q)) :
    ∀ q, l' = some q ↔ f (if q = p then v else procs q) = true := by
  intro q
  by_cases hq : q = p
  · subst hq; simpa using hp
  · simp only [hq, if_false]; rw [hother q hq]; exact hl q

omit [DecidableEq K] [DecidableEq F] in
theorem mem_kids {c : List B} {k : K} : k ∈ S.kids c ↔ ∃ x ∈ c, (S.ident x).1 = k := by
  simp [AcceptSpec.kids]

omit [DecidableEq K] [DecidableEq F] in
theorem mem_fps {c : List B} {f : F} : f ∈ S.fps c ↔ ∃ x ∈ c, (S.ident x).2 = f := by
  simp [AcceptSpec.fps]

theorem fresh_iff {c : List B} {b : B} :
    S.fresh c b = true ↔ (S.ident b).1 ∉ S.kids c ∧ (S.ident b).2 ∉ S.fps c := by
  simp [AcceptSpec.fresh]

theorem chainValid_cons {c : List B} {b : B} :
    S.chainValid (b :: c) = true ↔
      S.chainValid c = true ∧ S.linkOK c b = true ∧ S.blockOK b = true ∧ S.fresh c b = true := by
  simp [AcceptSpec.chainValid, Bool.and_eq_true, and_assoc]

theorem replayUnique_cons {r : K × F} {rs : List (K × F)} :
    replayUnique (r :: rs) = true ↔
      replayUnique rs = true ∧ r.1 ∉ rs.map Prod.fst ∧ r.2 ∉ rs.map Prod.snd := by
  simp [replayUnique, Bool.and_eq_true, and_assoc]

omit [DecidableEq K] [DecidableEq F] in
theorem mem_reconAll {c : List B} {x : B} (hx : x ∈ c) :
    (∀ k ∈ (S.recon x).keys, k ∈ (S.reconAll c).keys) ∧
      ∀ f ∈ (S.recon x).leaves, f ∈ (S.reconAll c).leaves := by
  constructor
  · intro k hk; simp only [AcceptSpec.reconAll, List.mem_flatMap]; exact ⟨x, hx, hk⟩
  · intro f hf; simp only [AcceptSpec.reconAll, List.mem_flatMap]; exact ⟨x, hx, hf⟩

/-! ### Process facts under changes made by another process -/

theorem procOK_replay_super {ad : Bool} {d : Disk B K F} {ph : Phase B K F}
    (rs : List (K × F)) (t : Tail) (hsub : ∀ r ∈ d.replay, r ∈ rs)
    (hph : ph.holdsLedger = false) (h : ProcOK S ad d ph) :
    ProcOK S ad { d with replay := rs, replayTail := t } ph := by
  have hk : ∀ k, k ∈ d.replay.map Prod.fst → k ∈ rs.map Prod.fst := by
    intro k hk; simp only [List.mem_map] at hk ⊢
    obtain ⟨r, hr, rfl⟩ := hk; exact ⟨r, hsub r hr, rfl⟩
  have hf : ∀ f, f ∈ d.replay.map Prod.snd → f ∈ rs.map Prod.snd := by
    intro f hf; simp only [List.mem_map] at hf ⊢
    obtain ⟨r, hr, rfl⟩ := hf; exact ⟨r, hsub r hr, rfl⟩
  cases ph with
  | dead => trivial
  | idle m => trivial
  | loaded m b =>
    obtain ⟨hacc, hck, hcf⟩ := h
    refine ⟨hacc, ?_, ?_⟩
    · intro k hk'
      rcases hck k hk' with h1 | ⟨h2, h3⟩
      · exact Or.inl h1
      · exact Or.inr ⟨h2, hk k h3⟩
    · intro f hf'
      rcases hcf f hf' with h1 | ⟨h2, h3⟩
      · exact Or.inl h1
      · exact Or.inr ⟨h2, hf f h3⟩
  | checked m b => simp [Phase.holdsLedger] at hph
  | recorded m b =>
    obtain ⟨hacc, hfr, hin⟩ := h
    exact ⟨hacc, hfr, fun had => ⟨hk _ (hin had).1, hf _ (hin had).2⟩⟩
  | payChecked m b => simp [Phase.holdsLedger] at hph

theorem procOK_replay_delete {d : Disk B K F} {ph : Phase B K F} (h : ProcOK S true d ph) :
    ProcOK S true { d with replay := [], replayTail := .clean } ph := by
  cases ph with
  | dead => trivial
  | idle m => trivial
  | loaded m b =>
    obtain ⟨hacc, hck, hcf⟩ := h
    refine ⟨hacc, ?_, ?_⟩
    · intro k hk'
      rcases hck k hk' with h1 | ⟨h2, _⟩
      · exact Or.inl h1
      · exact absurd h2 (by decide)
    · intro f hf'
      rcases hcf f hf' with h1 | ⟨h2, _⟩
      · exact Or.inl h1
      · exact absurd h2 (by decide)
  | checked m b =>
    obtain ⟨hacc, hfr, _, _⟩ := h
    exact ⟨hacc, hfr, by simp, by simp⟩
  | recorded m b =>
    obtain ⟨hacc, hfr, _⟩ := h
    exact ⟨hacc, hfr, fun h => absurd h (by decide)⟩
  | payChecked m b => exact ⟨by simp, by simp⟩

theorem procOK_chain_change {ad : Bool} {d : Disk B K F} {ph : Phase B K F}
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
variable {B K F : Type} [DecidableEq K] [DecidableEq F] (S : AcceptSpec B K F) {ad : Bool}

theorem procs_after_set {d : Disk B K F} {procs : ℕ → Phase B K F} {p : ℕ} {v : Phase B K F}
    (hall : ∀ q, q ≠ p → ProcOK S ad d (procs q)) (hv : ProcOK S ad d v) :
    ∀ q, ProcOK S ad d (if q = p then v else procs q) := by
  intro q
  by_cases hq : q = p
  · simp only [hq, if_true]; exact hv
  · simp only [hq, if_false]; exact hall q hq

/-- `release` never leaves the releasing process as owner. -/
theorem release_ne_self {p : ℕ} {l : Option ℕ} : ¬ release p l = some p := by
  rw [release_eq_some]; exact fun h => h.2 rfl

theorem release_other {p q : ℕ} {l : Option ℕ} (hq : q ≠ p) : release p l = some q ↔ l = some q := by
  rw [release_eq_some]; exact ⟨fun h => h.1, fun h => ⟨h, hq⟩⟩

omit [DecidableEq K] [DecidableEq F] in
theorem not_holds_of_owner {procs : ℕ → Phase B K F} {f : Phase B K F → Bool} {l : Option ℕ}
    (hl : ∀ q, l = some q ↔ f (procs q) = true) {p q : ℕ} (hp : l = some p) (hq : q ≠ p) :
    f (procs q) = false := by
  by_contra hc
  have : l = some q := (hl q).2 (by simpa using hc)
  rw [hp] at this; exact hq (Option.some.inj this).symm

theorem covers_of_inv (hrecon : S.LosslessRecon ∨ ad = false) {s : Sys B K F} (h : Inv S ad s)
    (m : Mem K F) : Covers S ad (m.union (S.reconAll s.disk.chain)) s.disk := by
  constructor
  · intro k hk
    rcases hrecon with hl | had
    · left
      obtain ⟨x, hx, rfl⟩ := (mem_kids S).1 hk
      obtain ⟨c, hc⟩ := h.chain_admitted x hx
      simp only [Mem.union, List.mem_append]
      exact Or.inr ((mem_reconAll S hx).1 _ (hl c x hc).1)
    · exact Or.inr ⟨had, (h.chain_in_replay had).1 k hk⟩
  · intro f hf
    rcases hrecon with hl | had
    · left
      obtain ⟨x, hx, rfl⟩ := (mem_fps S).1 hf
      obtain ⟨c, hc⟩ := h.chain_admitted x hx
      simp only [Mem.union, List.mem_append]
      exact Or.inr ((mem_reconAll S hx).2 _ (hl c x hc).2)
    · exact Or.inr ⟨had, (h.chain_in_replay had).2 f hf⟩

theorem inv_begin (hrecon : S.LosslessRecon ∨ ad = false) {s : Sys B K F} (h : Inv S ad s)
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

theorem inv_ledger {s : Sys B K F} (h : Inv S ad s) (p : ℕ) :
    Inv S ad (step Discipline.implemented S ad s (.ledger p)) := by
  simp only [step]
  split
  · rename_i m b hp
    have hown : s.chainLock = some p := (h.chain_lock p).2 (by rw [hp]; rfl)
    have hnl : s.ledgerLock ≠ some p := by
      intro hc; have := (h.ledger_lock p).1 hc; rw [hp] at this; simp [Phase.holdsLedger] at this
    have hpk : ProcOK S ad s.disk (s.procs p) := h.procs_ok p
    rw [hp] at hpk
    obtain ⟨hacc, hck, hcf⟩ := hpk
    simp only [Discipline.implemented, canTake, take, Bool.not_true, Bool.false_or, if_true,
      Bool.true_and]
    -- the three outcomes: scan failure, duplicate, fresh
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
        simp only [Mem.hits, Mem.union, fileMem, List.mem_append, Bool.or_eq_true,
          decide_eq_true_eq, not_or] at hhit
        obtain ⟨⟨hk1, hk2⟩, hf1, hf2⟩ := hhit
        refine ⟨hacc, ?_, hk2, hf2⟩
        rw [fresh_iff]
        constructor
        · intro hk
          rcases hck _ hk with h1 | ⟨_, h3⟩
          · exact hk1 h1
          · exact hk2 h3
        · intro hf
          rcases hcf _ hf with h1 | ⟨_, h3⟩
          · exact hf1 h1
          · exact hf2 h3
    · exact h
  · exact h


omit [DecidableEq K] [DecidableEq F] in
theorem replay_super_of {rs : List (K × F)} (i : K × F) (add : Bool) :
    ∀ r ∈ rs, r ∈ (if add then i :: rs else rs) := by
  intro r hr; split_ifs
  · exact List.mem_cons_of_mem _ hr
  · exact hr

theorem inv_writeRecord {s : Sys B K F} (h : Inv S ad s) (p : ℕ) (o : WriteOutcome) :
    Inv S ad (step Discipline.implemented S ad s (.writeRecord p o)) := by
  simp only [step]
  split
  · rename_i m b hp
    have hownC : s.chainLock = some p := (h.chain_lock p).2 (by rw [hp]; rfl)
    have hownL : s.ledgerLock = some p := (h.ledger_lock p).2 (by rw [hp]; rfl)
    have hpk : ProcOK S ad s.disk (s.procs p) := h.procs_ok p
    rw [hp] at hpk
    obtain ⟨hacc, hfr, hk, hf⟩ := hpk
    -- facts about the new disk, independent of the outcome
    set d' : Disk B K F :=
      { s.disk with
        replay := if o.effect.adds then S.ident b :: s.disk.replay else s.disk.replay
        replayTail := o.effect.tail s.disk.replayTail } with hd'
    have hsup : ∀ r ∈ s.disk.replay, r ∈ d'.replay := replay_super_of _ _
    have huniq : replayUnique d'.replay = true := by
      simp only [hd']
      split_ifs
      · exact replayUnique_cons.2 ⟨h.replay_unique, hk, hf⟩
      · exact h.replay_unique
    have hcir : ad = false →
        (∀ k ∈ S.kids d'.chain, k ∈ d'.replay.map Prod.fst) ∧
        (∀ f ∈ S.fps d'.chain, f ∈ d'.replay.map Prod.snd) := by
      intro had
      obtain ⟨h1, h2⟩ := h.chain_in_replay had
      refine ⟨fun k hk' => ?_, fun f hf' => ?_⟩
      · obtain ⟨r, hr, hrk⟩ := List.mem_map.1 (h1 k hk')
        exact List.mem_map.2 ⟨r, hsup r hr, hrk⟩
      · obtain ⟨r, hr, hrf⟩ := List.mem_map.1 (h2 f hf')
        exact List.mem_map.2 ⟨r, hsup r hr, hrf⟩
    have hothers : ∀ q, q ≠ p → ProcOK S ad d' (s.procs q) := by
      intro q hq
      exact procOK_replay_super S _ _ hsup (not_holds_of_owner h.ledger_lock hownL hq) (h.procs_ok q)
    cases o with
    | ok =>
      refine ⟨h.chain_valid, huniq, h.chain_admitted, hcir, ?_, ?_, ?_⟩
      · exact lock_after_set h.chain_lock (by simp [hownC, Phase.holdsChain]) (fun _ _ => Iff.rfl)
      · exact lock_after_set h.ledger_lock (by simp [release_ne_self, Phase.holdsLedger])
          (fun q hq => release_other hq)
      · refine procs_after_set S hothers ⟨hacc, hfr, fun _ => ?_⟩
        simp [hd', WriteOutcome.effect, Effect.adds]
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

theorem inv_writeBlock (hsound : S.AcceptSound) {s : Sys B K F} (h : Inv S ad s) (p : ℕ)
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
    set d' : Disk B K F :=
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
    have hcir : ad = false →
        (∀ k ∈ S.kids d'.chain, k ∈ d'.replay.map Prod.fst) ∧
        (∀ f ∈ S.fps d'.chain, f ∈ d'.replay.map Prod.snd) := by
      intro had
      obtain ⟨h1, h2⟩ := h.chain_in_replay had
      obtain ⟨hb1, hb2⟩ := hin had
      simp only [hd']
      split_ifs
      · refine ⟨fun k hk' => ?_, fun f hf' => ?_⟩
        · simp only [AcceptSpec.kids, List.map_cons, List.mem_cons] at hk'
          rcases hk' with rfl | hk'
          · exact hb1
          · exact h1 k hk'
        · simp only [AcceptSpec.fps, List.map_cons, List.mem_cons] at hf'
          rcases hf' with rfl | hf'
          · exact hb2
          · exact h2 f hf'
      · exact ⟨h1, h2⟩
    have hothers : ∀ q, q ≠ p → ProcOK S ad d' (s.procs q) := by
      intro q hq
      exact procOK_chain_change S _ _ (not_holds_of_owner h.chain_lock hownC hq) (h.procs_ok q)
    have hlockL : ∀ v : Phase B K F, v.holdsLedger = false →
        ∀ q, s.ledgerLock = some q ↔ (if q = p then v else s.procs q).holdsLedger = true :=
      fun v hv => lock_after_set h.ledger_lock (by simp [hnl, hv]) (fun _ _ => Iff.rfl)
    have hlockC : ∀ v : Phase B K F, v.holdsChain = false →
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

theorem inv_payBegin {s : Sys B K F} (h : Inv S ad s) (p : ℕ) (b : B) :
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
        simp only [Mem.hits, Mem.union, fileMem, List.mem_append, Bool.or_eq_true,
          decide_eq_true_eq, not_or] at hhit
        exact ⟨hhit.1.2, hhit.2.2⟩
    · exact h
    · exact h
  · exact h

theorem inv_payWrite {s : Sys B K F} (h : Inv S ad s) (p : ℕ) (o : WriteOutcome) :
    Inv S ad (step Discipline.implemented S ad s (.payWrite p o)) := by
  simp only [step]
  split
  · rename_i m b hp
    have hownL : s.ledgerLock = some p := (h.ledger_lock p).2 (by rw [hp]; rfl)
    have hnc : s.chainLock ≠ some p := by
      intro hc; have := (h.chain_lock p).1 hc; rw [hp] at this; simp [Phase.holdsChain] at this
    have hpk : ProcOK S ad s.disk (s.procs p) := h.procs_ok p
    rw [hp] at hpk
    obtain ⟨hk, hf⟩ := hpk
    set d' : Disk B K F :=
      { s.disk with
        replay := if o.effect.adds then S.ident b :: s.disk.replay else s.disk.replay
        replayTail := o.effect.tail s.disk.replayTail } with hd'
    have hsup : ∀ r ∈ s.disk.replay, r ∈ d'.replay := replay_super_of _ _
    have huniq : replayUnique d'.replay = true := by
      simp only [hd']
      split_ifs
      · exact replayUnique_cons.2 ⟨h.replay_unique, hk, hf⟩
      · exact h.replay_unique
    have hcir : ad = false →
        (∀ k ∈ S.kids d'.chain, k ∈ d'.replay.map Prod.fst) ∧
        (∀ f ∈ S.fps d'.chain, f ∈ d'.replay.map Prod.snd) := by
      intro had
      obtain ⟨h1, h2⟩ := h.chain_in_replay had
      refine ⟨fun k hk' => ?_, fun f hf' => ?_⟩
      · obtain ⟨r, hr, hrk⟩ := List.mem_map.1 (h1 k hk')
        exact List.mem_map.2 ⟨r, hsup r hr, hrk⟩
      · obtain ⟨r, hr, hrf⟩ := List.mem_map.1 (h2 f hf')
        exact List.mem_map.2 ⟨r, hsup r hr, hrf⟩
    have hothers : ∀ q, q ≠ p → ProcOK S ad d' (s.procs q) := by
      intro q hq
      exact procOK_replay_super S _ _ hsup (not_holds_of_owner h.ledger_lock hownL hq) (h.procs_ok q)
    have hlockC : ∀ v : Phase B K F, v.holdsChain = false →
        ∀ q, s.chainLock = some q ↔ (if q = p then v else s.procs q).holdsChain = true :=
      fun v hv => lock_after_set h.chain_lock (by simp [hnc, hv]) (fun _ _ => Iff.rfl)
    have hlockL : ∀ v : Phase B K F, v.holdsLedger = false →
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

theorem inv_crash {s : Sys B K F} (h : Inv S ad s) (p : ℕ) :
    Inv S ad (step Discipline.implemented S ad s (.crash p)) := by
  simp only [step]
  refine ⟨h.chain_valid, h.replay_unique, h.chain_admitted, h.chain_in_replay, ?_, ?_, ?_⟩
  · exact lock_after_set h.chain_lock (by simp [release_ne_self, Phase.holdsChain])
      (fun q hq => release_other hq)
  · exact lock_after_set h.ledger_lock (by simp [release_ne_self, Phase.holdsLedger])
      (fun q hq => release_other hq)
  · exact procs_after_set S (fun q _ => h.procs_ok q) trivial

theorem inv_restart {s : Sys B K F} (h : Inv S ad s) (p : ℕ) :
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

theorem inv_deleteReplay {s : Sys B K F} (h : Inv S ad s) :
    Inv S ad (step Discipline.implemented S ad s .deleteReplay) := by
  simp only [step]
  split_ifs with had
  · subst had
    refine ⟨h.chain_valid, rfl, h.chain_admitted, fun h' => absurd h' (by decide),
      h.chain_lock, h.ledger_lock, fun q => procOK_replay_delete S (h.procs_ok q)⟩
  · exact h

/-- **Preservation.** Every event of the implemented discipline preserves the invariant, provided
acceptance entails the chain validator's per-block checks, and either reconstruction from the
chain is lossless or the side replay file is never lost. -/
theorem inv_step (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    {s : Sys B K F} (h : Inv S ad s) (e : Event B) :
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

theorem inv_init : Inv S ad (Sys.init : Sys B K F) := by
  refine ⟨rfl, rfl, by simp [Sys.init], fun _ => ⟨by simp [Sys.init, AcceptSpec.kids],
    by simp [Sys.init, AcceptSpec.fps]⟩, ?_, ?_, fun _ => trivial⟩
  · intro p; simp [Sys.init, Phase.holdsChain]
  · intro p; simp [Sys.init, Phase.holdsLedger]

/-- The invariant holds after every finite trace from the empty host. -/
theorem inv_run (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) : Inv S ad (run Discipline.implemented S ad es) := by
  unfold run
  suffices ∀ (es : List (Event B)) (s : Sys B K F), Inv S ad s →
      Inv S ad (es.foldl (step Discipline.implemented S ad) s) from this es _ (inv_init S)
  intro es
  induction es with
  | nil => intro s hs; exact hs
  | cons e es ih => intro s hs; exact ih _ (inv_step S hsound hrecon hs e)

end Preservation


/-! ## Consequences for arbitrary finite traces -/

section Consequences
variable {B K F : Type} [DecidableEq K] [DecidableEq F] (S : AcceptSpec B K F) {ad : Bool}

theorem chainValid_nodup : ∀ c : List B, S.chainValid c = true → (S.kids c).Nodup ∧ (S.fps c).Nodup
  | [], _ => by simp [AcceptSpec.kids, AcceptSpec.fps]
  | b :: c, h => by
    obtain ⟨hc, _, _, hfr⟩ := (chainValid_cons S).1 h
    obtain ⟨hk, hf⟩ := (fresh_iff S).1 hfr
    obtain ⟨ihk, ihf⟩ := chainValid_nodup c hc
    exact ⟨List.nodup_cons.2 ⟨hk, ihk⟩, List.nodup_cons.2 ⟨hf, ihf⟩⟩

/-- **Single use in the authoritative chain.**  After any finite trace — any interleaving of any
number of same-host processes, any write failures, crashes at any modelled point, restarts and
loss of the side replay file — no two stored blocks share a `one_time_key_id` and no two share a
`public_key_fingerprint`. -/
theorem chain_single_use (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) :
    (S.kids (run Discipline.implemented S ad es).disk.chain).Nodup ∧
      (S.fps (run Discipline.implemented S ad es).disk.chain).Nodup :=
  chainValid_nodup S _ (inv_run S hsound hrecon es).chain_valid

theorem run_append (D : Discipline) (es es' : List (Event B)) :
    run D S ad (es ++ es') = es'.foldl (step D S ad) (run D S ad es) := by
  simp [run, List.foldl_append]

/-- The durable chain is append-only under every discipline (no event removes a block). -/
theorem chain_suffix_step (D : Discipline) (s : Sys B K F) (e : Event B) :
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

theorem chain_suffix_foldl (D : Discipline) (es : List (Event B)) (s : Sys B K F) :
    s.disk.chain <:+ (es.foldl (step D S ad) s).disk.chain := by
  induction es generalizing s with
  | nil => exact List.suffix_refl _
  | cons e es ih => exact (chain_suffix_step S D s e).trans (ih _)

/-- **Consumption is fresh.**  Whenever a process is about to durably record a consumption
(`_append` under the ledger lock), neither identity is consumed in the authoritative durable
state — accepted chain *and* replay records — at that instant. -/
theorem consume_fresh (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es : List (Event B)) {p : ℕ} {m : Mem K F} {b : B}
    (hp : (run Discipline.implemented S ad es).procs p = .checked m b) :
    (S.ident b).1 ∉ S.consumedKeys (run Discipline.implemented S ad es).disk ∧
      (S.ident b).2 ∉ S.consumedLeaves (run Discipline.implemented S ad es).disk := by
  have h := (inv_run S hsound hrecon es).procs_ok p
  rw [hp] at h
  obtain ⟨_, hfr, hk, hf⟩ := h
  obtain ⟨hk', hf'⟩ := (fresh_iff S).1 hfr
  simp only [AcceptSpec.consumedKeys, AcceptSpec.consumedLeaves, List.mem_append, not_or]
  exact ⟨⟨hk', hk⟩, hf', hf⟩

/-- **Consumption records both identities.**  A record append whose bytes land (successful,
or followed by an I/O error) makes the key id and the fingerprint consumed. -/
theorem writeRecord_consumes_both (D : Discipline) {s : Sys B K F} {p : ℕ} {m : Mem K F} {b : B}
    (hp : s.procs p = .checked m b) {o : WriteOutcome} (ho : o.effect.adds = true) :
    (S.ident b).1 ∈ S.consumedKeys (step D S ad s (.writeRecord p o)).disk ∧
      (S.ident b).2 ∈ S.consumedLeaves (step D S ad s (.writeRecord p o)).disk := by
  simp only [step, hp]
  cases o <;> simp_all [AcceptSpec.consumedKeys, AcceptSpec.consumedLeaves, WriteOutcome.effect]

/-- A block append whose bytes land puts the block itself in the authoritative chain. -/
theorem writeBlock_appends (D : Discipline) {s : Sys B K F} {p : ℕ} {m : Mem K F} {b : B}
    (hp : s.procs p = .recorded m b) {o : WriteOutcome} (ho : o.effect.adds = true) :
    (step D S ad s (.writeBlock p o)).disk.chain = b :: s.disk.chain := by
  simp only [step, hp]
  cases o <;> simp_all [WriteOutcome.effect]

/-- **At most one acceptance per identity, over any interleaving.**  If some process durably
appended `b₁` (a reported acceptance, or an append whose bytes landed before an I/O error), no
process can later even reach the post-consumption phase for a block sharing either identity. -/
theorem at_most_one_acceptance (hsound : S.AcceptSound) (hrecon : S.LosslessRecon ∨ ad = false)
    (es₁ es₂ : List (Event B)) {p q : ℕ} {o : WriteOutcome} {m₁ m₂ : Mem K F} {b₁ b₂ : B}
    (hadds : o.effect.adds = true)
    (h₁ : (run Discipline.implemented S ad es₁).procs p = .recorded m₁ b₁)
    (h₂ : (run Discipline.implemented S ad (es₁ ++ .writeBlock p o :: es₂)).procs q =
      .recorded m₂ b₂) :
    (S.ident b₁).1 ≠ (S.ident b₂).1 ∧ (S.ident b₁).2 ≠ (S.ident b₂).2 := by
  set s₁ := run Discipline.implemented S ad es₁
  set s₂ := run Discipline.implemented S ad (es₁ ++ .writeBlock p o :: es₂)
  have hmem₁ : b₁ ∈ (step Discipline.implemented S ad s₁ (.writeBlock p o)).disk.chain := by
    simp only [step, h₁]
    cases o <;> simp_all [WriteOutcome.effect]
  have hsuf : (step Discipline.implemented S ad s₁ (.writeBlock p o)).disk.chain <:+ s₂.disk.chain := by
    have : s₂ = es₂.foldl (step Discipline.implemented S ad)
        (step Discipline.implemented S ad s₁ (.writeBlock p o)) := by
      simp only [s₂, s₁, run_append, List.foldl_cons]
    rw [this]; exact chain_suffix_foldl S _ es₂ _
  have hb₁ : b₁ ∈ s₂.disk.chain := hsuf.subset hmem₁
  have h := (inv_run S hsound hrecon (es₁ ++ .writeBlock p o :: es₂)).procs_ok q
  rw [h₂] at h
  obtain ⟨_, hfr, _⟩ := h
  obtain ⟨hk, hf⟩ := (fresh_iff S).1 hfr
  refine ⟨fun heq => hk ?_, fun heq => hf ?_⟩
  · exact (mem_kids S).2 ⟨b₁, hb₁, heq⟩
  · exact (mem_fps S).2 ⟨b₁, hb₁, heq⟩

/-- Rejecting and control events never touch durable state. -/
theorem control_events_keep_disk (D : Discipline) (s : Sys B K F) (e : Event B)
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
consumed identity from the authoritative durable state. -/
theorem consumed_monotone (D : Discipline) (s : Sys B K F) (e : Event B) (he : e ≠ .deleteReplay) :
    (∀ k ∈ S.consumedKeys s.disk, k ∈ S.consumedKeys (step D S ad s e).disk) ∧
      ∀ f ∈ S.consumedLeaves s.disk, f ∈ S.consumedLeaves (step D S ad s e).disk := by
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
  constructor
  · intro k hk
    simp only [AcceptSpec.consumedKeys, List.mem_append] at hk ⊢
    rcases hk with hk | hk
    · left; obtain ⟨x, hx, rfl⟩ := (mem_kids S).1 hk; exact (mem_kids S).2 ⟨x, hchain.subset hx, rfl⟩
    · right; obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hk; exact List.mem_map.2 ⟨r, hrep r hr, rfl⟩
  · intro f hf
    simp only [AcceptSpec.consumedLeaves, List.mem_append] at hf ⊢
    rcases hf with hf | hf
    · left; obtain ⟨x, hx, rfl⟩ := (mem_fps S).1 hf; exact (mem_fps S).2 ⟨x, hchain.subset hx, rfl⟩
    · right; obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hf; exact List.mem_map.2 ⟨r, hrep r hr, rfl⟩

/-- **Restart reconstruction.**  A restarted process holds every authoritative consumed
identity in memory when reconstruction is lossless. -/
theorem restart_recovers (hl : S.LosslessRecon) {s : Sys B K F} (h : Inv S ad s) {p : ℕ}
    (hp : s.procs p = .dead) (hload : S.replayLoads s.disk = true ∧ S.chainLoads s.disk = true) :
    ∃ m, (step Discipline.implemented S ad s (.restart p)).procs p = .idle m ∧
      (∀ k ∈ S.consumedKeys s.disk, k ∈ m.keys) ∧ ∀ f ∈ S.consumedLeaves s.disk, f ∈ m.leaves := by
  refine ⟨(fileMem s.disk.replay).union (S.reconAll s.disk.chain), ?_, ?_, ?_⟩
  · simp [step, hp, hload, Sys.set]
  · intro k hk
    simp only [AcceptSpec.consumedKeys, List.mem_append] at hk
    simp only [Mem.union, fileMem, List.mem_append]
    rcases hk with hk | hk
    · right
      obtain ⟨x, hx, rfl⟩ := (mem_kids S).1 hk
      obtain ⟨c, hc⟩ := h.chain_admitted x hx
      exact (mem_reconAll S hx).1 _ (hl c x hc).1
    · left; exact hk
  · intro f hf
    simp only [AcceptSpec.consumedLeaves, List.mem_append] at hf
    simp only [Mem.union, fileMem, List.mem_append]
    rcases hf with hf | hf
    · right
      obtain ⟨x, hx, rfl⟩ := (mem_fps S).1 hf
      obtain ⟨c, hc⟩ := h.chain_admitted x hx
      exact (mem_reconAll S hx).2 _ (hl c x hc).2
    · left; exact hf

/-- **Fail-closed under torn chain writes.**  A garbage chain tail is permanent and no
acceptance can start: every later `begin` is a no-op. -/
theorem chain_garbage_absorbing (D : Discipline) (s : Sys B K F) (e : Event B)
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

theorem begin_blocked_by_garbage (D : Discipline) (s : Sys B K F) (p : ℕ) (b : B)
    (h : s.disk.chainTail = .garbage) : step D S ad s (.begin p b) = s := by
  simp only [step]
  split
  · have : S.chainLoads s.disk = false := by simp [AcceptSpec.chainLoads, h]
    simp [this]
  · rfl

/-- **Fail-closed under a torn or unreadable replay file.**  With the implemented discipline a
process that has passed the chain checks never reaches the consumption phase while the replay
file does not load. -/
theorem ledger_blocked_by_bad_replay (s : Sys B K F) {p : ℕ} {m : Mem K F} {b : B}
    (hp : s.procs p = .loaded m b) (h : S.replayLoads s.disk = false) (m' : Mem K F) (b' : B) :
    (step Discipline.implemented S ad s (.ledger p)).procs p ≠ .checked m' b' := by
  simp only [step, hp, Discipline.implemented, canTake, take, Bool.not_true, Bool.false_or,
    if_true, Bool.true_and, h, Bool.not_false]
  split_ifs <;> simp [Sys.set, hp]

end Consequences


/-! ## Liveness: acceptance never makes the node's own state unloadable -/

section Liveness
variable {B K F : Type} [DecidableEq K] [DecidableEq F] (S : AcceptSpec B K F) {ad : Bool}

/-- Load-relevant knowledge of a process about the block or record it is about to write. -/
def LoadOK : Phase B K F → Prop
  | .checked _ b => S.recordOK (S.ident b) = true ∧ S.reconFail b = false
  | .recorded _ b => S.reconFail b = false
  | .payChecked _ b => S.recordOK (S.ident b) = true
  | _ => True

structure LoadInv (s : Sys B K F) : Prop where
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

omit [DecidableEq K] [DecidableEq F] in
theorem loadOK_after_set {procs : ℕ → Phase B K F} {p : ℕ} {v : Phase B K F}
    (hall : ∀ q, LoadOK S (procs q)) (hv : LoadOK S v) : ∀ q, LoadOK S (if q = p then v else procs q) := by
  intro q; by_cases hq : q = p
  · simp only [hq, if_true]; exact hv
  · simp only [hq, if_false]; exact hall q

theorem clean_tail {o : WriteOutcome} (ho : o.effect = .line ∨ o.effect = .nothing) :
    o.effect.tail .clean = .clean := by
  rcases ho with h | h <;> rw [h] <;> rfl

omit [DecidableEq K] [DecidableEq F] in
theorem loadInv_set {s : Sys B K F} {p : ℕ} {v : Phase B K F} {c l : Option ℕ}
    (h : LoadInv S s) (hv : LoadOK S v) :
    LoadInv S { s with procs := s.set p v, chainLock := c, ledgerLock := l } :=
  ⟨h.chain_tail, h.replay_tail, h.records_ok, h.chain_total, loadOK_after_set S h.procs hv⟩

theorem loadInv_step (hadm : S.RecordAdmissible) (htot : S.ReconTotal) {s : Sys B K F}
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
      have hrecs : ∀ r ∈ (if o.effect.adds then S.ident b :: s.disk.replay else s.disk.replay),
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
      have hrecs : ∀ r ∈ (if o.effect.adds then S.ident b :: s.disk.replay else s.disk.replay),
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
  have key : ∀ (es : List (Event B)) (s : Sys B K F), (∀ e ∈ es, CleanIO e) →
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
  have hinit : LoadInv S (Sys.init : Sys B K F) :=
    ⟨rfl, rfl, by simp [Sys.init], by simp [Sys.init], fun _ => trivial⟩
  obtain ⟨hi, hl⟩ := key es _ hclean (inv_init S) hinit
  simp only [run]
  constructor
  · simp only [AcceptSpec.chainLoads, Bool.and_eq_true, List.all_eq_true, Bool.not_eq_true']
    refine ⟨⟨by simp [hl.chain_tail], hi.chain_valid⟩, fun x hx => hl.chain_total x hx⟩
  · simp only [AcceptSpec.replayLoads, Bool.and_eq_true, List.all_eq_true]
    exact ⟨by simp [hl.replay_tail], fun r hr => hl.records_ok r hr⟩

end Liveness

/-! ## Concrete instance: `ots_ledger.py` + `server.py` at the frozen revision -/

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

/-- `isinstance(kid, str) and kid != ""`, the identity typing of `LEDGER_VERSION = 1` records. -/
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
  /-- `public_key.merkle_root`: the committed one-time key material -/
  material : ℕ
  /-- `verify_block_integrity ∧ verify_ots_block` (signature, transcript binding, header context) -/
  authentic : Bool
  deriving DecidableEq, Repr

def cTip : List CBlock → ℕ
  | [] => 0
  | t :: _ => t.hash

/-- `_verify_pow_and_linkage` / `verify_ots_chain` linkage on a newest-first chain. -/
def cLink (c : List CBlock) (b : CBlock) : Bool := b.index == c.length + 1 && b.parent == cTip c

/-- `index_signature` at `b310ea5`: `str(...)` of each field, then skip the *string* `"None"`. -/
def reconLegacy (b : CBlock) : Mem ℕ ℕ :=
  ⟨if b.kid.render = 1 then [] else [b.kid.render], [b.fp]⟩

/-- `index_signature` after this change: skip only a missing (`None`) field. -/
def reconFixed (b : CBlock) : Mem ℕ ℕ :=
  ⟨match b.kid with
    | .null => []
    | k => [k.render], [b.fp]⟩

/-- The Python acceptance protocol.  `fixed = false` is the frozen revision `b310ea5`;
`fixed = true` adds this change's two ledger corrections. -/
def pySpec (fixed : Bool) : AcceptSpec CBlock ℕ ℕ where
  ident b := (b.kid.render, b.fp)
  blockOK b := b.authentic && b.kid.hashable
  linkOK := cLink
  acceptOK c b := b.authentic && cLink c b && (!fixed || b.kid.nonEmptyString)
  payOK b := b.authentic && (!fixed || b.kid.nonEmptyString)
  recon := if fixed then reconFixed else reconLegacy
  reconFail b := b.kid == .null
  recordOK i := i.1 != 0

theorem nonEmptyString_str {k : RawKid} (h : k.nonEmptyString = true) : ∃ c, k = .str c ∧ c ≠ 0 := by
  cases k with
  | str c => exact ⟨c, rfl, by simpa [RawKid.nonEmptyString] using h⟩
  | _ => simp [RawKid.nonEmptyString] at h

theorem pySpec_acceptSound : (pySpec true).AcceptSound := by
  intro c b h
  simp only [pySpec, Bool.not_true, Bool.false_or, Bool.and_eq_true] at h
  obtain ⟨⟨ha, hl⟩, hs⟩ := h
  obtain ⟨k, hk, _⟩ := nonEmptyString_str hs
  refine ⟨hl, ?_⟩
  simp [pySpec, ha, hk, RawKid.hashable]

theorem pySpec_losslessRecon : (pySpec true).LosslessRecon := by
  intro c b h
  simp only [pySpec, Bool.not_true, Bool.false_or, Bool.and_eq_true] at h
  obtain ⟨k, hk, _⟩ := nonEmptyString_str h.2
  simp [pySpec, reconFixed, hk, RawKid.render]

theorem pySpec_recordAdmissible : (pySpec true).RecordAdmissible := by
  constructor
  · intro c b h
    simp only [pySpec, Bool.not_true, Bool.false_or, Bool.and_eq_true] at h
    obtain ⟨k, hk, hk0⟩ := nonEmptyString_str h.2
    simp [pySpec, hk, RawKid.render, hk0]
  · intro b h
    simp only [pySpec, Bool.not_true, Bool.false_or, Bool.and_eq_true] at h
    obtain ⟨k, hk, hk0⟩ := nonEmptyString_str h.2
    simp [pySpec, hk, RawKid.render, hk0]

theorem pySpec_reconTotal : (pySpec true).ReconTotal := by
  intro c b h
  simp only [pySpec, Bool.not_true, Bool.false_or, Bool.and_eq_true] at h
  obtain ⟨k, hk, _⟩ := nonEmptyString_str h.2
  simp [pySpec, hk]

/-- **Headline theorem (implementation model).**  For the corrected WaveLock-OTS acceptance path,
every finite same-host trace — any interleaving of processes, write failures, crashes at any
modelled point, restarts, and (with `ad = true`) loss of the side replay file — leaves a durable
chain in which no `one_time_key_id` and no `public_key_fingerprint` occurs twice. -/
theorem wavelock_ots_chain_single_use (ad : Bool) (es : List (Event CBlock)) :
    ((pySpec true).kids (run Discipline.implemented (pySpec true) ad es).disk.chain).Nodup ∧
      ((pySpec true).fps (run Discipline.implemented (pySpec true) ad es).disk.chain).Nodup :=
  chain_single_use _ pySpec_acceptSound (Or.inl pySpec_losslessRecon) es

/-- For the corrected acceptance path no sequence of blocks or payload signatures can make the
node's own files unloadable; only a torn write (or hostile storage) can. -/
theorem wavelock_ots_no_self_dos (ad : Bool) (es : List (Event CBlock))
    (hclean : ∀ e ∈ es, CleanIO e) :
    (pySpec true).chainLoads (run Discipline.implemented (pySpec true) ad es).disk = true ∧
      (pySpec true).replayLoads (run Discipline.implemented (pySpec true) ad es).disk = true :=
  loadable_run _ pySpec_acceptSound (Or.inl pySpec_losslessRecon) pySpec_recordAdmissible
    pySpec_reconTotal es hclean


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
def acceptedBlock? {B K F : Type} (s : Sys B K F) : Event B → Option B
  | .writeBlock p .ok => match s.procs p with
    | .recorded _ b => some b
    | _ => none
  | _ => none

/-- Run a trace and collect every reported acceptance, in order. -/
def runLog {B K F : Type} [DecidableEq K] [DecidableEq F] (D : Discipline) (S : AcceptSpec B K F)
    (ad : Bool) (s : Sys B K F) (es : List (Event B)) : Sys B K F × List B :=
  es.foldl (fun (acc : Sys B K F × List B) e =>
    (step D S ad acc.1 e, acc.2 ++ (acceptedBlock? acc.1 e).toList)) (s, [])

end OTS
end WaveLockProofs
