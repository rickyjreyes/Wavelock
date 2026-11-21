import WaveLockProofs.RCAAgentState

/-!
# RCA reference safety kernel: executable transition semantics (issue #42)

**Status: `PROPOSED_MODEL`** of `wavelock/rca_reference/kernel.py :: GuardedAgent` (a proposed
reference mechanism; see `RCAAgentState`).

## Processes and the critical section

`GuardedAgent._transact` takes the memory lock (`exclusive_lock`, `fcntl.flock` or SQLite), reloads
the memory (`_load`), and calls `_submit`, which checks admissibility on the reloaded view, appends
the record (`_write_line`: `write`, `fsync`) and, for an action, invokes the effect, before the lock
is released.  Event `begin q` is lock + reload; event `submit q r o` is guard + append with write
outcome `o` (and release, except after a durable `execute` intent); event `effect q` runs the
effect of the committed intent, acknowledges it and releases the lock.  A crash between the two
leaves a durable intent whose effect never ran (at-most-once).  Several processes (or
`GuardedAgent` instances) share one memory directory.

| event | Python | observable | durable | reversible |
|---|---|---|---|---|
| `provision` | `GuardedAgent.provision` (exclusive atomic create) | yes | yes | no |
| `begin` | lock + `_load` (fails closed: `MemoryCorrupt`) | no | no | yes |
| `submit` (refused) | `_submit` returns `False` | yes | no | yes |
| `submit` (`ok`) | durable append; for `execute` the intent, lock still held | yes | yes | no |
| `effect` | `effect(action)` after the durable intent, then release | yes | — | no |
| `submit` (`fail`) | `OSError` from `_write_line`; effect never runs | yes | maybe | no |
| `submit` (`crash`) | process death during the write | no | maybe | no |
| `crash`, `restart` | process death / new `GuardedAgent(...)` | no | no | — |
| `issue` | operator computes `operator_credential` for the current prefix | operator | no | — |
| `anchorNow` | an external party records `(index, digest)` of the head | operator | external | — |

Out-of-boundary events are part of the model so that the hypotheses of every theorem are
explicit, and each has a counterexample (`RCACounterexamples`): `directEffect` (code that invokes
an effect without the kernel), `rollback` (the memory file is restored to a prefix or deleted, or
storage loses acknowledged writes), `tamper` (the memory is rewritten).

## Disciplines

`Discipline.reference` is the reference code.  Every other setting removes one architectural
element and exists only to show, by counterexample, that the element is necessary.  Removing an
element changes only the writer; the loader and the specification stay those of the reference.

## Named abstractions

* The hash chain and canonical encoding are abstracted: a log is its record list, and a credential
  or anchor names a prefix exactly (collision resistance of SHA-256 and HMAC unforgeability are
  assumed, and the operator key is assumed unavailable to the agent).
* `fsync` is honest; storage that loses acknowledged writes is the out-of-boundary `rollback`.
* A blocked lock acquisition leaves the state unchanged (the scheduler retries).
* The effect is atomic with the successful append (it runs under the lock, after `fsync`).  Its
  real-world consequences are outside the model.
-/

namespace WaveLockProofs
namespace RCA

/-! ## Durable memory file -/

/-- Bytes that reached the memory file during one append. -/
inductive Effect where
  /-- nothing reached the file -/
  | nothing
  /-- the complete, newline-terminated record -/
  | line
  /-- a partial or unterminated record: every later load fails closed -/
  | torn
  deriving DecidableEq, Repr

/-- Outcome of one `_write_line`.  `fail` raises `OSError` into the caller; `crash` kills the
process. -/
inductive WriteOutcome where
  | ok
  | fail (e : Effect)
  | crash (e : Effect)
  deriving DecidableEq, Repr

structure Disk where
  log : List Rec
  torn : Bool
  deriving DecidableEq, Repr

def Disk.write (d : Disk) (r : Rec) : Effect → Disk
  | .nothing => d
  | .line => { d with log := d.log ++ [r] }
  | .torn => { d with torn := true }

/-- Credentials the operator has issued, each for exactly one prefix. -/
abbrev Issued := List (ℕ × List Rec)

def credOK (issued : Issued) : CredOK := fun c pre => decide ((c, pre) ∈ issued)

/-- Credential `c` was issued for some prefix (the check without binding). -/
def credAny (issued : Issued) (c : ℕ) : Bool := issued.any fun x => x.1 == c

/-- What `_load` returns: nothing if the tail is torn or the log does not extend the anchor. -/
def Disk.view (issued : Issued) (anchor : List Rec) (d : Disk) : Option View :=
  if d.torn = false ∧ anchor.isPrefixOf d.log = true then load (credOK issued) d.log else none

/-! ## Disciplines -/

/-- Architectural elements of the reference kernel. -/
structure Discipline where
  /-- the critical section holds the memory lock -/
  lock : Bool := true
  /-- the memory is reloaded at the start of every critical section -/
  reload : Bool := true
  /-- records are appended durably (otherwise kept only in process memory) -/
  persist : Bool := true
  /-- the guard consults the drift latch before an action -/
  driftGate : Bool := true
  /-- replayed or out-of-order observations are refused -/
  seqCheck : Bool := true
  /-- policy updates may only tighten -/
  monotonePolicy : Bool := true
  /-- a reset credential must be bound to the exact prefix -/
  bindCred : Bool := true
  deriving DecidableEq, Repr

def Discipline.reference : Discipline := {}

@[simp] theorem Discipline.reference_lock : Discipline.reference.lock = true := rfl
@[simp] theorem Discipline.reference_reload : Discipline.reference.reload = true := rfl
@[simp] theorem Discipline.reference_persist : Discipline.reference.persist = true := rfl
@[simp] theorem Discipline.reference_driftGate : Discipline.reference.driftGate = true := rfl
@[simp] theorem Discipline.reference_seqCheck : Discipline.reference.seqCheck = true := rfl
@[simp] theorem Discipline.reference_monotonePolicy :
    Discipline.reference.monotonePolicy = true := rfl
@[simp] theorem Discipline.reference_bindCred : Discipline.reference.bindCred = true := rfl

/-- The writer's admissibility check under a discipline. -/
def guard (D : Discipline) (issued : Issued) (pre : List Rec) (v : View) : Rec → Bool
  | .genesis .. => false
  | .observe s xs => (!D.seqCheck || decide (v.lastSeq < s)) && xs.length == v.baseline.length
  | .policy p =>
    p.valid && (if D.monotonePolicy then p.successor v.policy else p.version == v.policy.version + 1)
  | .execute k => (!D.driftGate || !v.locked) && !decide (k ∈ v.policy.prohibited)
  | .reset c => v.locked && decide (v.lastDev < v.policy.threshold) &&
      (if D.bindCred then credOK issued c pre else credAny issued c)

theorem guard_reference (issued : Issued) (pre : List Rec) (v : View) (r : Rec) :
    guard .reference issued pre v r = admissible (credOK issued) pre v r := by
  cases r <;> simp [guard, admissible, View.authorizes]

/-! ## System state and events -/

/-- Control state of one process.  `holding pre v`: inside the critical section with the view `v`
of the prefix `pre`.  `committed pre v k`: still inside the critical section, the `execute k`
intent decided on `v` is durable, and the effect has not run yet. -/
inductive Phase where
  | idle
  | dead
  | holding (pre : List Rec) (v : View)
  | committed (pre : List Rec) (v : View) (k : ℕ)
  deriving DecidableEq, Repr

/-- Ghost record of one effect: action kind, the prefix the guard decided on, and the durable log
and issued credentials at the moment the effect ran. -/
structure Done where
  kind : ℕ
  decided : List Rec
  log : List Rec
  issued : Issued
  deriving DecidableEq, Repr

structure Sys where
  disk : Disk
  procs : ℕ → Phase
  /-- the last view each process loaded or produced (decisive only when `reload = false`) -/
  cache : ℕ → Option (List Rec × View)
  lock : Option ℕ
  issued : Issued
  anchor : List Rec
  /-- ghost: every effect that happened -/
  effects : List Done
  /-- ghost: records acknowledged to their callers as durable, in order -/
  acked : List Rec

inductive Event where
  | provision (agent : ℕ) (baseline : List ℤ) (p : Policy)
  | begin (q : ℕ)
  | submit (q : ℕ) (r : Rec) (o : WriteOutcome)
  /-- the effect of a committed `execute` intent runs (`effect(action)` in `_submit`) -/
  | effect (q : ℕ)
  | crash (q : ℕ)
  | restart (q : ℕ)
  | issue (c : ℕ)
  | anchorNow
  /-- out of boundary: an effect invoked without the kernel -/
  | directEffect (k : ℕ)
  /-- out of boundary: the memory is restored to its first `n` records (deleted when `n = 0`) -/
  | rollback (n : ℕ)
  /-- out of boundary: the memory is rewritten -/
  | tamper (L : List Rec)
  deriving DecidableEq, Repr

/-- Events inside the trust boundary. -/
def Event.inBoundary : Event → Bool
  | .directEffect _ | .rollback _ | .tamper _ => false
  | _ => true

def release (q : ℕ) (l : Option ℕ) : Option ℕ := if l = some q then none else l

namespace Sys

def set (S : Sys) (q : ℕ) (ph : Phase) : ℕ → Phase := fun x => if x = q then ph else S.procs x

def setCache (S : Sys) (q : ℕ) (c : Option (List Rec × View)) : ℕ → Option (List Rec × View) :=
  fun x => if x = q then c else S.cache x

/-- A fresh load of the durable memory. -/
def fresh (S : Sys) : Option (List Rec × View) :=
  (S.disk.view S.issued S.anchor).map fun v => (S.disk.log, v)

def init : Sys :=
  ⟨⟨[], false⟩, fun _ => .idle, fun _ => none, none, [], [], [], []⟩

end Sys

/-- One step.  Events whose process is not in the matching phase, or whose lock is busy, leave the
state unchanged. -/
def step (D : Discipline) (S : Sys) : Event → Sys
  | .provision a b p =>
    if S.disk.log = [] ∧ S.disk.torn = false ∧ (genesisView (.genesis a b p)).isSome = true then
      { S with disk := ⟨[.genesis a b p], false⟩ }
    else S
  | .begin q =>
    match S.procs q with
    | .idle =>
      if D.lock = false ∨ S.lock = none then
        match (if D.reload then S.fresh else (S.cache q).orElse fun _ => S.fresh) with
        | some (pre, v) =>
          { S with procs := S.set q (.holding pre v), lock := if D.lock then some q else S.lock,
                   cache := S.setCache q (some (pre, v)) }
        | none => S
      else S
    | _ => S
  | .submit q r o =>
    match S.procs q with
    | .holding pre v =>
      if guard D S.issued pre v r then
        match o with
        | .ok =>
          match r with
          | .execute k =>
            { S with disk := if D.persist then S.disk.write r .line else S.disk,
                     procs := S.set q (.committed pre v k),
                     cache := S.setCache q (some (pre ++ [r], next v r)) }
          | _ =>
            { S with disk := if D.persist then S.disk.write r .line else S.disk,
                     procs := S.set q .idle, lock := release q S.lock,
                     cache := S.setCache q (some (pre ++ [r], next v r)),
                     acked := S.acked ++ [r] }
        | .fail e =>
          { S with disk := if D.persist then S.disk.write r e else S.disk,
                   procs := S.set q .idle, lock := release q S.lock }
        | .crash e =>
          { S with disk := if D.persist then S.disk.write r e else S.disk,
                   procs := S.set q .dead, lock := release q S.lock }
      else { S with procs := S.set q .idle, lock := release q S.lock }
    | _ => S
  | .effect q =>
    match S.procs q with
    | .committed pre _ k =>
      { S with procs := S.set q .idle, lock := release q S.lock,
               acked := S.acked ++ [.execute k],
               effects := S.effects ++ [⟨k, pre, S.disk.log, S.issued⟩] }
    | _ => S
  | .crash q => { S with procs := S.set q .dead, lock := release q S.lock }
  | .restart q =>
    match S.procs q with
    | .dead => { S with procs := S.set q .idle, cache := S.setCache q S.fresh }
    | _ => S
  | .issue c =>
    if credAny S.issued c then S else { S with issued := S.issued ++ [(c, S.disk.log)] }
  | .anchorNow => { S with anchor := S.disk.log }
  | .directEffect k => { S with effects := S.effects ++ [⟨k, S.disk.log, S.disk.log, S.issued⟩] }
  | .rollback n => { S with disk := ⟨S.disk.log.take n, false⟩ }
  | .tamper L => { S with disk := ⟨L, false⟩ }

/-- Run a finite trace from the empty host. -/
def run (D : Discipline) (es : List Event) : Sys := es.foldl (step D) Sys.init

/-- The verdict a `submit` receives: `some (some b)` is the guard's verdict, `some none` means the
process was not inside the critical section (dead, or its load failed closed); other events give
`none`. -/
def decision (D : Discipline) (S : Sys) : Event → Option (Option Bool)
  | .submit q r _ =>
    match S.procs q with
    | .holding pre v => some (some (guard D S.issued pre v r))
    | _ => some none
  | _ => none

/-- Run a trace and also report the verdict of every request, in order. -/
def runOut (D : Discipline) (es : List Event) : Sys × List (Option Bool) :=
  es.foldl (fun acc e => (step D acc.1 e,
    match decision D acc.1 e with
    | some x => acc.2 ++ [x]
    | none => acc.2)) (Sys.init, [])

end RCA
end WaveLockProofs
