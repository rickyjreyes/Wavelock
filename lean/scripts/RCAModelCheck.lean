import WaveLockProofs.RCACounterexamples

/-!
# Bounded exhaustive search over the RCA model (issue #43, Phase 10)

Breadth-first search over `RCA.step` itself (not a re-implementation), from the provisioned state
`step D Sys.init prov`, deduplicating states (restricted to the processes 0 and 1 the alphabets
use).  For each discipline and event alphabet it reports the shortest trace that violates `safeB`
(an effect the specification forbids on the durable memory, or an acknowledged record missing from
it), or that none exists up to the depth bound, with the exact number of distinct states explored.
The search finds defects; the unbounded guarantee is `RCA.safe_run`.  Witnesses it reports are
re-checked by the kernel in `RCACounterexamples`.

Run from `lean/`: `lake env lean scripts/RCAModelCheck.lean`; the output is committed as
`lean/rca_model_check_receipt.txt`.
-/

open WaveLockProofs RCA

namespace RCAModelCheck

deriving instance Hashable for Policy, Rec, View, Disk, Phase, Done

abbrev Key := Disk × Phase × Phase × Option (List Rec × View) × Option (List Rec × View) ×
  Option ℕ × Issued × List Rec × List Done × List Rec

def key (S : Sys) : Key :=
  (S.disk, S.procs 0, S.procs 1, S.cache 0, S.cache 1, S.lock, S.issued, S.anchor, S.effects,
    S.acked)

/-- Breadth-first search; returns the shortest violating trace (if any within `depth`) and the
number of distinct states seen. -/
def search (D : Discipline) (alphabet : List Event) (depth : ℕ) :
    Option (List Event) × ℕ := Id.run do
  let s0 := step D Sys.init prov
  let mut frontier : Array (Sys × List Event) := #[(s0, [])]
  let mut seen : Std.HashSet Key := (Std.HashSet.emptyWithCapacity 1024).insert (key s0)
  for _ in [0:depth] do
    let mut next : Array (Sys × List Event) := #[]
    for (s, tr) in frontier do
      for e in alphabet do
        let s' := step D s e
        let tr' := tr ++ [e]
        if !safeB s' then return (some tr', seen.size)
        let k := key s'
        if !seen.contains k then
          seen := seen.insert k
          next := next.push (s', tr')
    frontier := next
  return (none, seen.size)

def procs : List ℕ := [0, 1]

/-- Requests of both processes: the latching and a benign observation, an action, a tightening and
a weakening policy, a reset, plus effects, crashes and restarts. -/
def broad : List Event :=
  procs.flatMap (fun q =>
    [.begin q, .submit q (.observe 1 [9]) .ok, .submit q (.observe 2 [0]) .ok,
     .submit q (.execute 3) .ok, .submit q (.policy ⟨1, 5, [3]⟩) .ok,
     .submit q (.policy ⟨1, 9, []⟩) .ok, .submit q (.policy ⟨2, 9, []⟩) .ok,
     .submit q (.reset 1) .ok, .effect q, .crash q, .restart q]) ++ [.issue 1]

/-- One process replaying a stale observation before a reset. -/
def replay : List Event :=
  [.begin 0, .submit 0 (.observe 2 [9]) .ok, .submit 0 (.observe 1 [0]) .ok, .issue 1,
   .submit 0 (.reset 1) .ok, .submit 0 (.execute 3) .ok, .effect 0]

/-- Requests with write failures, for the reference. -/
def faults : List Event :=
  procs.flatMap (fun q =>
    [.begin q, .submit q (.observe 1 [9]) .ok, .submit q (.observe 1 [9]) (.crash .torn),
     .submit q (.execute 3) .ok, .submit q (.execute 3) (.fail .line), .effect q, .crash q,
     .restart q])

def report (name : String) (D : Discipline) (alphabet : List Event) (depth : ℕ) : IO Unit := do
  let (w, n) := search D alphabet depth
  match w with
  | some tr => IO.println s!"{name}: shortest violation, length {tr.length} ({n} states): {repr tr}"
  | none => IO.println s!"{name}: no violation up to depth {depth} ({n} states, {alphabet.length} events)"

#eval report "reference, broad alphabet" .reference broad 8
#eval report "reference, write faults" .reference faults 8
#eval report "reference + directEffect" .reference (broad ++ [.directEffect 3]) 4
#eval report "reference + rollback" .reference (broad ++ [.rollback 1]) 5
#eval report "no lock" { lock := false } broad 6
#eval report "no reload" { reload := false } broad 8
#eval report "no persistence, no reload" { persist := false, reload := false } broad 8
#eval report "no drift gate" { driftGate := false } broad 6
#eval report "no monotone policy, no reload" { monotonePolicy := false, reload := false } broad 7
#eval report "no monotone policy (reload kept)" { monotonePolicy := false } broad 7
#eval report "reference, replay alphabet" .reference replay 10
#eval report "no sequence check, no reload" { seqCheck := false, reload := false } replay 10

end RCAModelCheck
