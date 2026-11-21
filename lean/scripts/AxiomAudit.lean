import WaveLockProofs

/-!
# Strict declaration and dependency audit

Run from `lean/`: `lake env lean scripts/AxiomAudit.lean`.  The output is committed as
`lean/axiom_audit_receipt.txt`; CI re-runs the audit and compares.

The audit fails (an `error`, non-zero exit) if

* an audited module is not part of the library;
* any constant defined in an audited module — every theorem, definition and generated auxiliary
  declaration, not only the named roots — depends on a foundational constant outside `approved`.  This catches
  `sorryAx` (`sorry`, `admit`, elaboration errors) and `Lean.ofReduceBool` (`native_decide`);
* an audited module declares an `axiomInfo` or `opaqueInfo` constant;
* a central root is missing or is not a theorem.

It reports exact counts of the declarations of each module and the axioms of each central root.
-/

open Lean Elab Command

namespace WaveLockAudit

/-- Every Lean module of the WaveLock-OTS formal work. -/
def auditedModules : List Name := [
  `WaveLockProofs.OTSKeyConsumption, `WaveLockProofs.LedgerLinearizability,
  `WaveLockProofs.KeyReuseCounterexample, `WaveLockProofs.OTSVerification,
  `WaveLockProofs.AuthenticatedBlockBinding, `WaveLockProofs.OTSConformance,
  `WaveLockProofs.OTSSignerGuard, `WaveLockProofs.OTSSignerConformance]

/-- Central theorem roots, by the property they establish. -/
def centralRoots : List (String × Name) := [
  ("signer: single use of key id and signing material",
    `WaveLockProofs.OTSSigner.signer_single_use),
  ("signer: secret values (given an injective derivation)",
    `WaveLockProofs.OTSSigner.signer_values_distinct),
  ("signer: safety memory append-only", `WaveLockProofs.OTSSigner.durable_markers_monotone),
  ("signer: present defect at c3cb0d1", `WaveLockProofs.OTSSigner.c3cb0d1_material_reused),
  ("signer: corrected guard refuses", `WaveLockProofs.OTSSigner.current_material_refused),
  ("signer: allow_reuse boundary", `WaveLockProofs.OTSSigner.allow_reuse_bypasses_guard),
  ("signer: Windows power-loss boundary", `WaveLockProofs.OTSSigner.windows_power_loss_forgets),
  ("signer: migration boundary", `WaveLockProofs.OTSSigner.legacy_material_signs_once_more),
  ("OTS (PR #47): accepted chain single use", `WaveLockProofs.OTS.chain_single_use),
  ("OTS (PR #47): at most one acceptance", `WaveLockProofs.OTS.at_most_one_acceptance)]

def approved : List Name := [``propext, ``Classical.choice, ``Quot.sound]

/-- Generated auxiliary declarations (equation lemmas, matchers, recursors, structure eta). -/
def isAuxiliary (env : Environment) (n : Name) : Bool :=
  n.isInternalDetail || isAuxRecursor env n || isNoConfusion env n ||
    (match n with
     | .str _ s => s.startsWith "eq_" || s == "injEq" || s == "inj" || s == "sizeOf_spec" ||
         s == "ctorIdx" || s.startsWith "match_" || s.startsWith "proof_"
     | _ => false)

elab "#strict_audit" : command => do
  let env ← getEnv
  let mut failures : Array String := #[]
  let mut total := (0, 0, 0, 0, 0)
  for m in auditedModules do
    let some idx := env.getModuleIdx? m | throwError "audited module {m} is not imported"
    let names := (env.header.moduleData[idx.toNat]!).constNames
    let mut thm := 0
    let mut defs := 0
    let mut insts := 0
    let mut types := 0
    let mut aux := 0
    for n in names do
      let some ci := env.find? n | continue
      let axs ← liftCoreM (collectAxioms n)
      for a in axs do
        unless approved.contains a do
          failures := failures.push s!"{n} depends on {a}"
      match ci with
      | .axiomInfo _ | .opaqueInfo _ => failures := failures.push s!"{m} declares {n}"
      | _ => pure ()
      if isAuxiliary env n then aux := aux + 1
      else match ci with
        | .thmInfo _ => thm := thm + 1
        | .defnInfo _ =>
          if (match n with | .str _ s => s.startsWith "inst" | _ => false) then insts := insts + 1
          else defs := defs + 1
        | .inductInfo _ => types := types + 1
        | _ => aux := aux + 1
    logInfo m!"{m}: {thm} theorems, {defs} definitions, {types} types, {insts} derived instances, {aux} auxiliary ({names.size} constants, all axiom-checked)"
    total := (total.1 + thm, total.2.1 + defs, total.2.2.1 + types, total.2.2.2.1 + insts,
      total.2.2.2.2 + aux)
  logInfo m!"total: {total.1} theorems, {total.2.1} definitions, {total.2.2.1} types, {total.2.2.2.1} derived instances, {total.2.2.2.2} auxiliary"
  let mut roots := 0
  for (what, n) in centralRoots do
    match env.find? n with
    | some (.thmInfo _) =>
      roots := roots + 1
      let axs ← liftCoreM (collectAxioms n)
      logInfo m!"root [{what}] {n}: {axs.toList}"
    | some _ => failures := failures.push s!"central root {n} is not a theorem"
    | none => failures := failures.push s!"central root {n} is missing"
  logInfo m!"audited theorem roots: {roots}; approved axioms: {approved}; unexpected axioms and failed declarations: {failures.toList}"
  unless failures.isEmpty do
    throwError "strict audit failed: {failures.toList}"

end WaveLockAudit

#strict_audit
