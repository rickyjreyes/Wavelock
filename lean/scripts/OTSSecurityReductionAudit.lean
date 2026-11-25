import WaveLockProofs.OTSSecurityReduction

open Lean Elab Command

/-- Strict audit of the entire new module, including generated declarations.
The historical receipt remains unchanged until the two audits are consolidated
after verification. -/
elab "#audit_ots_reduction" : command => do
  let env ← getEnv
  let moduleName := `WaveLockProofs.OTSSecurityReduction
  let some idx := env.getModuleIdx? moduleName | throwError "reduction module not imported"
  let names := (env.header.moduleData[idx.toNat]!).constNames
  let approved : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let mut failures : Array String := #[]
  for n in names do
    let some ci := env.find? n | continue
    let axs ← liftCoreM (collectAxioms n)
    for a in axs do
      unless approved.contains a do
        failures := failures.push s!"{n} depends on {a}"
    match ci with
    | .axiomInfo _ | .opaqueInfo _ => failures := failures.push s!"unexpected declaration {n}"
    | _ => pure ()
  for n in [`WaveLockProofs.OTSSecurityReduction.forgery_event_cover,
      `WaveLockProofs.OTSSecurityReduction.event_count_cover,
      `WaveLockProofs.OTSSecurityReduction.finite_forgery_count,
      `WaveLockProofs.OTSSecurityReduction.finite_uniform_bound] do
    match env.find? n with
    | some (.thmInfo _) =>
      let axs ← liftCoreM (collectAxioms n)
      logInfo m!"{n}: {axs.toList}"
    | _ => failures := failures.push s!"missing theorem {n}"
  unless failures.isEmpty do
    throwError "reduction audit failed: {failures.toList}"
  logInfo m!"Reduction audit passed: {names.size} constants"

#audit_ots_reduction
