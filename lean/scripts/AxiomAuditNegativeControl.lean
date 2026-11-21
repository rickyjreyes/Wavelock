import WaveLockProofs.OTSSignerGuard

/-!
# Negative control for `scripts/AxiomAudit.lean`

Declares one theorem proved by `sorryAx` and one by `native_decide` and checks that the audit's
mechanism (`collectAxioms` against the approved set) flags both.  Run from `lean/`:
`lake env lean scripts/AxiomAuditNegativeControl.lean`.  This file is not part of the library.
-/

open Lean Elab Command

namespace WaveLockAuditNegativeControl

theorem bySorryAx : (2 : ℕ) + 2 = 5 := sorryAx _ false

theorem byNativeDecide : (10 : ℕ) < 20 := by native_decide

def approved : List Name := [``propext, ``Classical.choice, ``Quot.sound]

elab "#expect_flagged" : command => do
  for n in [``bySorryAx, ``byNativeDecide] do
    let axs ← liftCoreM (collectAxioms n)
    let bad := axs.filter fun a => !approved.contains a
    if bad.isEmpty then throwError "the audit would accept {n}"
    logInfo m!"{n} flagged: {bad.toList}"

end WaveLockAuditNegativeControl

#expect_flagged
