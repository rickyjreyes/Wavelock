import WaveLockProofs.Algebra
import WaveLockProofs.NormativeConstants

/-!
# CC-Core-v1-B coupled round and replay model

The finite-field formulas and corrected hexadecimal constants below reproduce the *round-level*
equations of docs/CC_CORE_V1_NORMATIVE_PROTOCOL.md, including global time and
position weights. This does not yet model byte padding, bounded block counters,
IV construction, multi-block absorption, nor the four-round comparison squeeze.
Consequently replay lemmas here concern this specified round-level model, not
end-to-end executable parity with Python.
-/

namespace WaveLockProofs

variable {F : Type*} [Field F]

/-- The frozen 16x16 wave and accumulator fields. -/
abbrev CoupledState (F : Type*) := (TorusSite → F) × (TorusSite → F)

def candidateBWeight (t : Nat) (x : TorusSite) : F :=
  let cell : Nat := x.1.val * 16 + x.2.val + 1
  let step : Nat := t + 1
  1 + (step : F) * 40503 + (cell : F) * 50021 +
    (step : F) * (cell : F) * 60013

def candidateBRoundConstant (t : Nat) : F :=
  (ccCoreRho0 : F) + (ccCoreRho1 : F) * (t : F)

def candidateBWave (psi : TorusSite → F) : TorusSite → F :=
  waveRound torusLap 5 3 1431655765 psi

def candidateBAccumulator (t : Nat) (psi next c : TorusSite → F) :
    TorusSite → F :=
  fun x =>
    let cd := c x + 3 * torusLap c x
    5 * cd + 2 * cd ^ 2 +
      candidateBWeight t x * candidateBInjection 11 (psi x) (next x) +
      candidateBRoundConstant t

/-- Exact coupled algebraic round, using Jacobi updates of both fields. -/
def candidateBRound (t : Nat) (s : CoupledState F) : CoupledState F :=
  let next := candidateBWave s.1
  (next, candidateBAccumulator t s.1 next s.2)

theorem candidateBRound_wave (t : Nat) (s : CoupledState F) :
    (candidateBRound t s).1 = candidateBWave s.1 := rfl

theorem candidateBRound_acc (t : Nat) (s : CoupledState F) :
    (candidateBRound t s).2 =
      candidateBAccumulator t s.1 (candidateBWave s.1) s.2 := rfl

/-- Execute n rounds from absolute round t. The clock is not reset. -/
def runFrom (t : Nat) : Nat → CoupledState F → CoupledState F
  | 0, s => s
  | n + 1, s => runFrom (t + 1) n (candidateBRound t s)

theorem runFrom_zero (t : Nat) (s : CoupledState F) :
    runFrom t 0 s = s := rfl

theorem runFrom_succ (t n : Nat) (s : CoupledState F) :
    runFrom t (n + 1) s =
      runFrom (t + 1) n (candidateBRound t s) := rfl

theorem runFrom_two (t : Nat) (s : CoupledState F) :
    runFrom t 2 s =
      candidateBRound (t + 1) (candidateBRound t s) := rfl

/-- The output function is explicit: no unproved collision-resistance
    property is silently assigned to the squeeze. -/
def replayDigest {Digest : Type*} (output : CoupledState F → Digest)
    (rounds : Nat) (s : CoupledState F) : Digest :=
  output (runFrom 0 rounds s)

/-- Version matching and recomputation describe the verifier contract. -/
def ReplayAccept {Digest : Type*} (expected got : Nat)
    (output : CoupledState F → Digest) (rounds : Nat)
    (s : CoupledState F) (claimed : Digest) : Prop :=
  got = expected ∧ claimed = replayDigest output rounds s

theorem replay_accepts_honest {Digest : Type*}
    (version : Nat) (output : CoupledState F → Digest)
    (rounds : Nat) (s : CoupledState F) :
    ReplayAccept version version output rounds s
      (replayDigest output rounds s) := ⟨rfl, rfl⟩

theorem replay_rejects_wrong_version {Digest : Type*}
    (expected got : Nat) (h : got ≠ expected)
    (output : CoupledState F → Digest) (rounds : Nat)
    (s : CoupledState F) (claimed : Digest) :
    ¬ ReplayAccept expected got output rounds s claimed := by
  intro haccept
  exact h haccept.1

/-- Abstract length+version framing for a list of symbols. This is a
    separate structural lemma, NOT the full Python byte encoding. -/
def versionFrame (version : Nat) (payload : List Nat) : List Nat :=
  version :: payload.length :: payload

theorem versionFrame_payload_injective (version : Nat) :
    Function.Injective (versionFrame version) := by
  intro a b hab
  have htail := congrArg List.tail hab
  have htailtail := congrArg List.tail htail
  simpa [versionFrame] using htailtail

theorem versionFrame_separates_versions
    (v1 v2 : Nat) (h : v1 ≠ v2) (a b : List Nat) :
    versionFrame v1 a ≠ versionFrame v2 b := by
  intro heq
  have hhead := congrArg List.head? heq
  have hv : v1 = v2 := by simpa [versionFrame] using hhead
  exact h hv


/-- Parse the version-and-length envelope, rejecting any malformed length.
This is exact structural framing, not byte-level absorption or cryptography. -/
def parseVersionFrame : List Nat → Option (Nat × List Nat)
  | version :: declaredLen :: payload =>
      if declaredLen = payload.length then some (version, payload) else none
  | _ => none

/-- Every honestly constructed frame round-trips through the parser. -/
theorem parseVersionFrame_roundtrip (version : Nat) (payload : List Nat) :
    parseVersionFrame (versionFrame version payload) =
      some (version, payload) := by
  simp [parseVersionFrame, versionFrame]

/-- A forged length header is rejected before entering the protocol. -/
theorem parseVersionFrame_rejects_wrong_length
    (version declaredLen : Nat) (payload : List Nat)
    (h : declaredLen ≠ payload.length) :
    parseVersionFrame (version :: declaredLen :: payload) = none := by
  simp [parseVersionFrame, h]

/-- Successful parsing certifies the exact declared payload length. -/
theorem parseVersionFrame_correct_length
    (version declaredLen : Nat) (payload : List Nat)
    (resultVersion : Nat) (resultPayload : List Nat)
    (h : parseVersionFrame (version :: declaredLen :: payload) =
      some (resultVersion, resultPayload)) :
    declaredLen = payload.length ∧
      resultVersion = version ∧ resultPayload = payload := by
  by_cases hlen : declaredLen = payload.length
  · simp [parseVersionFrame, hlen] at h
    rcases h with ⟨rfl, rfl⟩
    exact ⟨hlen, rfl, rfl⟩
  · simp [parseVersionFrame, hlen] at h

end WaveLockProofs
