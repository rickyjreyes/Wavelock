import Mathlib.Tactic

/-!
# The WaveLock-OTS signing guard: the repository's preexisting safety memory and pre-action gate

At `rickyjreyes/Wavelock` `c3cb0d1` the only implemented mechanism of the shape *persistent
safety memory + pre-action intervention* is the signer side of WaveLock-OTS (issues #42/#43,
`docs/WAVELOCK_OTS_SIGNER_GUARD.md`).  It is not RCA agent logic.  The unsafe action is
revealing secret slices of one key twice (both halves of a position become public: forgery,
`attacks/ots_reuse_to_total_forgery.py`).  The safety memory is the host-local key-state
registry; the gate runs before any slice is revealed.

## What is modelled (Python reference)

* `wavelock/crypto/wavelock_ots.py :: sign_ots`: the scheme check; the advisory `used` flag
  (`OTSKeyReuseError`); unless `allow_reuse`, the atomic claims of the registry markers
  (`_claim_one_time_key`, and in this change `_claim_signing_material`), each an
  `atomic_write(..., exclusive=True)` (`os.link` of a flushed file, then `fsync_directory`);
  then the slices are revealed and the signature returned.
* `wavelock/crypto/keyfiles.py :: mark_secret_used` (the caller persists `used = true`).
* `_state_dir()`: the registry directory, `WAVELOCK_OTS_STATE_DIR`.

`Rev.c3cb0d1` claims the key-id marker only; `Rev.current` (this change) also the signing-material
marker, a domain-separated hash of `(params_hash, seed)`; `Rev.flagOnly` is the historical guard
before the registry (Finding D), with only the advisory flag.  Markers are injective in their key
id and material (collision resistance of SHA-256 is assumed).

## Platform

`dirSync = true` is POSIX: a successful claim is durable before `sign_ots` proceeds.  On Windows
`fsync_directory` is a no-op (`dirSync = false`), so a power loss can drop a claimed marker
(`powerLoss`).  A claim whose directory sync failed is published but not durable and raises.

## Three notions of reuse

`sign_ots` derives each revealed slice as `_secret_slice(seed, ψ-commitment(seed, params),
params_hash, i, b)` (`_h`: domain-separated, length-prefixed SHAKE-256).  The model's `mat` is the
class of the pair `(params_hash, seed)` that names the material marker.

| reuse | guarantee | status |
|---|---|---|
| same derivation inputs `(params_hash, seed)` | `signer_single_use` | `LEAN_PROVED` (model) + conformance |
| same secret values through `sign_ots` | `signer_values_distinct` given an injective derivation (collision resistance) | `CONDITIONAL` |
| same public commitment table | signer side: as the previous row (the table is a function of `params_hash` and the slices); verifier side: PR #47, `wavelock_ots_material_single_use` | `CONDITIONAL` / `LEAN_PROVED` |
| same secret values recommitted by custom code (not `sign_ots`) | none: no guard runs on that path | `OPEN` |

Equal inputs always give equal markers, so the guard's safety direction needs no hash assumption:
a collision between two *different* materials' markers can only cause a refusal.

## Results

* `signer_single_use` (`LEAN_PROVED`, for `Rev.current`): with one registry directory, POSIX
  durability and no `allow_reuse`, no two signatures — over any interleaving of any number of
  processes, crashes, claim failures, restarts and power losses — share a key id **or** a signing
  material.
* `c3cb0d1_material_reused` (`COUNTEREXAMPLE`, present defect at `c3cb0d1`, corrected here): two
  key files holding one seed under two key ids both sign.
* Boundaries (`COUNTEREXAMPLE`, not corrected): `allow_reuse`, a second registry directory, a
  wiped registry, Windows power loss, and material that signed before this change.
-/

namespace WaveLockProofs
namespace OTSSigner

/-- A marker file of the key-state registry. -/
inductive Marker where
  | kid (k : ℕ)
  | mat (m : ℕ)
  deriving DecidableEq, Repr

/-- A secret-key file as `sign_ots` reads it: key-id label, signing material
(`(params_hash, seed)`, as a code) and the advisory `used` flag. -/
structure KeyFile where
  kid : ℕ
  mat : ℕ
  used : Bool
  deriving DecidableEq, Repr

/-- Revision of the signing guard. -/
inductive Rev where
  | flagOnly
  | c3cb0d1
  | current
  deriving DecidableEq, Repr

/-- Markers a revision claims before revealing. -/
def Rev.markers : Rev → KeyFile → List Marker
  | .flagOnly, _ => []
  | .c3cb0d1, f => [.kid f.kid]
  | .current, f => [.kid f.kid, .mat f.mat]

/-- Outcome of claiming an absent marker. -/
inductive ClaimOutcome where
  /-- published, durable when the platform syncs directories -/
  | ok
  /-- `OSError` before the link: nothing published, `WaveLockOTSError` -/
  | failBefore
  /-- linked, then the directory `fsync` failed: published, not durable, `WaveLockOTSError` -/
  | failAfter
  deriving DecidableEq, Repr

/-- Control state of one signing process.  `claiming fid f dir msg att todo`: inside `sign_ots`
for file `fid` (read as `f`), registry `dir`, message `msg`, attempt `att`, with markers `todo`
still to claim. -/
inductive Phase where
  | idle
  | dead
  | claiming (fid : ℕ) (f : KeyFile) (dir msg att : ℕ) (todo : List Marker)
  | signed (fid : ℕ)
  deriving DecidableEq, Repr

/-- A signature that left `sign_ots` (the earliest point its slices can be published). -/
structure Sig where
  kid : ℕ
  mat : ℕ
  msg : ℕ
  att : ℕ
  deriving DecidableEq, Repr

structure Sys where
  files : ℕ → KeyFile
  /-- per registry directory: marker and whether it is durable -/
  reg : ℕ → List (Marker × Bool)
  procs : ℕ → Phase
  /-- ghost: signatures returned by `sign_ots` -/
  sigs : List Sig
  /-- ghost: successful claims (marker, attempt) -/
  claims : List (Marker × ℕ)
  /-- ghost: next attempt id -/
  next : ℕ

inductive Event where
  /-- a secret-key file is written (key generation, a copy, or a relabelled copy) -/
  | mkFile (fid kid mat : ℕ)
  /-- `sign_ots(file fid, msg)` starts under revision `rev`, registry `dir` -/
  | start (p fid dir msg : ℕ) (rev : Rev) (allowReuse : Bool)
  | claim (p : ℕ) (o : ClaimOutcome)
  /-- slices are derived and the signature returned -/
  | reveal (p : ℕ)
  /-- the caller's `mark_secret_used` -/
  | persist (p : ℕ)
  | crash (p : ℕ)
  | restart (p : ℕ)
  /-- non-durable registry entries vanish -/
  | powerLoss
  /-- the registry directory is deleted (out of boundary) -/
  | wipe (dir : ℕ)
  deriving DecidableEq, Repr

def upd {α : Type} (f : ℕ → α) (i : ℕ) (x : α) : ℕ → α := fun j => if j = i then x else f j

def emptyFile : KeyFile := ⟨0, 0, false⟩

def Sys.init : Sys := ⟨fun _ => emptyFile, fun _ => [], fun _ => .idle, [], [], 0⟩

/-- Whether marker `m` exists in registry list `r` (durable or not). -/
def present (r : List (Marker × Bool)) (m : Marker) : Bool := r.any fun x => x.1 == m

def step (dirSync : Bool) (S : Sys) : Event → Sys
  | .mkFile fid kid mat => { S with files := upd S.files fid ⟨kid, mat, false⟩ }
  | .start p fid dir msg rev allow =>
    match S.procs p with
    | .idle =>
      if (S.files fid).used && !allow then S
      else { S with procs := upd S.procs p
                       (.claiming fid (S.files fid) dir msg S.next
                          (if allow then [] else rev.markers (S.files fid))),
                    next := S.next + 1 }
    | _ => S
  | .claim p o =>
    match S.procs p with
    | .claiming fid f dir msg a (m :: todo) =>
      if present (S.reg dir) m then { S with procs := upd S.procs p .idle }
      else
        match o with
        | .ok => { S with reg := upd S.reg dir (S.reg dir ++ [(m, dirSync)]),
                          procs := upd S.procs p (.claiming fid f dir msg a todo),
                          claims := S.claims ++ [(m, a)] }
        | .failBefore => { S with procs := upd S.procs p .idle }
        | .failAfter => { S with reg := upd S.reg dir (S.reg dir ++ [(m, false)]),
                                 procs := upd S.procs p .idle }
    | _ => S
  | .reveal p =>
    match S.procs p with
    | .claiming fid f _ msg a [] =>
      { S with sigs := S.sigs ++ [⟨f.kid, f.mat, msg, a⟩], procs := upd S.procs p (.signed fid) }
    | _ => S
  | .persist p =>
    match S.procs p with
    | .signed fid => { S with files := upd S.files fid { S.files fid with used := true },
                              procs := upd S.procs p .idle }
    | _ => S
  | .crash p => { S with procs := upd S.procs p .dead }
  | .restart p =>
    match S.procs p with
    | .dead => { S with procs := upd S.procs p .idle }
    | _ => S
  | .powerLoss => { S with reg := fun d => (S.reg d).filter fun x => x.2 }
  | .wipe dir => { S with reg := upd S.reg dir [] }

def run (dirSync : Bool) (es : List Event) : Sys := es.foldl (step dirSync) Sys.init

/-- Events allowed by `signer_single_use`: the current guard without `allow_reuse`, one registry
directory `d`, and no deletion of the registry. -/
def Allowed (d : ℕ) : Event → Prop
  | .start _ _ dir _ rev allow => dir = d ∧ rev = .current ∧ allow = false
  | .wipe _ => False
  | _ => True

/-! ## The invariant -/

section Invariant

variable (d : ℕ)

/-- Inductive invariant of the current guard on registry `d` with POSIX durability. -/
structure Inv (S : Sys) : Prop where
  /-- every successful claim is a durable marker of registry `d` -/
  durable : ∀ x ∈ S.claims, (x.1, true) ∈ S.reg d
  /-- no marker was claimed successfully twice -/
  claimsNodup : (S.claims.map Prod.fst).Nodup
  /-- every signature's attempt claimed its key id and its material -/
  sigClaims : ∀ s ∈ S.sigs, (Marker.kid s.kid, s.att) ∈ S.claims ∧ (Marker.mat s.mat, s.att) ∈ S.claims
  /-- each attempt signed at most once -/
  attNodup : (S.sigs.map Sig.att).Nodup
  /-- an in-flight attempt claims the current markers of its file, has not signed, and is not
  shared with another process -/
  inflight : ∀ p fid f dir msg a todo, S.procs p = .claiming fid f dir msg a todo →
    dir = d ∧ (∃ done, Rev.current.markers f = done ++ todo ∧ ∀ m ∈ done, (m, a) ∈ S.claims) ∧
      a ∉ S.sigs.map Sig.att ∧ a < S.next
  distinct : ∀ p q fid f dir msg a todo fid' f' dir' msg' a' todo', p ≠ q →
    S.procs p = .claiming fid f dir msg a todo → S.procs q = .claiming fid' f' dir' msg' a' todo' →
      a ≠ a'
  /-- recorded attempts are older than `next` -/
  claimsOld : ∀ x ∈ S.claims, x.2 < S.next
  sigsOld : ∀ s ∈ S.sigs, s.att < S.next

theorem inv_init : Inv d Sys.init := by
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;> simp [Sys.init]

variable {d}

theorem upd_self {α : Type} (f : ℕ → α) (i : ℕ) (x : α) : upd f i x i = x := by simp [upd]

theorem upd_ne {α : Type} (f : ℕ → α) {i j : ℕ} (x : α) (h : j ≠ i) : upd f i x j = f j := by
  simp [upd, h]

/-- A process whose phase is replaced by a non-claiming phase no longer constrains anything. -/
theorem inflight_set {S : Sys} (_h : Inv d S) {p : ℕ} {ph : Phase}
    (hph : ∀ fid f dir msg a todo, ph ≠ .claiming fid f dir msg a todo) :
    ∀ x fid f dir msg a todo, upd S.procs p ph x = .claiming fid f dir msg a todo →
      S.procs x = .claiming fid f dir msg a todo ∧ x ≠ p := by
  intro x fid f dir msg a todo hx
  by_cases hxp : x = p
  · subst hxp; simp only [upd_self] at hx; exact absurd hx (hph _ _ _ _ _ _)
  · simp only [upd_ne _ _ hxp] at hx; exact ⟨hx, hxp⟩

theorem present_false {r : List (Marker × Bool)} {m : Marker} (h : present r m = false) :
    ∀ b, (m, b) ∉ r := by
  intro b hb
  have : present r m = true := List.any_eq_true.2 ⟨(m, b), hb, by simp⟩
  rw [h] at this; cases this

end Invariant

section Preservation

variable {d : ℕ} {S : Sys}

theorem inv_step (h : Inv d S) {e : Event} (he : Allowed d e) : Inv d (step true S e) := by
  cases e with
  | mkFile fid kid mat =>
    exact ⟨h.durable, h.claimsNodup, h.sigClaims, h.attNodup, h.inflight, h.distinct,
      h.claimsOld, h.sigsOld⟩
  | start p fid dir msg rev allow =>
    obtain ⟨rfl, rfl, rfl⟩ := he
    cases hp : S.procs p with
    | idle =>
      simp only [step, hp, Bool.not_false, Bool.and_true, Bool.false_eq_true, if_false]
      split_ifs with hu
      · exact h
      · refine ⟨h.durable, h.claimsNodup, h.sigClaims, h.attNodup, ?_, ?_, ?_, ?_⟩
        · intro x fid' f dir msg a todo hx
          by_cases hxp : x = p
          · subst hxp; simp only [upd_self] at hx
            simp only [Phase.claiming.injEq] at hx
            obtain ⟨-, rfl, rfl, -, rfl, rfl⟩ := hx
            refine ⟨rfl, ⟨[], by simp, by simp⟩, ?_, Nat.lt_succ_self _⟩
            intro hm; obtain ⟨s, hs, hsa⟩ := List.mem_map.1 hm
            have := h.sigsOld s hs; omega
          · simp only [upd_ne _ _ hxp] at hx
            obtain ⟨h1, h2, h3, h4⟩ := h.inflight x _ _ _ _ _ _ hx
            exact ⟨h1, h2, h3, Nat.lt_succ_of_lt h4⟩
        · intro x y fid1 f1 dir1 msg1 a1 todo1 fid2 f2 dir2 msg2 a2 todo2 hxy hx hy
          by_cases hxp : x = p
          · subst hxp
            simp only [upd_self] at hx; simp only [upd_ne _ _ (Ne.symm hxy)] at hy
            simp only [Phase.claiming.injEq] at hx
            have := (h.inflight y _ _ _ _ _ _ hy).2.2.2; omega
          · by_cases hyp : y = p
            · subst hyp
              simp only [upd_self] at hy; simp only [upd_ne _ _ hxp] at hx
              simp only [Phase.claiming.injEq] at hy
              have := (h.inflight x _ _ _ _ _ _ hx).2.2.2; omega
            · simp only [upd_ne _ _ hxp] at hx; simp only [upd_ne _ _ hyp] at hy
              exact h.distinct x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
        · intro x hx; have := h.claimsOld x hx; show x.2 < S.next + 1; omega
        · intro s hs; have := h.sigsOld s hs; show s.att < S.next + 1; omega
    | _ => simp only [step, hp]; exact h
  | claim p o =>
    cases hp : S.procs p with
    | claiming fid f dir msg a todo =>
      cases todo with
      | nil => simp only [step, hp]; exact h
      | cons m todo =>
        obtain ⟨hdir, ⟨done, hdone, hclaimed⟩, hnsig, hlt⟩ := h.inflight p _ _ _ _ _ _ hp
        subst hdir
        simp only [step, hp]
        cases hpr : present (S.reg dir) m with
        | true =>
          simp only [if_true]
          refine ⟨h.durable, h.claimsNodup, h.sigClaims, h.attNodup, ?_, ?_, h.claimsOld, h.sigsOld⟩
          · intro x fid' f' dir' msg' a' todo' hx
            obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
            exact h.inflight x _ _ _ _ _ _ hx
          · intro x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
            obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
            obtain ⟨hy, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) y _ _ _ _ _ _ hy
            exact h.distinct x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
        | false =>
          simp only [Bool.false_eq_true, if_false]
          have habs := present_false hpr
          cases o with
          | ok =>
            refine ⟨?_, ?_, ?_, h.attNodup, ?_, ?_, ?_, h.sigsOld⟩
            · intro x hx
              rcases List.mem_append.1 hx with hx | hx
              · have := h.durable x hx
                by_cases hdd : dir = dir
                · simp only [upd_self]; exact List.mem_append_left _ this
                · exact absurd rfl hdd
              · rw [List.mem_singleton] at hx; subst hx; simp [upd_self]
            · rw [List.map_append, List.nodup_append]
              refine ⟨h.claimsNodup, List.nodup_singleton _, ?_⟩
              intro x hx y hy hxy
              simp only [List.map_cons, List.map_nil, List.mem_singleton] at hy
              subst hy; subst hxy
              obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hx
              exact habs true (h.durable c hc)
            · intro s hs
              obtain ⟨h1, h2⟩ := h.sigClaims s hs
              exact ⟨List.mem_append_left _ h1, List.mem_append_left _ h2⟩
            · intro x fid' f' dir' msg' a' todo' hx
              by_cases hxp : x = p
              · subst hxp; simp only [upd_self] at hx
                simp only [Phase.claiming.injEq] at hx
                obtain ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ := hx
                refine ⟨rfl, ⟨done ++ [m], by rw [hdone]; simp, ?_⟩, hnsig, hlt⟩
                intro m' hm'
                rcases List.mem_append.1 hm' with hm' | hm'
                · exact List.mem_append_left _ (hclaimed m' hm')
                · rw [List.mem_singleton] at hm'; subst hm'; simp
              · simp only [upd_ne _ _ hxp] at hx
                obtain ⟨h1, ⟨dn, h2, h3⟩, h4, h5⟩ := h.inflight x _ _ _ _ _ _ hx
                exact ⟨h1, ⟨dn, h2, fun m' hm' => List.mem_append_left _ (h3 m' hm')⟩, h4, h5⟩
            · intro x y fid1 f1 dir1 msg1 a1 todo1 fid2 f2 dir2 msg2 a2 todo2 hxy hx hy
              have hx' : S.procs x = .claiming fid1 f1 dir1 msg1 a1 todo1 ∨
                  (x = p ∧ a1 = a) := by
                by_cases hxp : x = p
                · subst hxp; simp only [upd_self] at hx; simp only [Phase.claiming.injEq] at hx
                  exact Or.inr ⟨rfl, hx.2.2.2.2.1.symm⟩
                · simp only [upd_ne _ _ hxp] at hx; exact Or.inl hx
              have hy' : S.procs y = .claiming fid2 f2 dir2 msg2 a2 todo2 ∨
                  (y = p ∧ a2 = a) := by
                by_cases hyp : y = p
                · subst hyp; simp only [upd_self] at hy; simp only [Phase.claiming.injEq] at hy
                  exact Or.inr ⟨rfl, hy.2.2.2.2.1.symm⟩
                · simp only [upd_ne _ _ hyp] at hy; exact Or.inl hy
              rcases hx' with hx' | ⟨hxp, hxa⟩ <;> rcases hy' with hy' | ⟨hyp, hya⟩
              · exact h.distinct x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx' hy'
              · rw [hya]; rw [hyp] at hxy
                exact h.distinct x p _ _ _ _ _ _ _ _ _ _ _ _ hxy hx' hp
              · rw [hxa]; rw [hxp] at hxy
                exact h.distinct p y _ _ _ _ _ _ _ _ _ _ _ _ hxy hp hy'
              · exact absurd (hxp.trans hyp.symm) hxy
            · intro x hx
              rcases List.mem_append.1 hx with hx | hx
              · exact h.claimsOld x hx
              · rw [List.mem_singleton] at hx; subst hx; exact hlt
          | failBefore | failAfter =>
            all_goals
              refine ⟨?_, h.claimsNodup, h.sigClaims, h.attNodup, ?_, ?_, h.claimsOld, h.sigsOld⟩
              · first
                | exact h.durable
                | (intro x hx
                   have := h.durable x hx
                   simp only [upd_self]; exact List.mem_append_left _ this)
              · intro x fid' f' dir' msg' a' todo' hx
                obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
                exact h.inflight x _ _ _ _ _ _ hx
              · intro x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
                obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
                obtain ⟨hy, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) y _ _ _ _ _ _ hy
                exact h.distinct x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
    | _ => simp only [step, hp]; exact h
  | reveal p =>
    cases hp : S.procs p with
    | claiming fid f dir msg a todo =>
      cases todo with
      | cons m todo => simp only [step, hp]; exact h
      | nil =>
        obtain ⟨-, ⟨done, hdone, hclaimed⟩, hnsig, hlt⟩ := h.inflight p _ _ _ _ _ _ hp
        simp only [step, hp]
        simp only [Rev.markers, List.append_nil] at hdone
        refine ⟨h.durable, h.claimsNodup, ?_, ?_, ?_, ?_, h.claimsOld, ?_⟩
        · intro s hs
          rcases List.mem_append.1 hs with hs | hs
          · exact h.sigClaims s hs
          · rw [List.mem_singleton] at hs; subst hs
            exact ⟨hclaimed _ (by rw [← hdone]; simp), hclaimed _ (by rw [← hdone]; simp)⟩
        · rw [List.map_append, List.nodup_append]
          refine ⟨h.attNodup, List.nodup_singleton _, ?_⟩
          intro x hx y hy hxy
          simp only [List.map_cons, List.map_nil, List.mem_singleton] at hy
          subst hy; subst hxy; exact hnsig hx
        · intro x fid' f' dir' msg' a' todo' hx
          obtain ⟨hx, hxp⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
          obtain ⟨h1, h2, h3, h4⟩ := h.inflight x _ _ _ _ _ _ hx
          refine ⟨h1, h2, ?_, h4⟩
          intro hm
          rw [List.map_append, List.mem_append] at hm
          rcases hm with hm | hm
          · exact h3 hm
          · simp only [List.map_cons, List.map_nil, List.mem_singleton] at hm
            exact h.distinct x p _ _ _ _ _ _ _ _ _ _ _ _ hxp hx hp hm
        · intro x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
          obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
          obtain ⟨hy, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) y _ _ _ _ _ _ hy
          exact h.distinct x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
        · intro s hs
          rcases List.mem_append.1 hs with hs | hs
          · exact h.sigsOld s hs
          · rw [List.mem_singleton] at hs; subst hs; exact hlt
    | _ => simp only [step, hp]; exact h
  | persist p =>
    cases hp : S.procs p with
    | signed fid =>
      simp only [step, hp]
      refine ⟨h.durable, h.claimsNodup, h.sigClaims, h.attNodup, ?_, ?_, h.claimsOld, h.sigsOld⟩
      · intro x _ _ _ _ _ _ hx
        obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
        exact h.inflight x _ _ _ _ _ _ hx
      · intro x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
        obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
        obtain ⟨hy, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) y _ _ _ _ _ _ hy
        exact h.distinct x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
    | _ => simp only [step, hp]; exact h
  | crash p =>
    simp only [step]
    refine ⟨h.durable, h.claimsNodup, h.sigClaims, h.attNodup, ?_, ?_, h.claimsOld, h.sigsOld⟩
    · intro x _ _ _ _ _ _ hx
      obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
      exact h.inflight x _ _ _ _ _ _ hx
    · intro x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
      obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
      obtain ⟨hy, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) y _ _ _ _ _ _ hy
      exact h.distinct x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
  | restart p =>
    cases hp : S.procs p with
    | dead =>
      simp only [step, hp]
      refine ⟨h.durable, h.claimsNodup, h.sigClaims, h.attNodup, ?_, ?_, h.claimsOld, h.sigsOld⟩
      · intro x _ _ _ _ _ _ hx
        obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
        exact h.inflight x _ _ _ _ _ _ hx
      · intro x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
        obtain ⟨hx, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) x _ _ _ _ _ _ hx
        obtain ⟨hy, -⟩ := inflight_set h (by intro _ _ _ _ _ _ hc; cases hc) y _ _ _ _ _ _ hy
        exact h.distinct x y _ _ _ _ _ _ _ _ _ _ _ _ hxy hx hy
    | _ => simp only [step, hp]; exact h
  | powerLoss =>
    simp only [step]
    refine ⟨?_, h.claimsNodup, h.sigClaims, h.attNodup, h.inflight, h.distinct, h.claimsOld,
      h.sigsOld⟩
    intro x hx; exact List.mem_filter.2 ⟨h.durable x hx, rfl⟩
  | wipe dir => exact absurd he id

theorem inv_foldl : ∀ (es : List Event) (S : Sys), Inv d S → (∀ e ∈ es, Allowed d e) →
    Inv d (es.foldl (step true) S)
  | [], _, h, _ => h
  | e :: es, S, h, hb => by
    simp only [List.foldl_cons]
    exact inv_foldl es _ (inv_step h (hb e List.mem_cons_self)) fun x hx =>
      hb x (List.mem_cons_of_mem _ hx)

/-- **Signer single use** (`Rev.current`, POSIX durability).  In every run in which every signing
call uses the current guard without `allow_reuse` on registry `d`, and the registry is never
deleted, the signatures that left `sign_ots` have pairwise distinct key ids and pairwise distinct
signing materials — over any interleaving of any number of processes, claim failures, crashes,
restarts and power losses, and whatever key files (copies, relabelled copies, keys generated from
one seed) are presented. -/
theorem signer_single_use {es : List Event} (hb : ∀ e ∈ es, Allowed d e) :
    ((run true es).sigs.map Sig.kid).Nodup ∧ ((run true es).sigs.map Sig.mat).Nodup := by
  have h := inv_foldl es _ (inv_init d) hb
  have hsn : (run true es).sigs.Nodup := List.Nodup.of_map _ h.attNodup
  have hatt := List.inj_on_of_nodup_map h.attNodup
  have hcl := List.inj_on_of_nodup_map h.claimsNodup
  constructor
  · refine List.Nodup.map_on ?_ hsn
    intro x hx y hy hxy
    have := hcl (h.sigClaims x hx).1 (h.sigClaims y hy).1 (by simp [hxy])
    simp only [Prod.mk.injEq] at this
    exact hatt hx hy this.2
  · refine List.Nodup.map_on ?_ hsn
    intro x hx y hy hxy
    have := hcl (h.sigClaims x hx).2 (h.sigClaims y hy).2 (by simp [hxy])
    simp only [Prod.mk.injEq] at this
    exact hatt hx hy this.2

/-- **Same secret values (`CONDITIONAL`).**  If the slice derivation is injective on signing
materials (for distinct materials: collision resistance of SHAKE-256), no two signatures that left
`sign_ots` reveal slices of the same secret table. -/
theorem signer_values_distinct {es : List Event} (hb : ∀ e ∈ es, Allowed d e) {β : Type}
    (derive : ℕ → β) (hinj : Function.Injective derive) :
    ((run true es).sigs.map fun s => derive s.mat).Nodup := by
  have h := (signer_single_use hb).2
  rw [show (fun s : Sig => derive s.mat) = derive ∘ Sig.mat from rfl, ← List.map_map]
  exact h.map hinj

/-- The safety memory of the signer is append-only inside the boundary: no allowed event removes
a durable marker (power loss removes only markers that were never durable). -/
theorem durable_markers_monotone {e : Event} (he : Allowed d e) (dir : ℕ) (m : Marker)
    (hm : (m, true) ∈ S.reg dir) : (m, true) ∈ (step true S e).reg dir := by
  cases e with
  | claim p o =>
    cases hp : S.procs p with
    | claiming fid f dir' msg a todo =>
      cases todo with
      | nil => simp only [step, hp]; exact hm
      | cons m' todo =>
        simp only [step, hp]
        split_ifs
        · exact hm
        · cases o <;> dsimp only <;> (try simp only [upd]) <;> (try split_ifs with hd) <;>
            first | exact hm | (subst hd; exact List.mem_append_left _ hm)
    | _ => simp only [step, hp]; exact hm
  | powerLoss => simp only [step]; exact List.mem_filter.2 ⟨hm, rfl⟩
  | wipe _ => exact absurd he id
  | start p fid dir' msg rev allow =>
    simp only [step]; split <;> try exact hm
    split_ifs <;> exact hm
  | reveal p => simp only [step]; split <;> exact hm
  | persist p => simp only [step]; split <;> exact hm
  | restart p => simp only [step]; split <;> exact hm
  | mkFile _ _ _ => exact hm
  | crash _ => exact hm

end Preservation

/-! ## Counterexamples (kernel-checked traces) -/

section Counterexamples

/-- Two key files with one signing material (`generate_ots_keypair(seed=s)` twice) and key ids
1 and 2, signed one after the other on registry 0. -/
def twoKeysOneSeed (rev : Rev) : List Event :=
  [.mkFile 0 1 7, .mkFile 1 2 7,
   .start 0 0 0 100 rev false, .claim 0 .ok, .claim 0 .ok, .reveal 0, .persist 0,
   .start 1 1 0 200 rev false, .claim 1 .ok, .claim 1 .ok, .reveal 1, .persist 1]

/-- **Present defect at `c3cb0d1`** (corrected here): the registry claimed only the key id, so one
seed under two key ids signed twice (both halves of every position where the two digests differ
become public; reproduced against the Python code by the conformance test). -/
theorem c3cb0d1_material_reused :
    (run true (twoKeysOneSeed .c3cb0d1)).sigs.map Sig.mat = [7, 7] := by decide

/-- The corrected guard refuses the second key file: its material marker exists. -/
theorem current_material_refused :
    (run true (twoKeysOneSeed .current)).sigs.map Sig.mat = [7] := by decide

/-- A relabelled copy of a used secret-key file (new key id, same seed) under `c3cb0d1`. -/
theorem c3cb0d1_relabelled_copy_signs :
    (run true [.mkFile 0 1 7, .start 0 0 0 100 .c3cb0d1 false, .claim 0 .ok, .reveal 0,
      .persist 0, .mkFile 1 9 7, .start 1 1 0 200 .c3cb0d1 false, .claim 1 .ok,
      .reveal 1]).sigs.length = 2 := by decide

/-- **Boundary (historical, Finding D).**  The advisory `used` flag alone: a copy taken before
the first signature signs again. -/
theorem flag_only_copy_signs_twice :
    (run true [.mkFile 0 1 7, .mkFile 1 1 7, .start 0 0 0 100 .flagOnly false, .reveal 0,
      .persist 0, .start 1 1 0 200 .flagOnly false, .reveal 1]).sigs.length = 2 := by decide

/-- The registry stops that copy (same key id). -/
theorem current_copy_refused :
    (run true [.mkFile 0 1 7, .mkFile 1 1 7, .start 0 0 0 100 .current false, .claim 0 .ok,
      .claim 0 .ok, .reveal 0, .persist 0, .start 1 1 0 200 .current false, .claim 1 .ok,
      .claim 1 .ok, .reveal 1]).sigs.length = 1 := by decide

/-- **Boundary: `allow_reuse=True`** skips the registry (documented as tests only). -/
theorem allow_reuse_bypasses_guard :
    (run true [.mkFile 0 1 7, .start 0 0 0 100 .current true, .reveal 0, .persist 0,
      .start 1 0 0 200 .current true, .reveal 1]).sigs.map Sig.kid = [1, 1] := by decide

/-- **Boundary: a second registry directory** (`WAVELOCK_OTS_STATE_DIR` changed, or another
host) is a separate safety memory. -/
theorem second_registry_bypasses_guard :
    (run true [.mkFile 0 1 7, .start 0 0 0 100 .current false, .claim 0 .ok, .claim 0 .ok,
      .reveal 0, .mkFile 1 1 7, .start 1 1 1 200 .current false, .claim 1 .ok, .claim 1 .ok,
      .reveal 1]).sigs.length = 2 := by decide

/-- **Boundary: a deleted registry** forgets every claim. -/
theorem wiped_registry_forgets :
    (run true [.mkFile 0 1 7, .mkFile 1 1 7, .start 0 0 0 100 .current false, .claim 0 .ok,
      .claim 0 .ok, .reveal 0, .wipe 0, .start 1 1 0 200 .current false, .claim 1 .ok,
      .claim 1 .ok, .reveal 1]).sigs.length = 2 := by decide

/-- **Boundary: Windows power loss.**  Without directory `fsync` a claimed marker is not durable;
a power loss after the signature lets a copy sign again (issue #26). -/
theorem windows_power_loss_forgets :
    (run false [.mkFile 0 1 7, .mkFile 1 1 7, .start 0 0 0 100 .current false, .claim 0 .ok,
      .claim 0 .ok, .reveal 0, .powerLoss, .start 1 1 0 200 .current false, .claim 1 .ok,
      .claim 1 .ok, .reveal 1]).sigs.length = 2 := by decide

/-- The same power loss on POSIX: the markers were durable before any slice was revealed. -/
theorem posix_power_loss_keeps :
    (run true [.mkFile 0 1 7, .mkFile 1 1 7, .start 0 0 0 100 .current false, .claim 0 .ok,
      .claim 0 .ok, .reveal 0, .powerLoss, .start 1 1 0 200 .current false, .claim 1 .ok,
      .claim 1 .ok, .reveal 1]).sigs.length = 1 := by decide

/-- **Boundary: migration.**  Material that signed under `c3cb0d1` has no material marker; a
relabelled copy signs once more under the corrected guard. -/
theorem legacy_material_signs_once_more :
    (run true [.mkFile 0 1 7, .start 0 0 0 100 .c3cb0d1 false, .claim 0 .ok, .reveal 0,
      .mkFile 1 2 7, .start 1 1 0 200 .current false, .claim 1 .ok, .claim 1 .ok,
      .reveal 1]).sigs.map Sig.mat = [7, 7] := by decide

/-- A crash after the claims and before any slice is revealed burns the key: a retry is refused
(`tests/test_bounty_durability.py :: test_process_exit_after_signer_claim_burns_key`). -/
theorem crash_after_claims_burns_key :
    (run true [.mkFile 0 1 7, .start 0 0 0 100 .current false, .claim 0 .ok, .claim 0 .ok,
      .crash 0, .restart 0, .start 0 0 0 200 .current false, .claim 0 .ok,
      .reveal 0]).sigs = [] := by decide

/-- A claim whose directory sync failed raises before signing; the published marker still blocks a
retry (`tests/test_bounty_durability.py :: test_directory_fsync_failure_does_not_report_success`). -/
theorem failed_dir_sync_blocks_retry :
    (run true [.mkFile 0 1 7, .start 0 0 0 100 .current false, .claim 0 .failAfter,
      .start 0 0 0 200 .current false, .claim 0 .ok, .reveal 0]).sigs = [] := by decide

end Counterexamples

end OTSSigner
end WaveLockProofs
