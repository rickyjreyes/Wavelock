import WaveLockProofs.SecurityGames

/-!
# WaveLock-OTS public verification (`verify_ots`) as an explicit check sequence

Transcription of `wavelock/crypto/wavelock_ots.py :: verify_ots` and
`_check_public_key_canonical` at revision `b310ea5` (scheme `WaveLock-OTS-v1`, `VERSION = 1`,
`HASH_ALG = "SHAKE256-256"`).  Objects are the dictionaries returned by `json.loads`.  Every
hash-derived quantity is an opaque parameter (`OTSCrypto`); the Python operations the checks rely
on (`str(x)`, `x == y`) are parameters with only the properties CPython guarantees (`PyEnv`).
No collision resistance, preimage resistance or unforgeability is assumed anywhere in this file;
where such a property matters the theorem *extracts* the corresponding event (`Collision`).

Every failure in `verify_ots` returns `False` (including exceptions, which the `try` maps to
`False`), so the verifier is the conjunction of its checks and their order is irrelevant.
-/

namespace WaveLockProofs
namespace OTS

/-- JSON values as returned by Python's `json.loads` (finite numbers; `num` is a Python float). -/
inductive JVal where
  | null
  | bool (b : Bool)
  | int (n : Int)
  | num (q : ℚ)
  | str (s : String)
  | arr (xs : List JVal)
  | obj (kvs : List (String × JVal))

/-- A parsed JSON object (Python `dict`). -/
abbrev JObj := List (String × JVal)

namespace JObj

def get (o : JObj) (k : String) : Option JVal := o.lookup k

def keys (o : JObj) : List String := o.map Prod.fst

/-- `set(d.keys()) == set(FIELDS)`. -/
def hasExactly (o : JObj) (fs : List String) : Bool :=
  (JObj.keys o).all (· ∈ fs) && fs.all (· ∈ JObj.keys o)

/-- Replace the value stored under `k` (order preserved). -/
def set (o : JObj) (k : String) (v : JVal) : JObj := o.map fun p => if p.1 = k then (k, v) else p

/-- `d[k]` after the exact-field check (all looked-up keys are present). -/
def field (o : JObj) (k : String) : JVal := (JObj.get o k).getD .null

end JObj

/-- Python `x == "lit"` (only a string equals a string). -/
def JVal.isStr : JVal → String → Bool
  | .str t, s => t == s
  | _, _ => false

/-- Python `x == n` for an `int` constant: `True == 1` and `1.0 == 1` hold in Python. -/
def JVal.eqInt : JVal → Int → Bool
  | .bool b, n => (if b then 1 else 0) == n
  | .int k, n => k == n
  | .num q, n => q == (n : ℚ)
  | _, _ => false

/-- `isinstance(x, list) and len(x) == n`. -/
def JVal.isListOfLen : JVal → ℕ → Bool
  | .arr xs, n => xs.length == n
  | _, _ => false

def SCHEME : String := "WaveLock-OTS-v1"
def HASH_ALG : String := "SHAKE256-256"
def N_BITS : ℕ := 256

def PUBLIC_KEY_FIELDS : List String :=
  ["scheme", "version", "hash_alg", "params_hash", "psi_commitment", "one_time_key_id",
   "pk_commitments", "merkle_root", "public_key_fingerprint"]

def SIGNATURE_FIELDS : List String :=
  ["scheme", "version", "hash_alg", "one_time_key_id", "public_key_fingerprint", "params_hash",
   "psi_commitment", "message_digest", "revealed_slices"]

/-- `_canonical_public_key_payload`: every public-key field except the fingerprint. -/
def PAYLOAD_FIELDS : List String :=
  ["scheme", "version", "hash_alg", "params_hash", "psi_commitment", "one_time_key_id",
   "pk_commitments", "merkle_root"]

/-- CPython facts used by the verifier. -/
structure PyEnv where
  /-- `str(x)` -/
  render : JVal → String
  render_str : ∀ s, render (.str s) = s
  /-- `x == y` on JSON values -/
  eq : JVal → JVal → Bool
  eq_refl : ∀ x, eq x x = true
  eq_str : ∀ s x, eq (.str s) x = true ↔ x = .str s

/-- Hash-derived functions of WaveLock-OTS, all opaque (SHAKE256-256 with length-prefixed
domain separation in the implementation). -/
structure OTSCrypto (Msg : Type) where
  /-- `public_key_fingerprint`: hex of `H(FP-domain, canonical_json(payload))` -/
  fingerprint : JVal → String
  /-- `_merkle_root_from_commitments(pk_commitments).hex()`; `none` when it raises -/
  merkleRoot : JVal → Option String
  /-- `bytes.fromhex(str(params_hash))` succeeds -/
  paramsOK : String → Bool
  /-- `_message_digest(message, p_hash).hex()`; `none` when it raises -/
  digest : String → Msg → Option String
  /-- `len(revealed) == 256` and each revealed slice hashes to the committed half it selects -/
  slicesOK : String → String → JVal → JVal → Bool
  /-- `bytes.fromhex` of every half of a 256-row table of `[half0, half1]` rows; `none` when a
  row or half is malformed.  Bytes are lists of byte values. -/
  decodeTable : JVal → Option (List (List ℕ × List ℕ))
  /-- `_merkle_root(leaves).hex()` over the decoded leaves (domain-separated SHAKE256) -/
  rootOf : List (List ℕ × List ℕ) → String
  /-- `_merkle_root_from_commitments` decodes the table, then hashes the decoded bytes -/
  merkleRoot_eq : ∀ x, merkleRoot x = (decodeTable x).map rootOf

section Verify
variable {Msg : Type} (E : PyEnv) (C : OTSCrypto Msg)

/-- The fingerprint-covered payload object. -/
def payloadOf (pk : JObj) : JVal := .obj (PAYLOAD_FIELDS.map fun k => (k, JObj.field pk k))

/-- `_check_public_key_canonical`. -/
def pkCanonical (pk : JObj) : Bool :=
  JObj.hasExactly pk PUBLIC_KEY_FIELDS &&
  (JObj.field pk "scheme").isStr SCHEME &&
  (JObj.field pk "version").eqInt 1 &&
  (JObj.field pk "hash_alg").isStr HASH_ALG &&
  (JObj.field pk "pk_commitments").isListOfLen N_BITS &&
  C.merkleRoot (JObj.field pk "pk_commitments") == some (E.render (JObj.field pk "merkle_root")) &&
  C.fingerprint (payloadOf pk) == E.render (JObj.field pk "public_key_fingerprint")

/-- The message-dependent tail of `verify_ots` (digest recomputation and the slice loop). -/
def msgOK (pk sig : JObj) (m : Msg) : Bool :=
  match C.digest (E.render (JObj.field pk "params_hash")) m with
  | some d => E.render (JObj.field sig "message_digest") == d &&
      C.slicesOK (E.render (JObj.field pk "params_hash")) d (JObj.field sig "revealed_slices")
        (JObj.field pk "pk_commitments")
  | none => false

/-- The signature-side checks of `verify_ots`, given a canonical public key. -/
def sigChecks (pk sig : JObj) : Bool :=
  C.paramsOK (E.render (JObj.field pk "params_hash")) &&
  JObj.hasExactly sig SIGNATURE_FIELDS &&
  (JObj.field sig "scheme").isStr SCHEME &&
  (JObj.field sig "version").eqInt 1 &&
  (JObj.field sig "hash_alg").isStr HASH_ALG &&
  E.eq (JObj.field sig "one_time_key_id") (JObj.field pk "one_time_key_id") &&
  E.render (JObj.field sig "public_key_fingerprint") == C.fingerprint (payloadOf pk) &&
  E.render (JObj.field sig "params_hash") == E.render (JObj.field pk "params_hash") &&
  E.render (JObj.field sig "psi_commitment") == E.render (JObj.field pk "psi_commitment")

/-- `verify_ots(public_key, message, signature)`. -/
def verifyOTS (pkv : JVal) (m : Msg) (sigv : JVal) : Bool :=
  match pkv, sigv with
  | .obj pk, .obj sig => pkCanonical E C pk && sigChecks E C pk sig && msgOK E C pk sig m
  | _, _ => false

end Verify

/-! ## Fail-closed characterisation -/

section Theorems
variable {Msg : Type} (E : PyEnv) (C : OTSCrypto Msg)

/-- **Fail-closed verification.**  Acceptance implies every check of `verify_ots`; equivalently,
each listed malformation is rejected: a non-object key or signature; missing or additional fields
on either object; a scheme, version or hash algorithm differing from the constants; a malformed
commitment table; a Merkle root that does not recompute; a public-key fingerprint that does not
recompute; a signature naming another key id, fingerprint, parameter set or ψ-commitment; an
unparseable parameter hash; a carried digest differing from the recomputed one; any slice that
does not open its commitment. -/
theorem verifyOTS_sound {pkv sigv : JVal} {m : Msg} (h : verifyOTS E C pkv m sigv = true) :
    ∃ pk sig : JObj, pkv = .obj pk ∧ sigv = .obj sig ∧
      JObj.hasExactly pk PUBLIC_KEY_FIELDS = true ∧ JObj.hasExactly sig SIGNATURE_FIELDS = true ∧
      (JObj.field pk "scheme").isStr SCHEME = true ∧ (JObj.field sig "scheme").isStr SCHEME = true ∧
      (JObj.field pk "version").eqInt 1 = true ∧ (JObj.field sig "version").eqInt 1 = true ∧
      (JObj.field pk "hash_alg").isStr HASH_ALG = true ∧ (JObj.field sig "hash_alg").isStr HASH_ALG = true ∧
      (JObj.field pk "pk_commitments").isListOfLen N_BITS = true ∧
      C.merkleRoot (JObj.field pk "pk_commitments") = some (E.render (JObj.field pk "merkle_root")) ∧
      C.fingerprint (payloadOf pk) = E.render (JObj.field pk "public_key_fingerprint") ∧
      C.paramsOK (E.render (JObj.field pk "params_hash")) = true ∧
      E.eq (JObj.field sig "one_time_key_id") (JObj.field pk "one_time_key_id") = true ∧
      E.render (JObj.field sig "public_key_fingerprint") = C.fingerprint (payloadOf pk) ∧
      E.render (JObj.field sig "params_hash") = E.render (JObj.field pk "params_hash") ∧
      E.render (JObj.field sig "psi_commitment") = E.render (JObj.field pk "psi_commitment") ∧
      ∃ d, C.digest (E.render (JObj.field pk "params_hash")) m = some d ∧
        E.render (JObj.field sig "message_digest") = d ∧
        C.slicesOK (E.render (JObj.field pk "params_hash")) d (JObj.field sig "revealed_slices")
          (JObj.field pk "pk_commitments") = true := by
  cases pkv <;> cases sigv <;> simp only [verifyOTS, Bool.false_eq_true] at h
  rename_i pk sig
  simp only [pkCanonical, sigChecks, msgOK, Bool.and_eq_true, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨hfields, hscheme⟩, hver⟩, halg⟩, hshape⟩, hroot⟩, hfp⟩,
    ⟨⟨⟨⟨⟨⟨⟨⟨hparams, hsfields⟩, hsscheme⟩, hsver⟩, hsalg⟩, hkid⟩, hsfp⟩, hsph⟩, hspsi⟩⟩, hmsg⟩ := h
  refine ⟨pk, sig, rfl, rfl, hfields, hsfields, hscheme, hsscheme, hver, hsver, halg, hsalg, hshape,
    hroot, hfp, hparams, hkid, hsfp, hsph, hspsi, ?_⟩
  revert hmsg
  cases hd : C.digest (E.render (JObj.field pk "params_hash")) m with
  | none => simp
  | some d =>
    intro hmsg
    simp only [Bool.and_eq_true, beq_iff_eq] at hmsg
    exact ⟨d, rfl, hmsg.1, hmsg.2⟩

/-- **Identity binding.**  The replay identities carried by an accepted signature are those of
the verified public key: a string `one_time_key_id` equals the key's, and the carried fingerprint
is the fingerprint recomputed from the key's canonical payload. -/
theorem verifyOTS_identity {pk sig : JObj} {m : Msg} {s : String}
    (h : verifyOTS E C (.obj pk) m (.obj sig) = true) (hs : JObj.field sig "one_time_key_id" = .str s) :
    JObj.field pk "one_time_key_id" = .str s ∧
      E.render (JObj.field sig "public_key_fingerprint") = C.fingerprint (payloadOf pk) ∧
      E.render (JObj.field pk "public_key_fingerprint") = C.fingerprint (payloadOf pk) := by
  obtain ⟨pk', sig', hpk, hsig, -, -, -, -, -, -, -, -, -, -, hfp, -, hkid, hsfp, -⟩ :=
    verifyOTS_sound E C h
  cases hpk; cases hsig
  rw [hs] at hkid
  exact ⟨(E.eq_str s _).1 hkid, hsfp, hfp.symm⟩

/-- **Fingerprint binding, as collision extraction.**  Two accepted signatures carrying the same
fingerprint string either verify against public keys with the same canonical payload, or exhibit
an explicit collision of the fingerprint function.  (Payload equality fixes the key id,
parameters, ψ-commitment, commitment table and Merkle root.) -/
theorem fingerprint_binding {pk₁ sig₁ pk₂ sig₂ : JObj} {m₁ m₂ : Msg}
    (h₁ : verifyOTS E C (.obj pk₁) m₁ (.obj sig₁) = true)
    (h₂ : verifyOTS E C (.obj pk₂) m₂ (.obj sig₂) = true)
    (hfp : E.render (JObj.field sig₁ "public_key_fingerprint") =
      E.render (JObj.field sig₂ "public_key_fingerprint")) :
    payloadOf pk₁ = payloadOf pk₂ ∨ Collision C.fingerprint (payloadOf pk₁) (payloadOf pk₂) := by
  obtain ⟨_, _, hpk₁, hsig₁, -, -, -, -, -, -, -, -, -, -, -, -, -, hsfp₁, -⟩ := verifyOTS_sound E C h₁
  obtain ⟨_, _, hpk₂, hsig₂, -, -, -, -, -, -, -, -, -, -, -, -, -, hsfp₂, -⟩ := verifyOTS_sound E C h₂
  cases hpk₁; cases hsig₁; cases hpk₂; cases hsig₂
  by_cases heq : payloadOf pk₁ = payloadOf pk₂
  · exact Or.inl heq
  · exact Or.inr ⟨heq, by rw [← hsfp₁, ← hsfp₂, hfp]⟩

/-! ## The key id is a free label -/

theorem JObj.keys_set (o : JObj) (k : String) (v : JVal) :
    JObj.keys (JObj.set o k v) = JObj.keys o := by
  induction o with
  | nil => rfl
  | cons p o ih =>
    simp only [JObj.set, JObj.keys, List.map_cons, List.map_map] at ih ⊢
    split_ifs with h
    · simp [h, ih]
    · simp [ih]

theorem JObj.get_set_ne (o : JObj) {k k' : String} (v : JVal) (h : k' ≠ k) :
    JObj.get (JObj.set o k v) k' = JObj.get o k' := by
  induction o with
  | nil => rfl
  | cons p o ih =>
    obtain ⟨a, x⟩ := p
    simp only [JObj.set, JObj.get, List.map_cons] at ih ⊢
    have hk : (k' == k) = false := by simpa using h
    by_cases h1 : a = k
    · subst h1
      simp only [if_true, List.lookup, hk]
      exact ih
    · simp only [h1, if_false, List.lookup]
      split <;> simp_all

theorem JObj.get_set_self (o : JObj) {k : String} (v : JVal) (hk : k ∈ JObj.keys o) :
    JObj.get (JObj.set o k v) k = some v := by
  induction o with
  | nil => simp [JObj.keys] at hk
  | cons p o ih =>
    obtain ⟨a, x⟩ := p
    simp only [JObj.set, JObj.get, List.map_cons] at ih ⊢
    by_cases h1 : a = k
    · subst h1; simp [List.lookup]
    · have hk' : k ∈ JObj.keys o := by
        simp only [JObj.keys, List.map_cons, List.mem_cons] at hk
        rcases hk with hk | hk
        · exact absurd hk.symm h1
        · exact hk
      have hne : (k == a) = false := by simpa using Ne.symm h1
      simp only [h1, if_false, List.lookup, hne]
      exact ih hk'

theorem JObj.field_set_ne (o : JObj) {k k' : String} (v : JVal) (h : k' ≠ k) :
    JObj.field (JObj.set o k v) k' = JObj.field o k' := by
  simp [JObj.field, JObj.get_set_ne o v h]

theorem JObj.field_set_self (o : JObj) {k : String} (v : JVal) (hk : k ∈ JObj.keys o) :
    JObj.field (JObj.set o k v) k = v := by
  simp [JObj.field, JObj.get_set_self o v hk]

theorem JObj.hasExactly_set (o : JObj) (k : String) (v : JVal) (fs : List String) :
    JObj.hasExactly (JObj.set o k v) fs = JObj.hasExactly o fs := by
  simp [JObj.hasExactly, JObj.keys_set]

theorem mem_keys_of_hasExactly {o : JObj} {fs : List String} (h : JObj.hasExactly o fs = true)
    {k : String} (hk : k ∈ fs) : k ∈ JObj.keys o := by
  simp only [JObj.hasExactly, Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at h
  exact h.2 k hk

/-- The fingerprint that recomputes after relabelling `pk` with key id `k`. -/
def relabelFP (C : OTSCrypto Msg) (pk : JObj) (k : JVal) : String :=
  C.fingerprint (payloadOf (JObj.set pk "one_time_key_id" k))

/-- Relabel a public key with a new key id and the fingerprint that recomputes. -/
def relabelPK (C : OTSCrypto Msg) (pk : JObj) (k : JVal) : JObj :=
  JObj.set (JObj.set pk "one_time_key_id" k) "public_key_fingerprint" (.str (relabelFP C pk k))

/-- Relabel a signature consistently (same message digest, same revealed slices). -/
def relabelSig (C : OTSCrypto Msg) (pk sig : JObj) (k : JVal) : JObj :=
  JObj.set (JObj.set sig "one_time_key_id" k) "public_key_fingerprint" (.str (relabelFP C pk k))

theorem payloadOf_relabel (pk : JObj) (k : JVal) :
    payloadOf (relabelPK C pk k) = payloadOf (JObj.set pk "one_time_key_id" k) := by
  simp only [payloadOf, relabelPK]
  congr 1
  apply List.map_congr_left
  intro f hf
  have : f ≠ "public_key_fingerprint" := by
    intro h; subst h; simp [PAYLOAD_FIELDS] at hf
  rw [JObj.field_set_ne _ _ this]

theorem field_relabel_other (o : JObj) (k v : JVal) {f : String} (h1 : f ≠ "one_time_key_id")
    (h2 : f ≠ "public_key_fingerprint") :
    JObj.field (JObj.set (JObj.set o "one_time_key_id" k) "public_key_fingerprint" v) f =
      JObj.field o f := by
  rw [JObj.field_set_ne _ _ h2, JObj.field_set_ne _ _ h1]

/-- **The key id is not bound to the key material.**  From any accepted `(pk, m, sig)` *anyone*
obtains another accepted triple for the same message, with the same revealed slices and the same
commitment table, under an arbitrary new `one_time_key_id` and the fingerprint that recomputes.
The pure verifier therefore cannot distinguish a fresh one-time key from a relabelled used one;
on the block path the transcript binds the carried public key (see `AuthenticatedBlockBinding`),
so there only the holder of the secret seed can mint such relabellings. -/
theorem kid_relabel_verifies {pk sig : JObj} {m : Msg} (k : JVal)
    (h : verifyOTS E C (.obj pk) m (.obj sig) = true) :
    verifyOTS E C (.obj (relabelPK C pk k)) m (.obj (relabelSig C pk sig k)) = true := by
  obtain ⟨_, _, hpk, hsig, hfields, hsfields, hscheme, hsscheme, hver, hsver, halg, hsalg, hshape,
    hroot, -, hparams, -, -, hsph, hspsi, d, hd, hmd, hsl⟩ := verifyOTS_sound E C h
  cases hpk; cases hsig
  have hkpk : "one_time_key_id" ∈ JObj.keys pk :=
    mem_keys_of_hasExactly hfields (by simp [PUBLIC_KEY_FIELDS])
  have hfpk : "public_key_fingerprint" ∈ JObj.keys (JObj.set pk "one_time_key_id" k) := by
    rw [JObj.keys_set]; exact mem_keys_of_hasExactly hfields (by simp [PUBLIC_KEY_FIELDS])
  have hksig : "one_time_key_id" ∈ JObj.keys sig :=
    mem_keys_of_hasExactly hsfields (by simp [SIGNATURE_FIELDS])
  have hfsig : "public_key_fingerprint" ∈ JObj.keys (JObj.set sig "one_time_key_id" k) := by
    rw [JObj.keys_set]; exact mem_keys_of_hasExactly hsfields (by simp [SIGNATURE_FIELDS])
  have hpkid : JObj.field (relabelPK C pk k) "one_time_key_id" = k := by
    simp only [relabelPK]; rw [JObj.field_set_ne _ _ (by decide)]; exact JObj.field_set_self _ _ hkpk
  have hsigid : JObj.field (relabelSig C pk sig k) "one_time_key_id" = k := by
    simp only [relabelSig]; rw [JObj.field_set_ne _ _ (by decide)]; exact JObj.field_set_self _ _ hksig
  have hpkfp : JObj.field (relabelPK C pk k) "public_key_fingerprint" = .str (relabelFP C pk k) := by
    simp only [relabelPK]; exact JObj.field_set_self _ _ hfpk
  have hsigfp : JObj.field (relabelSig C pk sig k) "public_key_fingerprint" =
      .str (relabelFP C pk k) := by
    simp only [relabelSig]; exact JObj.field_set_self _ _ hfsig
  have hpl : payloadOf (relabelPK C pk k) = payloadOf (JObj.set pk "one_time_key_id" k) :=
    payloadOf_relabel C pk k
  have hexpk : JObj.hasExactly (relabelPK C pk k) PUBLIC_KEY_FIELDS = true := by
    simp only [relabelPK, JObj.hasExactly_set]; exact hfields
  have hexsig : JObj.hasExactly (relabelSig C pk sig k) SIGNATURE_FIELDS = true := by
    simp only [relabelSig, JObj.hasExactly_set]; exact hsfields
  have opk : ∀ f, f ≠ "one_time_key_id" → f ≠ "public_key_fingerprint" →
      JObj.field (relabelPK C pk k) f = JObj.field pk f := fun f h1 h2 =>
    field_relabel_other pk k _ h1 h2
  have osig : ∀ f, f ≠ "one_time_key_id" → f ≠ "public_key_fingerprint" →
      JObj.field (relabelSig C pk sig k) f = JObj.field sig f := fun f h1 h2 =>
    field_relabel_other sig k _ h1 h2
  simp only [verifyOTS, pkCanonical, sigChecks, msgOK, Bool.and_eq_true, beq_iff_eq,
    hexpk, hexsig, hpkid, hsigid, hpkfp, hsigfp, hpl, E.render_str, E.eq_refl,
    opk "scheme" (by decide) (by decide), opk "version" (by decide) (by decide),
    opk "hash_alg" (by decide) (by decide), opk "pk_commitments" (by decide) (by decide),
    opk "merkle_root" (by decide) (by decide), opk "params_hash" (by decide) (by decide),
    opk "psi_commitment" (by decide) (by decide),
    osig "scheme" (by decide) (by decide), osig "version" (by decide) (by decide),
    osig "hash_alg" (by decide) (by decide), osig "params_hash" (by decide) (by decide),
    osig "psi_commitment" (by decide) (by decide), osig "message_digest" (by decide) (by decide),
    osig "revealed_slices" (by decide) (by decide), hd, relabelFP]
  simp [hscheme, hver, halg, hshape, hroot, hparams, hsscheme, hsver, hsalg, hsph, hspsi, hmd, hsl]

/-- **Canonicalisation finding.**  The version checks are Python numeric equality, so the
non-canonical encodings `true` and `1.0` of version 1 pass them (reproduced against the
implementation in `tests/test_ots_formal_conformance.py`).  Each such encoding is a distinct
public key (the fingerprint covers it) and changes no replay decision, since the key id is
checked separately. -/
theorem version_check_accepts_noncanonical :
    (JVal.bool true).eqInt 1 = true ∧ (JVal.num 1).eqInt 1 = true ∧ (JVal.str "1").eqInt 1 = false := by
  decide


/-! ## The ψ-commitment is a free label too -/

theorem field_set_set_other (o : JObj) {k₁ k₂ f : String} (v₁ v₂ : JVal) (h1 : f ≠ k₁)
    (h2 : f ≠ k₂) : JObj.field (JObj.set (JObj.set o k₁ v₁) k₂ v₂) f = JObj.field o f := by
  rw [JObj.field_set_ne _ _ h2, JObj.field_set_ne _ _ h1]

/-- The fingerprint that recomputes after relabelling `pk` with ψ-commitment `v`. -/
def relabelPsiFP (C : OTSCrypto Msg) (pk : JObj) (v : JVal) : String :=
  C.fingerprint (payloadOf (JObj.set pk "psi_commitment" v))

def relabelPsiPK (C : OTSCrypto Msg) (pk : JObj) (v : JVal) : JObj :=
  JObj.set (JObj.set pk "psi_commitment" v) "public_key_fingerprint" (.str (relabelPsiFP C pk v))

def relabelPsiSig (C : OTSCrypto Msg) (pk sig : JObj) (v : JVal) : JObj :=
  JObj.set (JObj.set sig "psi_commitment" v) "public_key_fingerprint" (.str (relabelPsiFP C pk v))

/-- **The ψ-commitment is not bound to the key material either.**  `verify_ots` only compares the
signature's ψ-commitment with the key's; the slices are opened under `params_hash`.  Any accepted
`(pk, m, sig)` relabelled with an arbitrary ψ-commitment (and the fingerprint that recomputes)
verifies, with the same revealed slices.  So an identifier that hashes the ψ-commitment
(Candidate C of the material analysis) can be refreshed by relabelling. -/
theorem psi_relabel_verifies {pk sig : JObj} {m : Msg} (v : JVal)
    (h : verifyOTS E C (.obj pk) m (.obj sig) = true) :
    verifyOTS E C (.obj (relabelPsiPK C pk v)) m (.obj (relabelPsiSig C pk sig v)) = true := by
  obtain ⟨_, _, hpk, hsig, hfields, hsfields, hscheme, hsscheme, hver, hsver, halg, hsalg, hshape,
    hroot, -, hparams, hkid, -, hsph, -, d, hd, hmd, hsl⟩ := verifyOTS_sound E C h
  cases hpk; cases hsig
  have hppk : "psi_commitment" ∈ JObj.keys pk :=
    mem_keys_of_hasExactly hfields (by simp [PUBLIC_KEY_FIELDS])
  have hfpk : "public_key_fingerprint" ∈ JObj.keys (JObj.set pk "psi_commitment" v) := by
    rw [JObj.keys_set]; exact mem_keys_of_hasExactly hfields (by simp [PUBLIC_KEY_FIELDS])
  have hpsig : "psi_commitment" ∈ JObj.keys sig :=
    mem_keys_of_hasExactly hsfields (by simp [SIGNATURE_FIELDS])
  have hfsig : "public_key_fingerprint" ∈ JObj.keys (JObj.set sig "psi_commitment" v) := by
    rw [JObj.keys_set]; exact mem_keys_of_hasExactly hsfields (by simp [SIGNATURE_FIELDS])
  have hpkpsi : JObj.field (relabelPsiPK C pk v) "psi_commitment" = v := by
    simp only [relabelPsiPK]; rw [JObj.field_set_ne _ _ (by decide)]; exact JObj.field_set_self _ _ hppk
  have hsigpsi : JObj.field (relabelPsiSig C pk sig v) "psi_commitment" = v := by
    simp only [relabelPsiSig]; rw [JObj.field_set_ne _ _ (by decide)]
    exact JObj.field_set_self _ _ hpsig
  have hpkfp : JObj.field (relabelPsiPK C pk v) "public_key_fingerprint" =
      .str (relabelPsiFP C pk v) := by
    simp only [relabelPsiPK]; exact JObj.field_set_self _ _ hfpk
  have hsigfp : JObj.field (relabelPsiSig C pk sig v) "public_key_fingerprint" =
      .str (relabelPsiFP C pk v) := by
    simp only [relabelPsiSig]; exact JObj.field_set_self _ _ hfsig
  have hpl : payloadOf (relabelPsiPK C pk v) = payloadOf (JObj.set pk "psi_commitment" v) := by
    simp only [payloadOf, relabelPsiPK]
    congr 1
    apply List.map_congr_left
    intro f hf
    have : f ≠ "public_key_fingerprint" := by
      intro h; subst h; simp [PAYLOAD_FIELDS] at hf
    rw [JObj.field_set_ne _ _ this]
  have hexpk : JObj.hasExactly (relabelPsiPK C pk v) PUBLIC_KEY_FIELDS = true := by
    simp only [relabelPsiPK, JObj.hasExactly_set]; exact hfields
  have hexsig : JObj.hasExactly (relabelPsiSig C pk sig v) SIGNATURE_FIELDS = true := by
    simp only [relabelPsiSig, JObj.hasExactly_set]; exact hsfields
  have opk : ∀ f, f ≠ "psi_commitment" → f ≠ "public_key_fingerprint" →
      JObj.field (relabelPsiPK C pk v) f = JObj.field pk f := fun f h1 h2 =>
    field_set_set_other pk _ _ h1 h2
  have osig : ∀ f, f ≠ "psi_commitment" → f ≠ "public_key_fingerprint" →
      JObj.field (relabelPsiSig C pk sig v) f = JObj.field sig f := fun f h1 h2 =>
    field_set_set_other sig _ _ h1 h2
  simp only [verifyOTS, pkCanonical, sigChecks, msgOK, Bool.and_eq_true, beq_iff_eq,
    hexpk, hexsig, hpkpsi, hsigpsi, hpkfp, hsigfp, hpl, E.render_str,
    opk "scheme" (by decide) (by decide), opk "version" (by decide) (by decide),
    opk "hash_alg" (by decide) (by decide), opk "pk_commitments" (by decide) (by decide),
    opk "merkle_root" (by decide) (by decide), opk "params_hash" (by decide) (by decide),
    opk "one_time_key_id" (by decide) (by decide),
    osig "scheme" (by decide) (by decide), osig "version" (by decide) (by decide),
    osig "hash_alg" (by decide) (by decide), osig "params_hash" (by decide) (by decide),
    osig "one_time_key_id" (by decide) (by decide), osig "message_digest" (by decide) (by decide),
    osig "revealed_slices" (by decide) (by decide), hd, relabelPsiFP]
  simp [hscheme, hver, halg, hshape, hroot, hparams, hsscheme, hsver, hsalg, hkid, hsph, hmd, hsl]

/-! ## Key-material identity: the commitment table's Merkle root -/

/-- `ots_material_id(public_key)`: the Merkle root recomputed from the commitment table (not the
block's message `merkle_root`). -/
def materialId (C : OTSCrypto Msg) (pk : JObj) : Option String :=
  C.merkleRoot (JObj.field pk "pk_commitments")

/-- For a verified key the material id is the key's verified `merkle_root` field, i.e. what
`ots_material_id` returns. -/
theorem materialId_of_verified {pk sig : JObj} {m : Msg}
    (h : verifyOTS E C (.obj pk) m (.obj sig) = true) :
    materialId C pk = some (E.render (JObj.field pk "merkle_root")) := by
  obtain ⟨_, _, hpk, _, -, -, -, -, -, -, -, -, -, hroot, -⟩ := verifyOTS_sound E C h
  cases hpk; exact hroot

/-- **Equality preservation (no assumption).**  Keys whose commitment tables decode to the same
bytes have the same material id, whatever their key ids, ψ-commitments, fingerprints or hex
spellings. -/
theorem materialId_eq_of_decode_eq {pk₁ pk₂ : JObj}
    (h : C.decodeTable (JObj.field pk₁ "pk_commitments") =
      C.decodeTable (JObj.field pk₂ "pk_commitments")) :
    materialId C pk₁ = materialId C pk₂ := by
  simp only [materialId, C.merkleRoot_eq, h]

/-- Relabelling the key id does not change the material id. -/
theorem materialId_relabel_kid (pk : JObj) (k : JVal) :
    materialId C (relabelPK C pk k) = materialId C pk := by
  simp only [materialId, relabelPK]; rw [field_relabel_other pk k _ (by decide) (by decide)]

/-- Relabelling the ψ-commitment does not change the material id. -/
theorem materialId_relabel_psi (pk : JObj) (v : JVal) :
    materialId C (relabelPsiPK C pk v) = materialId C pk := by
  simp only [materialId, relabelPsiPK]; rw [field_set_set_other pk _ _ (by decide) (by decide)]

/-- **A relabelled replay carries the consumed material.**  Every relabelling the verifier
admits — key id (`kid_relabel_verifies`) or ψ-commitment (`psi_relabel_verifies`) — keeps the
material id, so a ledger that consumes it rejects the relabelled signature. -/
theorem relabelled_replays_keep_material {pk sig : JObj} {m : Msg} (k v : JVal)
    (h : verifyOTS E C (.obj pk) m (.obj sig) = true) :
    verifyOTS E C (.obj (relabelPK C pk k)) m (.obj (relabelSig C pk sig k)) = true ∧
      materialId C (relabelPK C pk k) = materialId C pk ∧
      verifyOTS E C (.obj (relabelPsiPK C pk v)) m (.obj (relabelPsiSig C pk sig v)) = true ∧
      materialId C (relabelPsiPK C pk v) = materialId C pk :=
  ⟨kid_relabel_verifies E C k h, materialId_relabel_kid C pk k, psi_relabel_verifies E C v h,
    materialId_relabel_psi C pk v⟩

/-- **Converse, as collision extraction.**  Equal material ids of two keys whose tables decode
mean equal decoded tables, or an explicit collision of the commitment Merkle root. -/
theorem material_binding {pk₁ pk₂ : JObj} {t₁ t₂ : List (List ℕ × List ℕ)}
    (h₁ : C.decodeTable (JObj.field pk₁ "pk_commitments") = some t₁)
    (h₂ : C.decodeTable (JObj.field pk₂ "pk_commitments") = some t₂)
    (h : materialId C pk₁ = materialId C pk₂) : t₁ = t₂ ∨ Collision C.rootOf t₁ t₂ := by
  simp only [materialId, C.merkleRoot_eq, h₁, h₂, Option.map_some, Option.some.injEq] at h
  by_cases heq : t₁ = t₂
  · exact Or.inl heq
  · exact Or.inr ⟨heq, h⟩

/-- **Material single use implies table single use.**  If the material ids of a list of keys are
pairwise distinct (as `OTSKeyConsumption.wavelock_ots_material_single_use` guarantees for the
keys of an accepted chain), their decoded commitment tables are pairwise distinct: no table
authorises two accepted blocks.  No cryptographic assumption is used in this direction. -/
theorem distinct_material_distinct_tables {ι : Type} (key : ι → JObj) (l : List ι)
    (h : (l.map fun i => materialId C (key i)).Nodup) :
    (l.map fun i => C.decodeTable (JObj.field (key i) "pk_commitments")).Nodup := by
  have heq : (l.map fun i => materialId C (key i)) =
      (l.map fun i => C.decodeTable (JObj.field (key i) "pk_commitments")).map
        (Option.map C.rootOf) := by
    simp [materialId, C.merkleRoot_eq, Function.comp_def]
  rw [heq] at h
  exact List.Nodup.of_map (Option.map C.rootOf) h

end Theorems

end OTS
end WaveLockProofs
