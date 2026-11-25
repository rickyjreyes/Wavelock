import WaveLockProofs.OTSVerification

/-!
# Authenticated block binding (`verify_ots_block`, transcript version 1)

Transcription of `wavelock/chain/ots_blocks.py` at `b310ea5`:
`_ots_block_transcript_payload`, `canonical_ots_block_digest`
(`sha256("WL-OTS-BLOCK-TRANSCRIPT-v1\0" ‖ canonical_json(payload))`) and `verify_ots_block`.
The digest and `_canonical_json` are opaque parameters.

Results:
* the OTS signature of an accepted block verifies against the digest *recomputed from the
  received block*, and the carried `message` must equal it (Mythos M1 fix);
* two accepted blocks carrying the same authentication material have identical transcripts —
  hence identical body, type, parent, metadata, public key and signed header context — unless the
  adversary exhibits an explicit collision of the transcript digest;
* a third party who relabels the carried public key (e.g. a fresh `one_time_key_id`) obtains an
  accepted block only by a transcript-digest collision or by producing a signature on a second
  message under the same commitment table, i.e. by breaking one-time unforgeability;
* the pre-M1 verifier was independent of the block body.

Cryptographic assumptions are never hypotheses of these theorems; each theorem names the event
(`Collision`, `TableUsedTwice`) whose infeasibility a cryptographic assumption would provide.
-/

namespace WaveLockProofs
namespace OTS

/-- The fields of `wavelock.chain.Block` read by OTS verification. -/
structure WBlock where
  headerVersion : ℕ
  index : ℕ
  timestamp : String
  difficulty : ℕ
  previousHash : JVal
  messages : List String
  blockType : String
  /-- `Block.metadata` -/
  metadata : JObj

/-- Block-layer hashing and serialisation, opaque. -/
structure BlockCrypto where
  /-- `canonical_ots_block_digest`: the hex string an OTS signer signs -/
  digest : JVal → String
  /-- `_canonical_json` -/
  canon : JVal → String

/-- Strip the self-referential `signature`/`message` from `meta.ots_auth`. -/
def stripAuthEntry (p : String × JVal) : String × JVal :=
  if p.1 = "ots_auth" then
    (p.1, match p.2 with
      | .obj a => .obj (a.filter fun q => q.1 ≠ "signature" ∧ q.1 ≠ "message")
      | v => v)
  else p

def stripAuth (md : JObj) : JObj := md.map stripAuthEntry

/-- `_ots_block_transcript_payload(block_type, previous_hash, messages, meta)`. -/
def transcriptPayload (b : WBlock) : JVal :=
  .obj [("wl_ots_block_transcript", .int 1),
        ("auth_scheme", JObj.field b.metadata "auth_scheme"),
        ("block_type", .str b.blockType),
        ("previous_hash", b.previousHash),
        ("messages", .arr (b.messages.map .str)),
        ("meta", .obj (stripAuth b.metadata))]

/-- The signed header context of header-version-2 producers. -/
def headerContext (b : WBlock) : JVal :=
  .obj [("version", .int b.headerVersion), ("index", .int b.index),
        ("timestamp", .str b.timestamp), ("difficulty", .int b.difficulty)]

section Verify
variable (E : PyEnv) (C : OTSCrypto String) (BC : BlockCrypto)

/-- `verify_ots_block(block)`. -/
def verifyOTSBlock (b : WBlock) : Bool :=
  match JObj.get b.metadata "ots_auth" with
  | some (.obj auth) =>
    (JObj.field b.metadata "auth_scheme").isStr SCHEME &&
    (if b.headerVersion = 2 ∨ "block_header" ∈ JObj.keys b.metadata then
      BC.canon (JObj.field b.metadata "block_header") == BC.canon (headerContext b) else true) &&
    (match JObj.get auth "message" with
      | some (.str msg) =>
        msg == BC.digest (transcriptPayload b) &&
        verifyOTS E C (JObj.field auth "public_key") (BC.digest (transcriptPayload b))
          (JObj.field auth "signature")
      | _ => false)
  | _ => false

/-- The verifier before the Mythos M1 fix: the signature was checked against the carried
`message`, never against the received body. -/
def verifyOTSBlockPreM1 (b : WBlock) : Bool :=
  match JObj.get b.metadata "ots_auth" with
  | some (.obj auth) =>
    (match JObj.get auth "message" with
      | some (.str msg) =>
        verifyOTS E C (JObj.field auth "public_key") msg (JObj.field auth "signature")
      | _ => false)
  | _ => false

end Verify

section Theorems
variable (E : PyEnv) (C : OTSCrypto String) (BC : BlockCrypto)

/-- **Binding to the received block.**  An accepted block carries a message equal to the digest
of its own transcript, and its OTS signature verifies against exactly that digest. -/
theorem verifyOTSBlock_binds {b : WBlock} (h : verifyOTSBlock E C BC b = true) :
    ∃ auth : JObj, JObj.get b.metadata "ots_auth" = some (.obj auth) ∧
      JObj.get auth "message" = some (.str (BC.digest (transcriptPayload b))) ∧
      (JObj.field b.metadata "auth_scheme").isStr SCHEME = true ∧
      verifyOTS E C (JObj.field auth "public_key") (BC.digest (transcriptPayload b))
        (JObj.field auth "signature") = true := by
  unfold verifyOTSBlock at h
  split at h
  · rename_i auth hauth
    simp only [Bool.and_eq_true] at h
    obtain ⟨⟨hscheme, _⟩, hmsg⟩ := h
    split at hmsg
    · rename_i msg hm
      simp only [Bool.and_eq_true, beq_iff_eq] at hmsg
      obtain ⟨rfl, hv⟩ := hmsg
      exact ⟨auth, hauth, hm, hscheme, hv⟩
    · simp at hmsg
  · simp at h

/-- The transcript payload determines body, type, parent and (stripped) metadata. -/
theorem transcriptPayload_injective {b₁ b₂ : WBlock}
    (h : transcriptPayload b₁ = transcriptPayload b₂) :
    b₁.messages = b₂.messages ∧ b₁.blockType = b₂.blockType ∧
      b₁.previousHash = b₂.previousHash ∧ stripAuth b₁.metadata = stripAuth b₂.metadata := by
  simp only [transcriptPayload, JVal.obj.injEq, List.cons.injEq, Prod.mk.injEq, JVal.str.injEq,
    JVal.arr.injEq, true_and, and_true] at h
  obtain ⟨_, hty, hprev, hmsgs, hmeta⟩ := h
  refine ⟨?_, hty, hprev, hmeta⟩
  exact List.map_injective_iff.2 (fun _ _ h => by injection h) hmsgs

/-- **Lifting authentication between blocks needs a digest collision.**  If two accepted blocks
carry the same `ots_auth` object (a signature copied from one block into another), their
transcripts — body, type, parent, metadata including the public key and signed header — are
equal, or the two transcripts collide under the transcript digest. -/
theorem block_binding {b₁ b₂ : WBlock} (h₁ : verifyOTSBlock E C BC b₁ = true)
    (h₂ : verifyOTSBlock E C BC b₂ = true)
    (hauth : JObj.get b₁.metadata "ots_auth" = JObj.get b₂.metadata "ots_auth") :
    transcriptPayload b₁ = transcriptPayload b₂ ∨
      Collision BC.digest (transcriptPayload b₁) (transcriptPayload b₂) := by
  obtain ⟨a₁, ha₁, hm₁, -, -⟩ := verifyOTSBlock_binds E C BC h₁
  obtain ⟨a₂, ha₂, hm₂, -, -⟩ := verifyOTSBlock_binds E C BC h₂
  rw [ha₁, ha₂] at hauth
  cases hauth
  rw [hm₁] at hm₂
  have hd : BC.digest (transcriptPayload b₁) = BC.digest (transcriptPayload b₂) := by
    simpa using hm₂
  by_cases heq : transcriptPayload b₁ = transcriptPayload b₂
  · exact Or.inl heq
  · exact Or.inr ⟨heq, hd⟩

/-- Under header version 2 the signed header context is the block's own header, provided the
canonical serialisation separates header contexts (true of `json.dumps` with fixed key order:
distinct integers and strings serialise distinctly). -/
theorem header_bound {b : WBlock} (h : verifyOTSBlock E C BC b = true) (hv : b.headerVersion = 2)
    (hcanon : ∀ x, BC.canon x = BC.canon (headerContext b) → x = headerContext b) :
    JObj.field b.metadata "block_header" = headerContext b := by
  unfold verifyOTSBlock at h
  split at h
  · simp only [Bool.and_eq_true] at h
    obtain ⟨⟨_, hh⟩, _⟩ := h
    simp only [hv, true_or, if_true, beq_iff_eq] at hh
    exact hcanon _ hh
  · simp at h

/-- Two different messages accepted under public keys sharing one commitment table: the
event excluded by one-time unforgeability for anyone who saw at most one signature of that
table, and caused by its holder when the key is reused. -/
def TableUsedTwice (pk₁ : JObj) (m₁ : String) (s₁ : JVal) (pk₂ : JObj) (m₂ : String) (s₂ : JVal) :
    Prop :=
  verifyOTS E C (.obj pk₁) m₁ s₁ = true ∧ verifyOTS E C (.obj pk₂) m₂ s₂ = true ∧ m₁ ≠ m₂ ∧
    JObj.field pk₁ "pk_commitments" = JObj.field pk₂ "pk_commitments"

/-- Relabelling a public key keeps its commitment table. -/
theorem relabel_keeps_table (pk : JObj) (k : JVal) :
    JObj.field (relabelPK C pk k) "pk_commitments" = JObj.field pk "pk_commitments" :=
  field_relabel_other pk k _ (by decide) (by decide)

/-- **Third-party relabelling on the block path.**  Let `b` be accepted under public key `pk`,
and let `b'` be any accepted block that carries a different transcript under a public key with the
same commitment table (for example `relabelPK C pk k`, a fresh key id spliced onto the same
material).  Then either the transcript digest collides, or the commitment table has been used to
sign two different messages.  Contrast `kid_relabel_verifies`: on the payload path the same
relabelling needs neither. -/
theorem block_relabel_needs_second_message {b b' : WBlock} {auth auth' pk pk' : JObj}
    (h : verifyOTSBlock E C BC b = true) (h' : verifyOTSBlock E C BC b' = true)
    (ha : JObj.get b.metadata "ots_auth" = some (.obj auth))
    (ha' : JObj.get b'.metadata "ots_auth" = some (.obj auth'))
    (hpk : JObj.field auth "public_key" = .obj pk) (hpk' : JObj.field auth' "public_key" = .obj pk')
    (htable : JObj.field pk "pk_commitments" = JObj.field pk' "pk_commitments")
    (hne : transcriptPayload b ≠ transcriptPayload b') :
    Collision BC.digest (transcriptPayload b) (transcriptPayload b') ∨
      TableUsedTwice E C pk (BC.digest (transcriptPayload b)) (JObj.field auth "signature")
        pk' (BC.digest (transcriptPayload b')) (JObj.field auth' "signature") := by
  obtain ⟨a, ha₁, -, -, hv⟩ := verifyOTSBlock_binds E C BC h
  obtain ⟨a', ha₁', -, -, hv'⟩ := verifyOTSBlock_binds E C BC h'
  rw [ha] at ha₁; rw [ha'] at ha₁'
  cases ha₁; cases ha₁'
  rw [hpk] at hv; rw [hpk'] at hv'
  by_cases hd : BC.digest (transcriptPayload b) = BC.digest (transcriptPayload b')
  · exact Or.inl ⟨hne, hd⟩
  · exact Or.inr ⟨hv, hv', hd, htable⟩

/-- Replacing the body of a block. -/
def withMessages (b : WBlock) (xs : List String) : WBlock := { b with messages := xs }

/-- **Historical (Mythos M1, fixed).**  The pre-M1 verifier ignored the block body: any
authentication material that verified for one body verified for every body. -/
theorem preM1_body_independent (b : WBlock) (xs ys : List String) :
    verifyOTSBlockPreM1 E C (withMessages b xs) = verifyOTSBlockPreM1 E C (withMessages b ys) := by
  rfl

/-- The current verifier is not: moving authentication material to another body requires a
transcript-digest collision. -/
theorem body_change_needs_collision {b : WBlock} {ys : List String}
    (h₁ : verifyOTSBlock E C BC b = true) (h₂ : verifyOTSBlock E C BC (withMessages b ys) = true) :
    b.messages = ys ∨ Collision BC.digest (transcriptPayload b) (transcriptPayload (withMessages b ys)) := by
  rcases block_binding E C BC h₁ h₂ rfl with heq | hcol
  · exact Or.inl (transcriptPayload_injective heq).1
  · exact Or.inr hcol

end Theorems

end OTS
end WaveLockProofs
