# WaveLock-OTS — Red-Team Audit

> ## STATUS (post-remediation)
>
> This report is preserved as the original audit. The findings have since been
> addressed as follows (see `docs/WAVELOCK_OTS_DESIGN.md` §6a/§6b,
> `docs/WAVELOCK_MERKLE_ROADMAP.md`, and `tests/test_ots_redteam.py`):
>
> | # | Finding | Status |
> |---|---------|--------|
> | A | merkle_root unverified / commitments unbound → key substitution | **FIXED** — `verify_ots`/`load_public_key` recompute the Merkle root from `pk_commitments` and recompute the canonical `public_key_fingerprint`; both are required to match. |
> | B | signature malleability (SUF broken) | **FIXED** — strict exact canonical signature field set (no missing/extra), enforced scheme/version/hash_alg, present+correct `message_digest`, and `one_time_key_id`/`public_key_fingerprint`/`params_hash`/`psi_commitment` all bound to the key. |
> | C | one-time reuse → total forgery | **INHERENT (NOT fixed)** — inherent to Lamport-style OTS; PoC preserved and still succeeds on reuse. Mitigation is non-reuse + ledger duplicate-key rejection, not a code fix. |
> | D | one-time enforcement is advisory file state | **MITIGATED host-locally / otherwise a deployment issue** — signing atomically claims `one_time_key_id` in a host-local registry (a copied key cannot sign twice on the same host); a copy on another host still bypasses it. Production MUST reject duplicate `one_time_key_id`/leaf at the verifier/ledger (`OTSReplayLedger`, roadmap). |
> | G | OTS not wired into consensus | **DOCUMENTED BLOCKER** — block acceptance still verifies legacy SIGv2. A fail-closed `verify_ots_payload` server entry point (with duplicate-key rejection, never accepting legacy where OTS is expected) is added, but consensus integration is not done. |
>
> The PoC scripts and this report are kept as regression evidence. The A/B PoCs
> now fail to forge (the breaks are closed); the C PoC still forges on reuse.
> **WaveLock-OTS remains experimental and is NOT production- or bounty-ready.**
>
> ---

> **Scope:** the *new* WaveLock-OTS construction only —
> `wavelock/crypto/wavelock_ots.py`, the `ots-*` CLI
> (`wavelock/crypto/ots_cli.py`), the server verification changes
> (`wavelock/network/server.py`), `docs/WAVELOCK_OTS_DESIGN.md`,
> `docs/WAVELOCK_OTS_REPORT.md`, and the OTS tests. Legacy SIGv2 is explicitly
> out of scope (it is already known-broken; those breaks are *preserved*).
>
> **Verdict:** WaveLock-OTS is *not* production-ready and should *not* go to a
> public bounty in its current state. The core Lamport math is sound for a
> *single, honest, full-public-key* use, but the construction has **four
> exploitable defects** and several residual risks. All are reproduced by
> runnable scripts in `attacks/` and pinned by tests in
> `tests/test_ots_redteam.py`. No claim of production readiness is made.

---

## 1. Summary of findings

| # | Finding | Severity | Exploit | Test |
|---|---------|----------|---------|------|
| A | `merkle_root` is never verified; `pk_commitments` are not bound to the key fingerprint → key-substitution forgery | **High** (deployment-dependent) | `attacks/ots_key_substitution.py` | `test_secure_verify_rejects_inconsistent_merkle_root` (xfail) |
| B | Signature malleability — `one_time_key_id`, `version`, `hash_alg`, `message_digest`, extra fields unchecked (SUF broken) | **Medium** | `attacks/ots_signature_malleability.py` | `test_secure_verify_rejects_mutated_key_id`, `test_secure_verify_requires_message_digest_field` (xfail) |
| C | One-time reuse → total forgery (canonical Lamport) | **Critical *if* reuse occurs** | `attacks/ots_reuse_to_total_forgery.py` | `test_break_reuse_enables_total_forgery` |
| D | One-time enforcement is advisory file state (copy / crash / restore / race bypass) | **High** | CLI repro in §3.D | `test_secure_copied_key_cannot_sign_twice` (xfail) |

Plus residual risks E–K in §4. **No total break of a single, honest, fully-pinned
public key was found** (the Lamport preimage argument holds); the breaks are in
*binding*, *malleability*, *reuse handling*, and *integration*.

---

## 2. What is actually solid

To be fair to the design, these properties were checked and **hold**:

- **Verifier-cannot-forge (single honest use).** With a fixed, fully-trusted
  public key and exactly one signature, forging `m′ ≠ m` requires a SHAKE256-256
  preimage of a published `pk[i][bit′_i]` for ≥1 differing bit. The naive
  "reuse the public commitments as secrets" forgery fails
  (`attempt_forge_ots_from_public`). Full Lamport (two slices/bit, reveal one)
  is correctly used, so **no checksum is required** and none is missing.
- **No ψ★/ψ₀/seed/raw-slice in the public key.** Verified by inspection and by
  `_FORBIDDEN_PUBLIC_FIELDS` guards on export *and* load.
- **Domain separation.** `_h` is length-prefixed (8-byte big-endian per field)
  and every call site uses a distinct domain tag. No cross-protocol collision
  was found. (Minor: `WL-PSI-COMMIT` lacks the `-v1` suffix the others carry.)
- **params_hash binding.** `params_hash` is recomputed from the published
  params and required to match both the public key and the signature; the
  message digest folds in `params_hash`. Tampering params fails closed.
- **Message digest is recomputed, not trusted.** Verify never relies on the
  signature's `message_digest` for the actual check (it recomputes), so a wrong
  digest cannot redirect verification (but see Finding B for the *missing*-field
  case).
- **`verify_ots` is wrapped in `try/except Exception: return False`** —
  structurally fail-closed against raised exceptions.
- **Entropy floor.** 128-bit minimum, 256-bit default, `os.urandom`; tiny and
  all-constant seeds rejected on generate *and* load. The legacy small-integer
  seed brute-force does not apply.

---

## 3. Findings in detail

### A. `merkle_root` is published but never verified; commitments are unbound (High)

`generate_ots_keypair` computes `merkle_root` over all `pk_commitments` and
publishes it. `docs/WAVELOCK_OTS_DESIGN.md` §4–5 and the deterministic-keypair
test present it as part of the *binding* public key. **But `verify_ots` never
references `merkle_root`, and never checks that `pk_commitments` are consistent
with either `merkle_root` or `psi_commitment`.** The only things bound to the
slices that are actually checked are `params_hash` (which covers *params*, not
the commitments) and a bare equality check on `psi_commitment`.

Consequences:

1. The published `merkle_root` provides **zero** verification value. Overwrite
   it with `"de"*32` and signatures still verify.
2. `pk_commitments` are accepted with **no integrity binding** to the key's
   advertised compact fingerprint. Any deployment that identifies / pins a key
   by `merkle_root`, `one_time_key_id`, or `psi_commitment` while accepting the
   `pk_commitments` array alongside is **fully forgeable**: the attacker
   presents a public key carrying the *victim's* fingerprint fields but the
   *attacker's* `pk_commitments` + `params` + `psi_commitment`, then signs an
   arbitrary message with their own secret key.

```
$ python -m attacks.ots_key_substitution
[A] garbage merkle_root still verifies:   True
[A] fingerprint-keyed substitution forge: True
```

Note on precondition: if a verifier obtains the *entire* `public_key` object
from a fully trusted channel and treats it as opaque, A is "only" a missing
defense-in-depth check. But the design *advertises* `merkle_root` as a binding
fingerprint, which invites exactly the fingerprint-keyed deployment that A
breaks. The Merkle root and ψ-commitment must be verified to mean anything.

### B. Signature malleability — strong unforgeability is broken (Medium)

`verify_ots` validates `scheme`, `params_hash`, `psi_commitment` (equality
only), and the revealed slices. It does **not** validate fields that are carried
*inside the signature object*:

- `one_time_key_id` — unchecked, freely mutable
- `version` — unchecked
- `hash_alg` — unchecked (you can set it to `"MD5"`; it still verifies)
- `message_digest` — may be **dropped entirely** (`... in (None, digest.hex())`)
- arbitrary extra keys — ignored

So from one valid signature an attacker mints unboundedly many distinct
byte-strings that all verify. SUF is violated. This is directly dangerous for
any replay / double-spend layer that dedupes by hashing the signature object or
keys on the signature's `one_time_key_id` — that id is attacker-controlled and
not bound to the verifying key.

```
$ python -m attacks.ots_signature_malleability
[B] mutated one_time_key_id          verifies=True
[B] mutated version + hash_alg       verifies=True
[B] message_digest removed           verifies=True
[B] injected extra field             verifies=True
```

### C. One-time reuse → total forgery (Critical if reuse occurs)

WaveLock-OTS is plain Lamport: each extra signature on a fresh message reveals,
on average, half the previously-unrevealed slices. After ~30–48 signatures the
attacker holds **both** `sk[i][0]` and `sk[i][1]` for every one of the 256 bit
positions and can assemble a valid signature for **any** message — never
touching the seed.

```
$ python -m attacks.ots_reuse_to_total_forgery
[OTS reuse] positions with BOTH halves: 256/256
[OTS reuse] forged arbitrary message verifies: True
```

This is inherent to Lamport and is *acknowledged* in the docs — but it elevates
Finding D from "hygiene" to "the load-bearing control," and there is no
two-signature/grinding mitigation (no checksum-based WOTS, no stateful counter).

### D. One-time enforcement is advisory file state (High)

The only thing preventing C is the `used` boolean. It is:

- a **mutable field in the secret-key dict / JSON file**, not a cryptographic or
  global guard;
- enforced **per file**, so a copy taken *before* signing signs again;
- persisted **after** the signature is already written to disk in `cmd_sign`
  (crash window between sig-write and `used=True`-write), with no `fsync` and no
  atomic replace;
- resettable by anyone who can edit the file (and `ots-mark-used` / restoring a
  backup trivially un-sets it).

Reproduced via the CLI (one public key, two valid signatures on two messages):

```
$ ots-keygen --out keys
$ cp keys/wl_ots_secret.json keys/wl_ots_secret.COPY.json   # copy BEFORE signing
$ ots-sign --secret keys/wl_ots_secret.json      --message "msg one" --sig sig1.json   # ok
$ ots-sign --secret keys/wl_ots_secret.json      --message "msg two" --sig sig2.json   # REFUSED (exit 2)
$ ots-sign --secret keys/wl_ots_secret.COPY.json --message "msg two" --sig sig2.json   # exit 0  ← bypass
$ ots-verify --public keys/wl_ots_public.json --message "msg one" --sig sig1.json      # VALID
$ ots-verify --public keys/wl_ots_public.json --message "msg two" --sig sig2.json      # VALID
```

Two valid signatures under one key is the exact foothold for Finding C.

---

## 4. Residual risks / missing proofs (no working break, but flagged)

- **E. ψ-binding is cosmetic at verify time.** `psi_commitment` is folded into
  `sk` derivation, but `verify_ots` only does an *equality* check
  (`sig.psi_commitment == pk.psi_commitment`); the slice math
  (`_public_slice`) does not include it. Security reduces entirely to SHAKE256
  one-wayness on `pk[i][b]`. The "curvature/coherence hardness" thesis adds
  **no proven verification-time security** — as the design doc itself admits.
- **F. `n_bits` is decorative.** `_message_digest` and `_digest_bits` are pinned
  to the hardcoded `_N_BITS = 256`, ignoring `params["n_bits"]`. Setting it
  elsewhere just breaks length checks (fail-closed) rather than re-parameterising
  — a footgun, not an exploit, but it means the parameter is a lie.
- **G. OTS is not wired into the chain.** `server.py` / block acceptance still
  verify *legacy SIGv2* (`CurvatureKeyPair.verify`, which requires ψ★). The new
  asymmetric scheme is standalone; the "server verification changes" only
  *hardened the legacy path* (fail-closed, no trust-only branch). So consensus
  is **not** actually protected by WaveLock-OTS today, and the legacy
  symmetric-MAC verification remains the live path. No OTS signature is accepted
  by the server anywhere.
- **H. Weak entropy test.** `_validate_seed` only checks length ≥ 128 bits and
  "not all one byte." A 16-byte seed with two distinct values passes. Default
  `os.urandom(32)` is fine; the concern is explicit/derived seeds.
- **I. No formal proof.** EUF-CMA (one-time) reduction to SHAKE256 preimage is
  asserted, not proven; ψ-binding is unmodeled.
- **J. Side channels** (ψ evolution timing, scrypt) unanalyzed;
  `_ct_eq` is constant-time-ish over hex strings but the surrounding control
  flow (early `return False`, per-bit loop) is not.
- **K. `version` / `scheme` integer-vs-string.** `scheme` string is checked;
  the numeric `version` is not (folds into B).

---

## 5. Audit-question scorecard

1. **Verifier forge a different message?** No, for one honest fully-pinned key
   (Lamport preimage). **Yes** via Finding A in a fingerprint-keyed deployment,
   and **yes** via C after reuse.
2. **Malleability change message/params/key_id/domain?** Message/params/domain:
   no (bound via `params_hash`/recompute). **key_id, version, hash_alg,
   message_digest: yes** — Finding B.
3. **Canonicalization / encoding ambiguity?** Message digest hashes raw bytes
   with length-prefixed domain sep; no JSON ambiguity for the message. Params
   use sorted-key canonical JSON. No collision found. (Dropping
   `message_digest` is a *missing-field* issue, B, not canonicalization.)
4. **Public artifacts leak secret material?** No — guarded on export and load;
   tests confirm no seed/ψ★/ψ₀.
5. **Key reuse via CLI/server/copy/race/--unsafe?** **Yes** — Finding D (copy,
   crash window, restore, race) and the documented `--unsafe-allow-reuse`.
6. **Verification bypass via legacy fallback / non-strict / missing fields /
   trust-list?** No legacy fallback inside `verify_ots`; no non-strict mode.
   **But** missing `message_digest` is accepted (B), `merkle_root` is ignored
   (A). Server trust-list is necessary-not-sufficient (good) but applies to the
   *legacy* path only (G).
7. **Domain separators complete/unique?** Yes; minor `-v1` inconsistency on
   `WL-PSI-COMMIT`.
8. **params_hash / psi_commitment / key_id / scheme bound in sign *and*
   verify?** params_hash: yes. psi_commitment: sign yes, verify equality-only
   (E). **key_id: no (B). scheme: string yes, version no (B/K).**
9. **Exactly one secret per digest bit, never both?** Yes for a single
   signature (`test_signature_reveals_only_the_selected_half`).
10. **Two+ signatures reveal both halves → forge?** **Yes** — Finding C.
11. **Digest length / hash / truncation / bit order weakness?** Bit order
    consistent (MSB-first, both sides); full 256-bit digest; no truncation. F is
    a parameter-hygiene issue, not a weakness.
12. **Seed entropy downgrade via CLI/config/defaults/tests?** Floor enforced;
    default 256-bit. Weak two-value-seed edge (H). No CLI path below 128.
13. **Commitments replaced / reordered / truncated / duplicated?** Count is
    checked (`len == n_bits`), but **content is unbound** (A) — replacement
    succeeds; truncation fails closed.
14. **Merkle root bypassed / mismatched with commitments?** **It is never
    checked at all** — Finding A.
15. **`verify_ots` fail-closed on every exception?** Yes (broad `except`).
16. **Server accepts old SIGv2 or malformed OTS?** Server accepts **only**
    legacy SIGv2 (G); it does not accept OTS at all, and does not fail open on
    malformed input (it rejects). OTS is simply not integrated.
17. **Unsafe debug exports impossible by default / loudly marked?** Yes —
    `--unsafe-export-secret-state` is off by default and prints a warning;
    `--unsafe-allow-reuse` likewise.
18. **Old attacks preserved and failing against OTS?** Yes — legacy forge /
    seed-bruteforce preserved and still pass against SIGv2; OTS variants fail.
19. **Tests meaningful or happy-path?** Existing OTS tests are decent (tamper,
    truncation, reuse-rejected, leak guards) but **missed A, B, D entirely** and
    assert a `merkle_root` that is never used. This audit adds adversarial tests.
20. **Exact changes needed before a public bounty?** See §6.

---

## 6. Required changes before a public bounty

**Must-fix (blockers):**

1. **Bind the commitments (A).** In `verify_ots`, recompute `_merkle_root` over
   `pk_commitments` and require it to equal `public_key["merkle_root"]`. Better:
   derive a single canonical *key fingerprint* = `H(domain ‖ params_hash ‖
   psi_commitment ‖ merkle_root)` and have signatures/verifiers bind to *that*,
   so `pk_commitments` cannot be swapped under a pinned identity.
2. **Kill malleability (B).** Verify (or strip-and-ignore by canonical
   reserialization) every signature field: require `version`, `hash_alg`
   to equal the expected constants; bind `one_time_key_id` to
   `public_key["one_time_key_id"]`; require `message_digest` to be **present and
   equal** to the recomputed digest; reject unknown keys. Define a canonical
   signature encoding and verify against it.
3. **Make one-time-ness enforceable (D).** Stop relying on a mutable bool in the
   signing input. Options: a stateful, append-only "used key_id" store consulted
   by `sign_ots`/`verify_ots`; write `used=True` *atomically and fsynced before*
   emitting the signature; and have *verifiers* reject a second signature seen
   under the same `one_time_key_id` (requires binding key_id, per B). At minimum
   document loudly that copy/restore defeats it and that this is unsafe for
   value transfer.
4. **Decide the integration story (G).** Either wire OTS into block acceptance
   (and delete/disable the legacy ψ★-MAC verification path from consensus) or
   state explicitly that OTS protects nothing on-chain yet. Do not ship a bounty
   implying the chain uses OTS while it verifies SIGv2.

**Should-fix:**

5. Make `_message_digest`/`_digest_bits` honor `params["n_bits"]`, or remove the
   parameter (F).
6. Strengthen `_validate_seed` (min distinct-byte count / simple entropy
   estimate) (H).
7. Add `-v1` to the `WL-PSI-COMMIT` domain tag for consistency (cosmetic).

**Before any "secure" claim:** a written one-time EUF-CMA argument, an explicit
statement that ψ-binding adds no verification-time hardness (E/I), and a
side-channel note (J).

---

## 7. Deliverables in this audit

- `attacks/WAVELOCK_OTS_REDTEAM.md` — this report.
- `attacks/ots_key_substitution.py` — Finding A (runnable).
- `attacks/ots_signature_malleability.py` — Finding B (runnable).
- `attacks/ots_reuse_to_total_forgery.py` — Finding C (runnable).
- `tests/test_ots_redteam.py` — demonstration tests (pass now, pin the breaks)
  + `xfail(strict=True)` tests asserting the desired secure behavior (fail now;
  flip to a hard failure when fixed, forcing the marker's removal).

No existing tests or attacks were deleted. The legacy SIGv2 breaks remain
preserved. **WaveLock-OTS is experimental and not production-ready.**
