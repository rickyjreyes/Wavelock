# WaveLock-Encrypt v1 — Security Note

**Status: EXPERIMENTAL. Not production audited.** Do not rely on this for
protecting real secrets until it has had external cryptographic review.

## What this is

`wavelock/crypto/wavelock_encrypt.py` is a thin, hybrid public-key encryption
wrapper. It is **not a new raw cipher.** All confidentiality and integrity come
from standard, well-reviewed primitives provided by
[`pyca/cryptography`](https://cryptography.io):

| Layer            | Primitive                          |
|------------------|------------------------------------|
| Key exchange     | **X25519** (ephemeral-static ECDH) |
| Key derivation   | **HKDF-SHA256**                    |
| Authenticated enc| **ChaCha20-Poly1305** (AEAD)       |
| Context encoding | Canonical JSON (sorted keys, compact, UTF-8, `allow_nan=False`) |

A fresh ephemeral X25519 keypair is generated for **every** encryption. The
recipient decrypts with their long-term X25519 private key. The 32-byte AEAD
key is derived via HKDF-SHA256 from the ECDH shared secret, a random 32-byte
salt, and an `info` string that also commits to the AAD hash. The nonce is a
random 12 bytes.

## Two modes: `standard` and `wavelock-experimental`

The envelope carries a strict `mode` field. It is authenticated **twice**: as a
top-level envelope field *and* inside the canonical-JSON AAD (via the context).
Decryption requires the envelope mode, the AAD-embedded mode, and the caller's
expected mode to all agree, so a ciphertext minted in one mode cannot be
reinterpreted in the other.

### `standard` (secure default)

Exactly the original v1 derivation — reviewed primitives only:

```
shared_secret = X25519(ephemeral_private, recipient_public)
final_key     = HKDF-SHA256(shared_secret, salt, info = KDF_INFO | sha256(aad))
ciphertext    = ChaCha20-Poly1305(final_key).encrypt(nonce, plaintext, aad)
```

This is the default for the API (`make_context(..., mode="standard")`) and the
CLI (`--mode standard`, the default if `--mode` is omitted).

### `wavelock-experimental` (research only)

**EXPERIMENTAL. Do not use for real secrets.** Same envelope and same
primitives, but it folds WaveLock binding material into a second HKDF-SHA256
pass under a distinct domain tag:

```
standard_key      = HKDF-SHA256(shared_secret, salt, info = KDF_INFO | sha256(aad))
wavelock_material = derive_wavelock_material(secret_input=standard_key, context, aad)
final_key         = HKDF-SHA256(standard_key || wavelock_material,
                                salt, info = KDF_INFO | sha256(aad) | "|WL-EXPERIMENTAL")
ciphertext        = ChaCha20-Poly1305(final_key).encrypt(nonce, plaintext, aad)
```

`derive_wavelock_material` is **deterministic**, uses **only HKDF-SHA256** (no
custom cipher), and commits to the WaveLock transcript: `purpose`,
`psi_commitment`, `block_digest`, `ots_public_key_fingerprint`, the canonical
AAD (via its SHA-256), and an optional WaveLock params hash supplied via
`extra["wavelock_params"]`. It is keyed by a shared-secret-derived input and
**never returns or logs** the shared secret, the standard key, the final key,
ψ★, seeds, or private OTS slices. It **fails closed** if any required WaveLock
context field is missing — enforced both in `make_context` (at construction) and
again inside `derive_wavelock_material` (defense in depth).

**Why this exists:** to study whether WaveLock ψ/context binding adds anything
over the standard path. Because the standard path *already* binds the same
context through the AEAD's AAD, the experimental layer is a re-derivation under
a different domain — useful for A/B comparison, **not** an established security
improvement. It does **not** remove SHA (both passes are HKDF-SHA256) and it is
**not** a production cipher.

## What "WaveLock" actually contributes

The WaveLock-specific part is **canonical transcript / context binding**, not
the cryptography. Encryption is bound to an authenticated context
(`make_context`) that becomes the AEAD's Additional Authenticated Data (AAD):

- `purpose` (required)
- `psi_commitment` (optional)
- `block_digest` (optional)
- `ots_public_key_fingerprint` (optional)
- `extra` (optional metadata dict)

The context is serialized with a deterministic canonical JSON encoding, so the
exact same context always produces the exact same AAD bytes regardless of key
insertion order. **Decryption fails closed if any authenticated context field
differs** from what was used at encryption time. This lets a caller bind a
ciphertext to a protocol purpose, a ψ-commitment, a canonical block digest, or
an OTS public-key fingerprint, so a ciphertext minted for one context cannot be
silently replayed under another.

## Envelope is strict

The JSON envelope has an **exact** field set (`ENVELOPE_FIELDS`). Decryption
rejects, fail-closed, on any of:

- missing or extra envelope fields
- wrong `scheme` / `version` / `kem` / `kdf` / `aead`
- AAD / context mismatch (and the redundant `aad_sha256` check)
- tampered ciphertext, nonce, salt, or ephemeral public key
- malformed base64
- wrong nonce / salt / ephemeral-key lengths
- wrong recipient private key
- a non-dict envelope or non-string base64 fields

### On `aad_sha256`

`aad_sha256` is **audit/debug metadata only.** It is retained in the envelope
and *is* verified during decrypt (it must equal `SHA256(expected_aad)`), but it
is **not** the source of authentication — the AEAD tag over the AAD is. We keep
it because it is convenient for logging/forensics and because it is cheap to
verify; we never treat a matching `aad_sha256` as sufficient on its own.

## Non-leakage

Errors and logs never include private key material, the ECDH shared secret, the
derived key, seeds, ψ★, unrevealed OTS slices, or plaintext. All decrypt
failures collapse to a generic `WaveLockDecryptError`; the unexpected-exception
path deliberately suppresses the underlying cause to avoid leaking internals.

## Red-team summary

`tests/test_wavelock_encrypt.py` (92 tests) exercises the required roundtrip
plus every mutation in the task's red-team list: extra/removed fields, mutated
ciphertext/nonce/salt/ephemeral-key, swapped AAD, mismatched context (purpose,
ψ-commitment, block digest, OTS fingerprint, extra), wrong scheme/version/
KEM/KDF/AEAD, replaced recipient key, replay under a different purpose, Unicode
/ huge-int / NaN / Infinity / empty-string contexts, malformed base64, multi-MB
fields, non-dict envelopes, and non-bytes plaintext.

The dual-mode tests additionally prove: standard and experimental roundtrips
both succeed; a standard envelope cannot decrypt as experimental and vice
versa; flipping the `mode` field (either direction) or using an unknown mode
rejects; changing any WaveLock context field rejects in experimental mode;
missing `psi_commitment` / `block_digest` / `ots_public_key_fingerprint` fails
closed in experimental mode; tampering and wrong-key rejection hold in both
modes; and the CLI supports `--mode standard` / `--mode wavelock-experimental`
and defaults to `standard`.

**Every mutation rejects; only the original envelope with the exact expected
context, mode, and correct private key decrypts.**

```
92 passed
```

## Final verdict

- **`standard` mode** — ready as an *experimental secure wrapper* over reviewed
  primitives (X25519 + HKDF-SHA256 + ChaCha20-Poly1305). Still pending external
  audit before any production claim, but it adds no custom cryptography.
- **`wavelock-experimental` mode** — **research only.** It is a hybrid built on
  standard primitives for study/comparison; it provides no demonstrated security
  advantage over standard mode and must not be used to protect real secrets.

## Limitations

- No forward secrecy for the recipient's *long-term* key: compromise of the
  recipient private key exposes all past ciphertexts addressed to it (standard
  for ephemeral-static hybrid encryption).
- No sender authentication: this is anonymous public-key encryption. If you
  need to know *who* sent a ciphertext, sign it (e.g. with WaveLock-OTS) and
  bind the signer's fingerprint into the context.
- No built-in replay/freshness beyond what the caller binds into the context.
- **Not production-ready. Experimental until external review.**
