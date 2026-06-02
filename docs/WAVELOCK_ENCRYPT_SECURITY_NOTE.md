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

`tests/test_wavelock_encrypt.py` (52 tests) exercises the required roundtrip
plus every mutation in the task's red-team list: extra/removed fields, mutated
ciphertext/nonce/salt/ephemeral-key, swapped AAD, mismatched context (purpose,
ψ-commitment, block digest, OTS fingerprint, extra), wrong scheme/version/
KEM/KDF/AEAD, replaced recipient key, replay under a different purpose, Unicode
/ huge-int / NaN / Infinity / empty-string contexts, malformed base64, multi-MB
fields, non-dict envelopes, and non-bytes plaintext. **Every mutation rejects;
only the original envelope with the exact expected context and the correct
private key decrypts.**

```
52 passed
```

## Limitations

- No forward secrecy for the recipient's *long-term* key: compromise of the
  recipient private key exposes all past ciphertexts addressed to it (standard
  for ephemeral-static hybrid encryption).
- No sender authentication: this is anonymous public-key encryption. If you
  need to know *who* sent a ciphertext, sign it (e.g. with WaveLock-OTS) and
  bind the signer's fingerprint into the context.
- No built-in replay/freshness beyond what the caller binds into the context.
- **Not production-ready. Experimental until external review.**
