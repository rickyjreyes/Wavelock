# Migration from WaveLock SIGv2

> **TL;DR:** Legacy WaveLock SIGv2 (`WLv2`) is **insecure**. Stop publishing
> ψ★. Rotate all keys. Use **WaveLock-OTS** for experiments. Use an
> established signature (Ed25519, SLH-DSA, LMS, XMSS) for anything that
> protects real value.

## Why migrate

Legacy SIGv2 is not an asymmetric signature. It signs with
`H("SIGv2" ‖ message ‖ header ‖ ψ★)` and verifies by recomputing the same hash,
so **anyone who can verify can forge**. See
`docs/WAVELOCK_OTS_DESIGN.md` §1 and `attacks/WAVELOCK_THEORY_BREAK_AUDIT.md`.

Existing SIGv2 signatures and keys must be treated as **compromised**.

## What to do

1. **Treat existing SIGv2 signatures as legacy/insecure.** Do not rely on them
   for authenticity. They prove nothing an adversary with the published ψ★
   could not also produce.

2. **Do NOT publish ψ★ as verifier material.** Any workflow, server config, or
   `commitments/*.npz` that distributes ψ★ for "strict verification" is leaking
   the secret. Remove it. The server now fails closed and no longer accepts
   blocks on trust-list membership alone, but the right fix is to stop using
   ψ★-based verification entirely.

3. **Rotate all keys.** Any key whose ψ★ was ever published, or that used a
   small integer seed (`12`, `42`, `123`, …), is recoverable. Generate new
   material with ≥128 bits (default 256) of entropy.

4. **Use WaveLock-OTS for experiments.**

   ```bash
   wavelock-ots ots-keygen --out keys/
   wavelock-ots ots-sign   --secret keys/wl_ots_secret.json --message "..." --sig sig.json
   wavelock-ots ots-verify --public keys/wl_ots_public.json --message "..." --sig sig.json
   ```

   - The public key (`wl_ots_public.json`) contains only commitments/hashes —
     safe to publish.
   - The secret key (`wl_ots_secret.json`) holds the seed — keep it local;
     optionally encrypt at rest (`--encrypt --passphrase ...`).
   - Each key is **one-time**. Generate a fresh key per message.

5. **Use established signatures for production security.** WaveLock-OTS is
   experimental and unproven. Until it is independently reviewed, protect real
   value with:
   - **Ed25519** — fast, ubiquitous classical signatures;
   - **SLH-DSA** (FIPS 205, SPHINCS+) — stateless hash-based, post-quantum;
   - **LMS / XMSS** (RFC 8554 / RFC 8391) — stateful hash-based, post-quantum.

## Mapping of concepts

| Legacy SIGv2                         | WaveLock-OTS                              |
|--------------------------------------|------------------------------------------|
| `keypair.json` (seed, ψ₀, ψ★ cleartext) | `wl_ots_secret.json` (seed only, optionally encrypted) |
| published ψ★ as verifier material    | `wl_ots_public.json` (commitments/hashes only) |
| `kp.sign(msg)` (MAC over ψ★)         | `wavelock-ots ots-sign` (reveals selected slices) |
| `kp.verify(msg, sig)` (needs ψ★)     | `wavelock-ots ots-verify` (public-only)  |
| reusable key                          | one-time key (reuse rejected)            |

## Server operators

The P2P server (`wavelock/network/server.py`) now:

- always requires a valid signature (no trust-only acceptance);
- rejects blocks whose proof material is unpublished;
- rejects on any verification error (fail closed).

This stops the previous fail-open behavior, but note the server still verifies
*legacy* SIGv2 signatures, which are themselves insecure. Plan to move
consensus verification to WaveLock-OTS (or an established signature) rather
than relying on ψ★-based verification at all.
