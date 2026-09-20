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

The migration is now implemented in the main CLI. `wavelock-cli keygen`,
`add`, `sign`, `mine`, `verify`, and `audit` use public WaveLock-OTS blocks.
Historical commands are available explicitly as `wavelock-cli legacy ...`.
This replaces the normal workflow; it does not make old SIGv2 signatures safe.

Start a new OTS ledger and generate fresh secret material:

```bash
wavelock-cli --data-dir new-ots-node keygen --out keys/ots-1
wavelock-cli --data-dir new-ots-node sign --secret keys/ots-1/wl_ots_secret.json --message "first OTS record" --output signed-block.json
wavelock-cli --data-dir new-ots-node mine --signed-path signed-block.json
wavelock-cli --data-dir new-ots-node verify
```

There is no automatic conversion of old signatures into OTS authorizations.
Existing keys, signatures, snapshots and ledgers are not rewritten or deleted.
To inspect the old package-local ledger, explicitly set `WAVELOCK_DATA_DIR` to
the old checkout's `wavelock/` directory before running historical tools; keep
that directory separate from the new OTS node. New default state is user-local,
and both node and CLI honor `WAVELOCK_DATA_DIR`.

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

- requires OTS by default (`require_ots=true`), including for otherwise generic
  blocks; this policy is actually loaded from JSON or `WAVELOCK_REQUIRE_OTS`;
- verifies OTS history from public material on startup;
- checks block hash, Merkle root, linkage, index, signature and replay state;
- rejects on verification errors without falling back to SIGv2;
- uses one data directory for block files, replay state and peer configuration.

Explicitly disabling `require_ots` retains the historical verification path for
compatibility experiments. Normal OTS operation does not publish or read ψ★.
