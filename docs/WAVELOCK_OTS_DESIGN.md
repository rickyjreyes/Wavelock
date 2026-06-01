# WaveLock-OTS Design

> **Status: EXPERIMENTAL.** WaveLock-OTS is a research construction. It is not
> a reviewed cryptographic standard and has no formal security proof. **Do not
> use it for production funds.** For production, use Ed25519, SLH-DSA, LMS, or
> XMSS.

## 1. Why the old WaveLock failed

Legacy WaveLock ("SIGv2" / `WLv2`) signs a message as:

```
signature = H("SIGv2" || message || header || ψ★)
```

and `verify(message, signature)` recomputes **exactly that hash** and compares
it to the signature. Therefore:

- **Verifying requires ψ★.**
- **Signing/forging requires ψ★.**

The set of parties able to verify and the set able to forge are *identical*.
This is a **symmetric MAC** keyed by ψ★ — and legacy "strict" mode literally
publishes ψ★ as verifier material. Consequences (all demonstrated as
regression tests in `attacks/` and `tests/test_legacy_sigv2_broken.py`):

1. **Forgery from a snapshot.** Anyone who obtains the published ψ★ can sign
   any message (`attacks/forge_from_snapshot.py`).
2. **Seed brute-force.** Example keys use tiny integer seeds (`12`, `42`,
   `123`); the commitment is a deterministic function of the seed, so the seed
   (and thus ψ★) is recoverable by enumeration (`attacks/seed_bruteforce.py`).
3. **Cleartext export.** `keypair.json` wrote `seed`, `psi_0`, and `psi_star`
   in cleartext.
4. **Fail-open server.** Non-strict mode accepted blocks on trust-list
   membership alone; strict mode allowed unpublished proofs by default.

There is no asymmetry to "fix" in SIGv2 — it was never an asymmetric signature.

## 2. The security property WaveLock-OTS restores

WaveLock-OTS restores the **minimum asymmetric one-time signature property**
that Lamport already satisfies:

> The public key, plus a signature on message *m*, must not let an adversary
> produce a valid signature on any *m′ ≠ m*. Verification uses only public
> material.

Concretely, the hard requirements (all covered by tests):

1. Public verification never requires ψ★.
2. The verifier never receives enough material to forge a different message.
3. The public key is **only** commitments, hashes, a Merkle root, parameters,
   and metadata.
4. The secret key holds high-entropy seed material (≥128 bits, default 256) and
   the ψ-generation parameters; ψ-derived secret slices are derived on demand,
   never stored in a public artifact.
5. A signature reveals only the message-selected ψ-derived slices.
6. Each key is one-time by default; reuse is detected and loudly rejected.

## 3. Comparison to Lamport OTS

WaveLock-OTS is structurally a **Lamport one-time signature**:

| Step       | Lamport                              | WaveLock-OTS |
|------------|--------------------------------------|--------------|
| Secret     | `sk[i][b]` = random 256-bit          | `sk[i][b]` = SHAKE256 of seed **+ ψ-commitment** + params + (i,b) |
| Public     | `pk[i][b] = H(sk[i][b])`             | `pk[i][b] = H("WL-OTS-PK" ‖ params_hash ‖ i ‖ b ‖ sk[i][b])` |
| Digest     | `H(message)` → bit string            | `SHAKE256("WL-OTS-MSG" ‖ params_hash ‖ message)` → 256 bits |
| Sign       | reveal `sk[i][bit_i]`                | reveal `sk[i][bit_i]` |
| Verify     | `H(revealed) == pk[i][bit_i]`        | strict canonical key+sig (see §6a) then `H(... ‖ revealed) == pk[i][bit_i]` |
| One-time   | required                              | required; host-local atomic registry + ledger model (§6b) — reuse still inherently catastrophic |

Like Lamport, security rests on the one-wayness of the hash: forging *m′*
requires the *unrevealed* `sk[i][bit′_i]` for at least one differing bit, which
is a 256-bit preimage of a published `pk` commitment.

The **difference** is the WaveLock binding: the secret slices are not raw
random bytes; they are bound to a WaveLock ψ-state evolution.

## 4. How ψ-state is used

```
seed (256-bit os.urandom)
   │  SHAKE256, domain "WL-PSI-INIT-v1"
   ▼
 ψ₀  (deterministic, byte-stable across CPU/GPU)
   │  WaveLock PDE evolution (same kernel as the rest of WaveLock)
   ▼
 ψ★
   │  quantize → SHAKE256, domain "WL-PSI-COMMIT"
   ▼
 psi_commitment   ── published in the public key (a 32-byte hash, NOT ψ★)
   │
   ▼
 sk[i][b] = SHAKE256("WL-OTS-SK" ‖ seed ‖ psi_commitment ‖ params_hash ‖ i ‖ b)
 pk[i][b] = SHAKE256("WL-OTS-PK" ‖ params_hash ‖ i ‖ b ‖ sk[i][b])
 merkle_root = Merkle(all pk[i][b])
```

The kernel parameters (PDE coefficients, step count, kernel hash) are folded
into `params_hash`, so the keys and signatures are bound to a specific ψ
evolution. Producing a valid signature requires re-running the exact ψ
pipeline from the seed — that is the WaveLock thesis carried into a
genuinely asymmetric construction.

## 5. Why full ψ★ is never public

ψ★ is the secret. If ψ★ (or enough of it) were public, an adversary could
recompute the secret slices `sk[i][b]` for **both** halves of every bit and
forge arbitrarily — exactly the SIGv2 break. WaveLock-OTS publishes only:

- `psi_commitment = H(quantized ψ★)` — a 32-byte hash, irreversible;
- the `pk[i][b]` commitments and their Merkle root — hashes of the secret
  slices.

None of these reveal ψ★ or any `sk`. The verifier learns a secret slice only
when the *signer chooses* to reveal it, and only the message-selected half.

## 6. Why one-time usage is required

Lamport (and thus WaveLock-OTS) is **one-time**. Signing two different messages
with the same key reveals secret slices for both bit patterns. Where the two
message digests differ in a bit, the adversary then holds *both* `sk[i][0]` and
`sk[i][1]`, and can mix-and-match to forge a third message. WaveLock-OTS:

- stores `used: false` in the secret key;
- sets `used: true` after the first signature;
- refuses a second signature (`OTSKeyReuseError` / CLI exit code 2) unless the
  explicit, loudly-warned `--unsafe-allow-reuse` flag is given (tests only);
- additionally claims the `one_time_key_id` in a host-local atomic registry so a
  *copy* of the secret file cannot sign twice on the same host.

The `used` flag is advisory and the registry is host-local — neither is
cryptographic reuse prevention. See §6b and `docs/WAVELOCK_MERKLE_ROADMAP.md`
for why a server/ledger duplicate-key check is required in production.

## 6a. Canonical objects, fingerprint binding, strict verification (Findings A & B)

The red-team audit (`attacks/WAVELOCK_OTS_REDTEAM.md`) found that the public key
and signature were under-bound. These are now closed.

### Canonical public key

A WaveLock-OTS public key is **exactly** these fields (strict verification and
`load_public_key` reject any key with unknown or missing fields):

```
scheme, version, hash_alg, params_hash, psi_commitment,
one_time_key_id, pk_commitments, merkle_root, public_key_fingerprint
```

Note `params` is **not** a public-key field. It is bound through `params_hash`
(and the fingerprint); the full parameter set lives only in the secret key. This
removes the previous ambiguity where `params` and `pk_commitments` rode along
unbound.

### Fingerprint binding (closes Finding A)

```
public_key_fingerprint = H( "WL-OTS-PK-FINGERPRINT-v1"
                            ‖ canonical_json(scheme, version, hash_alg,
                                             params_hash, psi_commitment,
                                             one_time_key_id, pk_commitments,
                                             merkle_root) )
```

On load **and** verify, WaveLock-OTS:

1. recomputes `merkle_root` from `pk_commitments` (in `(i, b)` leaf order) and
   requires it to equal the stored root — a garbage/tampered root is rejected;
2. recomputes `public_key_fingerprint` over the canonical payload and requires
   it to equal the stored fingerprint — so `pk_commitments` cannot be swapped
   under a victim's pinned identity (key substitution is rejected).

Because `pk_commitments` are folded into the fingerprint via canonical
serialization, the fingerprint *is* the binding identity.

### Strict signature verification (closes Finding B)

A WaveLock-OTS signature is **exactly** these fields (no missing, no extra):

```
scheme, version, hash_alg, one_time_key_id, public_key_fingerprint,
params_hash, psi_commitment, message_digest, revealed_slices
```

`verify_ots` rejects unless **all** hold (fail-closed on any exception):

- exact field set; `scheme`/`version`/`hash_alg` equal the expected constants;
- `message_digest` is present and equals the digest recomputed from the message;
- `one_time_key_id` equals the public key's `one_time_key_id`;
- `public_key_fingerprint` equals the recomputed public-key fingerprint;
- `params_hash` equals the public key's `params_hash`;
- `psi_commitment` equals the public key's `psi_commitment`;
- `len(revealed_slices)` equals the digest bit length (256);
- each revealed slice hashes to `pk[i][selected_bit]`.

This removes the malleability that previously let an attacker mint many distinct
verifying byte-strings from one signature.

### Domain separators

Every hash is domain-separated and versioned:

| Purpose | Domain tag |
|---------|------------|
| public-key fingerprint | `WL-OTS-PK-FINGERPRINT-v1` |
| Merkle leaf hash | `WL-OTS-MERKLE-LEAF-v1` |
| Merkle internal node hash | `WL-OTS-MERKLE-NODE-v1` |
| message digest | `WL-OTS-MSG-v1` |
| signature transcript | `WL-OTS-SIG-TRANSCRIPT-v1` |
| secret slice / public slice | `WL-OTS-SK-v1` / `WL-OTS-PK-v1` |
| ψ-commitment | `WL-PSI-COMMIT` |

## 6b. Reuse (C) and one-time enforcement (D)

- **Finding C is inherent and NOT fixed** by the A/B hardening. Reusing a key
  still leaks both Lamport halves and enables total forgery; strict canonical
  checks cannot stop an attacker who assembles a fully canonical signature from
  harvested slices. The PoC (`attacks/ots_reuse_to_total_forgery.py`) is
  preserved and still succeeds once the key is reused. The defenses are: never
  reuse a key, and reject duplicate key/leaf usage at the ledger.
- **Finding D is mitigated host-locally and otherwise a deployment issue.**
  Signing now atomically claims the `one_time_key_id` in a host-local key-state
  registry (`O_CREAT|O_EXCL`, `WAVELOCK_OTS_STATE_DIR`), so a copied secret key
  cannot sign twice **on the same host**. A copy moved to another host, or a
  wiped registry, still bypasses it — this is **not** cryptographic reuse
  prevention. Production verifiers/servers/ledgers MUST reject duplicate
  `one_time_key_id` (and, under WaveLock-Merkle, duplicate leaf indices);
  `OTSReplayLedger` models this check. See `docs/WAVELOCK_MERKLE_ROADMAP.md`.

## 7. What remains unproven / experimental

- **No formal proof.** We rely on the standard Lamport argument plus the
  one-wayness of SHAKE256; this has not been formally modeled for the
  ψ-binding step.
- **ψ-binding adds no proven hardness.** The ψ pipeline binds keys to a kernel
  and makes signing require ψ regeneration, but security currently reduces to
  the hash, not to any PDE hardness assumption. The "coherence/curvature
  hardness" thesis is **unproven**.
- **Quantization choices** for `psi_commitment` are heuristic.
- **Side channels** (timing of ψ evolution, scrypt) are not analyzed.
- **One-time only.** No many-signature mode yet (see roadmap).

## 8. Future roadmap

- **WaveLock-Merkle** — a Merkle tree of many WaveLock-OTS public keys (XMSS/
  LMS-style) so one root public key authorizes many one-time signatures.
- **WaveLock-ZK** — a zero-knowledge proof of correct ψ evolution from a
  committed seed, so the ψ-binding contributes verifiable structure without
  revealing ψ★, enabling commitment-only verification of the curvature claim.
- **Formal security proof** — EUF-CMA (one-time) reduction to SHAKE256
  preimage/second-preimage resistance, plus an explicit statement of what (if
  anything) the ψ-binding adds.
- **Post-quantum parameter review** — Lamport/WOTS over a 256-bit hash is
  already a conservative hash-based (PQ-friendly) design; this needs an
  explicit parameter and security-level review against SLH-DSA/LMS/XMSS.
