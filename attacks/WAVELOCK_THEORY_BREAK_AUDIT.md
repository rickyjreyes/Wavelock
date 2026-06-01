# WaveLock Theory-Break Audit Report

_Audit date: 2026-06-01 — performed against branch `claude/awesome-planck-mTs5B`._

## Executive verdict

**NOT READY: OBVIOUS THEORY WEAKNESS (and IMPLEMENTATION LEAKS THEORY SECRETS).**

WaveLock is presented as a "one-way curvature-lock" signature/identity primitive
whose security rests on the irreversibility of a nonlinear PDE evolution
ψ₀ → ψ★. In fact the deployed construction is a **symmetric keyed hash (a MAC)**:

```
sign(message)   = H( "SIGv2" ‖ message ‖ public_header ‖ ψ★ )
verify(message, sig) = ( sign(message) == sig )      # recomputed from ψ★
```

`verify()` recomputes the exact same hash, so **the verifier must possess the
full ψ★**. The project's own README and tooling ship ψ★ to verifiers as
`commitments/<hash>.npz` snapshots for "full strict verification"
(`tools/audit_multi_trust.py`, `wavelock/network/server.py:_strict_verify_curvature`).
Whoever can verify can forge. The PDE's claimed one-wayness is irrelevant to
forgery, because forgery never requires inverting the PDE — it requires ψ★,
which is published. Two working attack scripts in this folder demonstrate this.

This is not a packaging problem; it is a structural mismatch between the claimed
security model (asymmetric, one-way) and the implemented one (symmetric, shared
secret). It must be fixed before paying for any external review.

## Strongest break attempt

**Forgery from the verification material (`attacks/forge_from_snapshot.py`).**

The attacker is handed only what a legitimate strict verifier receives — the
published ψ★ snapshot and the public kernel parameters. Without a seed, without
running the PDE, and without inverting anything, the attacker reconstructs the
signature payload byte-for-byte and produces a valid signature for an arbitrary
new message. The signer's own `verify`, `verify` (secondary family) and
`verify_strict` all accept the forgery.

```
[verify] signer.verify(forged, primary)   = True
[verify] signer.verify(forged, secondary) = True
[verify] signer.verify_strict(forged,p,s)  = True
RESULT: FORGERY SUCCEEDED on an arbitrary message.
```

This is a complete break of existential unforgeability for any party with
verification access. There is no asymmetry anywhere in the codebase: every
verification path (`CurvatureKeyPair.verify`, `audit_multi_trust.py`,
`server._strict_verify_curvature`) reduces to "recompute the hash from ψ★."

## Second break: key space is the seed, not the curvature

**`attacks/seed_bruteforce.py`** shows that even an attacker who only has the
public commitment string (not ψ★) recovers the entire keypair, because the whole
secret is a deterministic function of one integer seed:

```
seed --SHAKE256--> ψ₀ --(50 fixed PDE steps)--> ψ★ --hash--> commitment
```

We never invert the PDE; we run it **forward** on candidate seeds and match the
commitment. Recovered seed `31337` in 160 s (≈193 evals/s/core). The honest work
factor is the entropy of `seed` plus ~5 ms of PDE key-stretching per guess —
i.e. a standard symmetric key-search, **not** anything curvature-related. The
README/CLI examples use tiny seeds (`--seed 42`, `--seed 12`, hard-coded `123`),
so the practical key space is whatever integer the operator typed.

## Strongest actual weakness found (summary)

The "curvature one-way function" provides **no asymmetry and no extra hardness**
over a plain keyed hash. Concretely:

1. **No public/private separation.** ψ★ is simultaneously the signing key and the
   verification key. Publishing it (required for strict verification) publishes
   the forging key. (`forge_from_snapshot.py`)
2. **Secrets stored in cleartext.** `cli.generate_key` writes `psi_0`, `psi_star`
   **and** `seed` into `keypair.json` unencrypted (`cli.py:121-131`).
   `generate_quantum_keys` writes ψ₀/ψ★ to `psi_keypair.json`. A published
   terminal field with full params + `seed=12` ships in `data/wavelock_data/`.
3. **Non-strict mode skips the signature entirely.** With
   `require_full_verify` off, `server._strict_verify_curvature` returns `True`
   on trust-list membership alone (`server.py:124-125`) — any message/signature
   attached to a trusted commitment is accepted. Even in strict mode, if ψ★ is
   not published it accepts by default policy (`server.py:132-134`).
4. **Security reduces to seed entropy** (`seed_bruteforce.py`), and the examples
   use guessable seeds.

The collision-resistance and preimage-resistance of the binding are exactly
those of SHA-256 / SHA3-256 — fine, but that is standard and not novel. The
novelty is entirely in the **framing**, not the math or the implementation.

## Files inspected

- `README.md` — workflow; documents publishing ψ★ snapshots for strict verify.
- `wavelock/chain/Wavelock_numpy.py` — reference `CurvatureKeyPairV3`; `sign`/
  `verify` = hash over `SIGv2‖msg‖header‖ψ★`. **Core break lives here.**
- `wavelock/chain/WaveLock.py` — GPU/Numpy `CurvatureKeyPair`; same sign/verify
  structure; `_sig_payload_v2` confirms ψ★ is hashed in plaintext.
- `wavelock/chain/xof_init.py` — SHAKE-256 ψ₀ derivation from integer/str seed.
- `wavelock/chain/hash_families.py` — SHA-256 / SHA3-256 / BLAKE3 dual-hash.
- `tools/audit_multi_trust.py` — loads ψ★ from `.npz`, calls `kp.verify` (proves
  verifier needs the secret).
- `tools/publish_trusted.py` — writes ψ★ to `commitments/*.npz` (ships the key).
- `wavelock/network/server.py` — `_strict_verify_curvature`; non-strict bypass.
- `wavelock/chain/cli.py` — `generate_key` writes seed+ψ₀+ψ★ in cleartext.
- `docs/inevitability/ATTACKBOUNDS.md`, `SURVIVABILITY.md` — hardness claims
  (audited below); they answer the wrong threat model.
- `data/wavelock_data/*` — a published terminal field with seed and all params.

## Commands run

```bash
pip install numpy pytest -q
pip install -e .
python3 -m wavelock.chain.Wavelock_numpy        # self-test (sign/verify OK)
python3 -m pytest tests/ -q                      # 95 passed, 10 skipped
python3 attacks/forge_from_snapshot.py           # forgery from ψ★ snapshot
python3 attacks/seed_bruteforce.py               # seed recovery -> full forgery
```

## Tests run

- `pytest tests/` → **95 passed, 10 skipped** (the suite confirms the code works
  as designed; it does not test the asymmetry assumption, which is the flaw).
- `Wavelock_numpy` self-test → sign/verify/tamper-reject all pass; trailing
  assertion `schema == "WLv2"` fails only because the schema is now `WLv3.1`
  (stale assertion, not a security finding).

## New scripts created

- `attacks/forge_from_snapshot.py` — forges a valid signature on an arbitrary
  message using only the published ψ★ + public params. **Primary break.**
- `attacks/seed_bruteforce.py` — recovers the seed (hence ψ★, hence full signing
  power) from the public commitment by forward search; reports honest work factor.

## Claim table

| Claim | Evidence offered | Attack attempted | Result | Confidence |
|---|---|---|---|---|
| "One-way curvature lock; ψ★ non-invertible" | `ATTACKBOUNDS.md` prose; empirical inversion tests | Showed inversion is unnecessary — forgery uses published ψ★ directly | **Claim is irrelevant to security; forgery succeeds without inversion** | High |
| "Signature / identity primitive" | `sign`/`verify` API | Forge from verification material | **Broken — symmetric MAC, not a signature** | High |
| "Verification is exact/byte-level, so shortcuts fail" (`ATTACKBOUNDS` cls 3) | prose | Used exactness in attacker's favor: exact ψ★ is published | **Argument defends wrong threat; aids forgery** | High |
| "Survives crypto breaks; identity = ψ★" (`SURVIVABILITY.md`) | prose | Noted ψ★ must be shared to verify | **ψ★-as-identity = shared secret; no survivability benefit** | High |
| "Collision resistance" | dual-hash SHA-256+SHA3 | None needed | Holds, but is just standard hashing (not novel) | High |
| "Quantum resistance (hash-based)" | prose | n/a | Plausible for the hash layer only; the PDE adds nothing | Medium |
| "Exponential difficulty to break" | prose | Seed brute force | **Difficulty = seed entropy (≤2^32 in examples), not exponential in curvature** | High |
| "Deterministic reproducibility" | XOF seed derivation; tests | Verified | True (and it is exactly what enables seed brute force) | High |

Classification: one-wayness/irreversibility claims are **unsupported as a
security property** (they defend a threat model that does not match a signature);
collision/preimage resistance is **proven but standard**; "exponential
difficulty" is **weak/unsupported**.

## Reduction to a known easier problem

WaveLock = (weak KDF: `SHAKE256(seed)` + 50 fixed explicit PDE steps, ~5 ms)
∘ (suffix-style keyed hash `H(… ‖ ψ★)` with SHA-256/SHA3-256). Both halves are
standard. It is strictly a **symmetric MAC with a slow key-derivation**, weaker
than HMAC and offering none of the asymmetry a signature requires. The novelty
is in the **framing**, not the math or the implementation.

## Bounty readiness

1. **$500 micro-bounty?** No. A reviewer will find the symmetric-MAC structure in
   minutes (it is the first thing the two scripts here show). You would be paying
   to be told what this report already shows.
2. **$2k–$5k theory bounty?** No. There is no asymmetric hardness claim left to
   attack once the verifier-holds-the-key issue is acknowledged.
3. **$10k public bounty?** No. Publishing as-is invites a public, reputational
   break, plus the repo currently commits/leaks secret material patterns.

### What must be fixed before each level

Before **any** paid review:
- Decide and implement an actual asymmetry. Either (a) reframe honestly as a
  **MAC / symmetric authenticator** and drop "one-way signature" language, or
  (b) build a real commitment-only verifier: verification must use **only the
  public commitment hash**, never ψ★. That requires a zero-knowledge / SNARK
  proof that "I know a seed whose evolution hashes to this commitment and binds
  this message" — a substantial new construction, not present today.
- Stop shipping ψ★ to verifiers; stop writing `seed`/`psi_0`/`psi_star` in
  cleartext (`cli.generate_key`, `generate_quantum_keys`, `data/wavelock_data/`).
- Remove the non-strict and unpublished-ψ★ accept-by-default paths in `server.py`.
- Enforce high-entropy seeds (≥128 bits) and remove tiny-seed examples.

For a **$2k–$5k** review: after the above, write a precise security definition
(EUF-CMA or a stated MAC game), state the hardness assumption formally, and
provide reference test vectors and a verifier that does not touch ψ★.

For a **$10k** review: a written security proof (or reduction) for the chosen
definition, plus independent reproduction of the verifier from spec alone.

## What still worries me even though I "broke" it

- The PDE itself may be genuinely hard to invert — but that is moot, because the
  scheme never needs inversion to be broken. If a future redesign moves to a
  commitment-only verifier, the PDE's irreversibility would need a **formal**
  argument; the current `ATTACKBOUNDS.md` is prose, not proof, and explicitly
  labels itself "attack-bounding arguments, not formal proofs."
- The numpy reference uses float64 with `np.roll`/`np.gradient`; cross-platform
  bit-exactness of ψ★ is assumed but only lightly tested. If verifiers ever
  recompute ψ★ (rather than load it), FP nondeterminism could cause false
  rejects — an availability risk distinct from the forgery break above.
