# PDE-Hash Architectural Audit

**Scope:** Locate every use of a conventional cryptographic primitive in the
WaveLock repository, classify it, trace the call graph from user input to the
final commitment/signature, and identify precisely where the project drifted
from the stated research objective

> Determine whether a nonlinear PDE evolution can itself serve as a
> deterministic one-way compression function, replacing SHA/SHAKE/BLAKE in
> the primitive's security path.

**Status:** read-only audit. No behavior was modified to produce this document.
**Branch:** `research/hash-free-pde-core`.
**Date:** 2026-06-17.

---

## 0. Executive summary

The repository does **not** currently test the stated hypothesis. A nonlinear
PDE evolution exists and runs, but it is sandwiched between conventional Keccak
(SHA-3 / SHAKE-256) calls and contributes nothing to the one-wayness of the
message → digest map. Two independent facts establish this:

1. **The message never enters the PDE.** In both security paths (the
   `wavelock/chain` commitment path and the `wavelock/crypto` OTS path), the
   message is absorbed by SHAKE-256 / SHA-256 *directly*. The PDE state `ψ★`
   is derived from a **seed**, not from the message `m`.

2. **One-wayness is supplied entirely by SHA/SHAKE.** The published commitment
   is `SHA-256(serialize(ψ★))` (chain) or `SHAKE-256(quantize(ψ★))` (OTS). Even
   if the PDE map `ψ₀ ↦ ψ★` were perfectly invertible, the published value
   would remain one-way, because a standard hash wraps it. The hash is doing the
   cryptography; the PDE is decorative.

The intended pipeline

```
m  ──►  PDE  ──►  y
```

has in practice become

```
m   ─SHAKE256─►  message_digest ─┐
                                 ├─► commitment / signature  ─SHA256/SHA3/BLAKE3─► y
seed ─SHAKE256─► ψ₀ ─PDE(float64)─► ψ★ ─quantize─SHAKE256─► psi_commitment ─┘
```

i.e. exactly the drift `m → SHA/SHAKE → PDE → SHA/SHAKE → y`, and in fact more
severe: the message bypasses the PDE entirely and is hashed on a separate track.

---

## 1. Inventory of conventional-primitive call sites

Every file under `wavelock/` that references a conventional digest, XOF, MAC,
or KDF, with classification:

| File | Primitive(s) | Role in code | Classification |
|---|---|---|---|
| `wavelock/chain/xof_init.py` | SHAKE-256 (`hashlib.shake_256`) | Derives initial field `ψ₀` from a seed | **(1) inside the proposed PDE-native security path** — state initialization |
| `wavelock/chain/WaveLock.py` | SHA-256, SHA3-256, BLAKE3, HMAC (compare) | Commitment of `ψ★`; signature = hash of `message‖header‖ψ★`; ψ₀ via `xof_init` | **(1) inside the security path** — message absorption + output extraction + init |
| `wavelock/chain/hash_families.py` | SHA-256, SHA3-256, BLAKE3 | Pluggable digest registry used by the commitment/signature | **(1) inside the security path** — output extraction |
| `wavelock/chain/Wavelock_numpy.py` | (mirrors chain hashing) | NumPy reference path for consensus commitments | **(1) inside the security path** |
| `wavelock/crypto/wavelock_ots.py` | SHAKE-256 (single primitive `_h`) | ψ₀ init, psi_commitment, Lamport sk/pk derivation, message digest | **(2) downstream application** (OTS signature) — but it *is* the security path of that application |
| `wavelock/crypto/wavelock_encrypt.py` | SHAKE/SHA (KDF/stream) | Experimental encryption wrapper | **(2) downstream application** |
| `wavelock/chain/CurvaChain.py`, `Block.py`, `pov.py`, `ledger_merkle.py`, `migrate.py`, `artifacts.py`, `chain_utils.py`, `kernel_decl.py`, `cli.py` | SHA-256 / SHA3 | Block IDs, Merkle trees, ledger object identity, proof-of-validity | **(2) downstream application** + **(5) unrelated infrastructure** (Git-like object identity) |
| `wavelock/network/server.py` | SHA/HMAC | Wire-protocol integrity | **(5) unrelated infrastructure** |
| `audit/*`, `tests/*`, `attacks/*` | SHA/SHAKE | Test vectors, golden vectors, attack harnesses | **(3) test / tooling only** |

There is **no** location in the current code where a conventional hash is used
*only* for legitimately out-of-scope purposes (Git object identity,
file-integrity, fixture bookkeeping) while staying clear of the security path.
Categories (1) and (2) overlap the message-absorption / init / output stages
that the research objective requires to be hash-free.

---

## 2. Call graph: user input → final commitment / signature

### 2.1 Chain commitment path (`CurvatureKeyPair`)

```
CurvatureKeyPair(n, seed, use_vX=...)                      WaveLock.py:442
  └─ derive_psi_zero(seed, (side, side))                   WaveLock.py:506
        └─ _shake256_stream(seed_bytes, n)                 xof_init.py:41   ◀── SHAKE-256  [state init]
        └─ _bytes_to_uniform_float64(...)                  xof_init.py:50
  └─ _evolve_capture(ψ₀, n)                                WaveLock.py:657  ◀── PDE (float64, 50 steps)
        ψ_{t+1} = ψ_t + dt·(α·Lap/ψ − θ·ψ·Lap(log ψ²)) − damping·ψ
  └─ _serialize_commitment_vX(ψ★)                          WaveLock.py:263+
  └─ hashlib.sha256(raw).hexdigest()                       WaveLock.py:568  ◀── SHA-256   [output extraction]
  └─ DualHash.from_data(raw, ...)                          WaveLock.py:572  ◀── SHA3-256 / BLAKE3 [output]
  └─ self.commitment = "WLv3:<sha256>:<secondary>"         WaveLock.py:579

CurvatureKeyPair.sign(message)                             WaveLock.py:757
  └─ _sig_payload_v2(message)                              WaveLock.py:690
        raw = b"SIGv2\0" + message + b"\0" + header + b"\0" + ψ★bytes
  └─ hash_hex(payload, primary_family)                     WaveLock.py:763  ◀── SHA-256   [message absorption]
```

**Where one-wayness lives:** lines 568, 572, 763 — all SHA/SHAKE/BLAKE. The
message enters cryptography *only* at line 763, where it is concatenated with
`ψ★` and hashed. The PDE output `ψ★` acts as a fixed per-key salt, not as the
one-way transform of the message.

### 2.2 OTS signature path (`wavelock_ots.py`)

From the module's own docstring (lines 37–42) and code:

```
evolve_psi_star(seed_bytes, params)                        wavelock_ots.py:252
  └─ derive_psi_zero(seed_bytes, (side, side))             :264   ◀── SHAKE-256 [state init]
  └─ float64 PDE, `steps` iterations                       :265-270  ◀── PDE
psi_commitment(ψ★)                                          :285
  └─ _quantize_psi(ψ★)  → _h(b"WL-PSI-COMMIT", ...)         :287   ◀── SHAKE-256 [output extraction]
_secret_slice(seed, psi_commit, p_hash, i, b)              :295
  └─ _h(b"WL-OTS-SK-v1", seed, psi_commit, p_hash, i, b)   :304   ◀── SHAKE-256 [key derivation]
_message_digest(message, p_hash)                           :343
  └─ _h(b"WL-OTS-MSG-v1", message, p_hash)                 :345+  ◀── SHAKE-256 [message absorption]
```

`_h` (line 172) is a single SHAKE-256 wrapper used **everywhere**: init, commit,
secret-key derivation, and message digest. The Lamport bit selection is driven
by `_message_digest`, which is a pure SHAKE-256 of the message. The PDE state
only perturbs the secret-key derivation as one input among several.

---

## 3. Drift analysis — exactly where the project left the hypothesis

| Stage the hypothesis requires to be PDE-native | What the code actually does | Verdict |
|---|---|---|
| `A`: absorb message → state | Message is SHAKE-256'd (OTS) or concatenated-then-SHA'd (chain). It **never touches `ψ`.** | **Drifted.** Absorption is a conventional hash. |
| State init | `ψ₀ = SHAKE-256(seed)` | **Drifted.** Init is a conventional XOF. |
| `Φ`: PDE evolution | Genuine float64 PDE runs (50 steps). | Present, but disconnected from `m`. |
| `Q`: squeeze → 256 bits | `SHA-256/SHAKE-256(serialize(ψ★))` | **Drifted.** Squeeze is a conventional hash. |
| Round constants / params | `_kernel_hash = SHA-256(params)` (`WaveLock.py:121`) | **Drifted.** Parameter fingerprint is SHA-256. |

**Root cause.** The security argument was migrated onto SHA/SHAKE for
reproducibility and "survivability" (see `hash_families.py` docstring, which
explicitly states *"Wavelock's security comes from the PHYSICS … The hash is
just a binding mechanism"* — but the code does the opposite: the hash binds and
also provides all the one-wayness, while the physics is a salt). Float64 PDE
output is not byte-reproducible across backends, so a hash was wrapped around it
to stabilize the commitment; once a hash wraps the output, the PDE's
cryptographic contribution becomes unobservable.

**Consequence for the research question.** As built, removing the PDE entirely
(replacing `ψ★` with any fixed per-key constant) would leave a conventional
SHA/SHAKE-based commitment and a SHAKE-based Lamport OTS — both still "secure"
in the conventional sense. Therefore the current system **cannot** answer
whether a PDE can replace a hash; it has already answered "we used a hash."

---

## 4. What will be isolated (and left untouched)

The restored primitive will live only in `wavelock/pde_hash/` (implementation)
and `pde_audit/` (adversarial tests). The following are classified as legacy /
downstream / infrastructure and **will not be modified, deleted, or relabeled**
as the PDE-native primitive:

- **Legacy primitive + ledger:** all of `wavelock/chain/` (incl. `WaveLock.py`,
  `Wavelock_numpy.py`, `hash_families.py`, `xof_init.py`, `CurvaChain.py`, …).
- **Downstream applications:** `wavelock/crypto/` (OTS, encryption, ledger, CLI).
- **Unrelated infrastructure:** `wavelock/network/`, `wavelock/storage/`.
- **Existing test/attack tooling:** `audit/`, `tests/`, `attacks/`.

No new conventional cryptographic primitive will be added anywhere. The new
primitive will not be connected to CurvaChain until it has been tested in
isolation.

---

## 5. Audit conclusion

The drift is unambiguous and is documented above with file/line citations. The
PDE is real but cryptographically inert with respect to the message. Restoring
the hypothesis requires a *native* `H_PDE(m) = Q(Φ(A(m)))` in which the message
drives the PDE state and the 256-bit output is read directly from the evolved
state — with no SHA/SHAKE/BLAKE in absorption, initialization, evolution, round
constants, or squeeze. That construction is specified in
`docs/PDE_HASH_SPEC.md` (Design A) and will be paired with an independent
exact-arithmetic translation of the original curvature-feedback dynamics
(Design B), tested separately.
