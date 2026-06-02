# WaveLock Hardening Recommendations

> **This document is advisory only.** It lists recommended **future** changes
> that live **outside** the `audit/` folder. **Nothing here is implemented by
> this folder, and this folder does not modify production code, serialization,
> seed handling, or backend behavior.** These are the changes that would move
> WaveLock from "audited research artifact" to "production bounty-ready M2M
> commitment / attestation / replay / drift layer."

Each recommendation cites the audit finding it addresses (see
[`REPORT.md`](./REPORT.md)).

---

## 1. Seed / input entropy (addresses C-2)

- **Require ≥128-bit** seed/input material in **production mode** (256-bit
  preferred), matching the OTS path that already uses `os.urandom`.
- **Mark integer seed mode as demo-only** — never a production input.
- **Reject the default seed `42`** (and other obvious low-entropy seeds) in
  production mode.
- Document explicitly that, for low-entropy seeds, the commitment's security
  ceiling is **seed entropy**, not SHA-256's 256 bits.

## 2. Canonical serialization (addresses H-1, and reduces C-1 surface)

Canonicalize the bytes **before** hashing:

- **Reject NaN.**
- **Reject Inf.**
- **Canonicalize `-0.0` to `+0.0`.**
- **Fixed endianness.**
- **Fixed schema** (one serialization version, not LE WLv2 vs BE WLv3).
- **Fixed dtype.**
- **Bind metadata into the hashed bytes:** kernel hash, parameters, lattice
  dimensions, step count, and XOF id — so the commitment is a function of the
  full logical state, not just raw float bytes.

## 3. Consensus backend definition (addresses C-1)

- Declare the **reference NumPy backend only** as consensus for the current
  design (or a **future fixed-point backend** if/when one exists).
- Treat **GPU / CuPy / fast-math / float-reassociation** modes as
  **research / non-consensus** unless and until they are made provably
  deterministic. The existing GPU "non-consensus" guard is the correct pattern;
  extend it rather than weaken it.

## 4. M2M security binding (architecture)

- Use **standard KEM/AEAD** for payload secrecy. WaveLock is **not** a cipher.
- Have WaveLock supply the M2M security primitives it is actually good at:
  **commitment, attestation, replay verification, drift signal, and transcript
  binding** — bound on top of the standard channel, not replacing it.

## 5. Future tests (to be added **outside** `audit/`)

These tests are recommended but **must not** be added to this audit folder; they
belong with the production test suite:

- **Seed-entropy checks** — production mode rejects sub-threshold / demo seeds.
- **Canonical-serialization tests** — NaN/Inf rejected, `-0.0`/`+0.0` unified,
  single schema/endian/dtype enforced.
- **Backend-mismatch rejection** — unsupported backends cannot emit consensus
  commitments.
- **Replay verification metadata binding** — wrong kernel/params/seed/`ψ*`/
  invariants are rejected.
- **Drift-evasion tests** — materially changed observable behavior cannot evade
  the drift signal.

---

## Mapping summary

| Recommendation | Audit finding |
|----------------|---------------|
| ≥128-bit seeds, demo-only integers, reject `42` | C-2 |
| Reject NaN/Inf, normalize signed zero, fix endian/schema/dtype, bind metadata | H-1 (and C-1) |
| NumPy-only / fixed-point consensus; GPU & fast-math non-consensus | C-1 |
| KEM/AEAD for secrecy; WaveLock for commitment/attestation/replay/drift | H-2 (PDE is cryptographically inert for diffusion) |
| Future production tests | C-1, C-2, H-1, plus replay/attestation/drift layers |
