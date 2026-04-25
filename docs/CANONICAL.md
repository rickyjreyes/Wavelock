# Canonical Implementation Map

This document declares which file, in which repository, is the **canonical
implementation** of each layer of WaveLock. Examiners, licensees, and
auditors should read this document first.

The repo hierarchy is intentionally redundant — multiple implementations
exist for performance and reproducibility reasons (CPU vs GPU, reference
vs production, archival vs live). This map disambiguates which one binds.

---

## 1. The PDE Operator F (Group I, Claims 1–4)

| Layer            | Canonical file                                       | Status     |
| ---------------- | ---------------------------------------------------- | ---------- |
| Reference (CPU)  | `wavelock/chain/Wavelock_numpy.py`                   | Canonical  |
| Production (GPU) | `wavelock/chain/WaveLock.py`                         | Canonical  |
| Test harness     | `tests/scientific/test_wavelock_jacob.py` (PyTorch)  | Surrogate  |

Both canonical files implement the same operator F:

    F(ψ) = α · Δψ / D(ψ) + θ · ψ · Δlog(ψ² + δ) − μ · ψ
    where  D(ψ) = ψ + ε · exp(−β · ψ²)

Discrete Laplacian: −4·center + up + down + left + right with periodic
wraparound (`np.roll` / `cp.roll`, shifts ±1 along axes 0 and 1).

The PyTorch surrogate in `tests/scientific/test_wavelock_jacob.py` is a
**research surrogate** for backprop-attack experiments only. It is NOT a
consensus reference. Differences between the surrogate and the canonical
operator are expected and not bugs.

### Consensus rule

Per `CurvatureKeyPair.__init__` in `WaveLock.py` (lines ~424–436): when
not in `test_mode`, the GPU backend refuses to emit consensus commitments
for any schema except WLv2. WLv3+ commitments must originate from the
NumPy reference. This rule exists because CuPy's reduction order is not
byte-stable across hardware generations.

---

## 2. ψ₀ Derivation (Claim 9)

| Mode       | Canonical file                          | Required for          |
| ---------- | --------------------------------------- | --------------------- |
| Consensus  | `wavelock/chain/xof_init.py` (SHAKE-256) | Reproducible commits  |
| Legacy     | `np.random` / `cp.random` seeded by int  | Local-only test runs  |

Pass `use_xof_init=True` to `CurvatureKeyPairV3` to select the consensus
path. The legacy path is retained for backward compatibility with
existing tests; new commitments that must verify on independent hardware
should use the XOF path.

---

## 3. Hash Families and Dual-Hash Commitment (Claims 7–8)

Canonical: `wavelock/chain/hash_families.py`.

- SHA-256: `hashlib.sha256` (CPython stdlib).
- SHA3-256: `hashlib.sha3_256` (CPython stdlib).
- BLAKE3: official `blake3` PyPI package. **No silent fallback.** If the
  `blake3` package is not installed, `HashFamily.BLAKE3` raises
  `RuntimeError`. Install with `pip install wavelock[blake3]`.

The previous BLAKE2b fallback has been removed — it produced digests that
would have masqueraded as BLAKE3 in commitments, which is both a
correctness hazard and a §112 enablement gap.

---

## 4. Ledger Record Merkle Root (Claim 15)

Canonical: `wavelock/chain/ledger_merkle.py`.

A ledger record's Merkle root binds, in fixed order:

  1. The wavefield commitment string (`schema:primary_hex:secondary_hex`).
  2. Operator parameters {α, β, θ, ε, δ}.
  3. Kernel descriptor {kernel_version, kernel_hash}.
  4. Curvature invariants {E_grad, E_fb, E_ent, E_tot}.
  5. Timestamp.
  +. Hash of the prior record (linkage).

Each leaf is `SHA-256(canonical_json({"field": name, "value": value}))`.
The internal Merkle tree is binary, with the standard duplicate-last-node
rule for odd levels.

This is distinct from `Block.calculate_merkle_root()` in
`wavelock/chain/Block.py`, which Merkles the ordered `messages` list of a
chain block. A ledger record's Merkle root is one of the messages stored
in a chain block; the block then Merkles its messages on top.

---

## 5. Chain Block Structure (CurvaChain)

Canonical: `wavelock/chain/Block.py`, `wavelock/chain/CurvaChain.py`,
`wavelock/chain/chain_utils.py`.

Each block stores: `index`, `timestamp`, `previous_hash`,
`merkle_root` (over messages), `difficulty`, `nonce`, `block_type`,
`meta`, `block_hash` (covers all of the above).

---

## What this map deliberately excludes

This repository contains the **Python implementation layer** of WaveLock.
References in supporting analyses to a separate `wavelock-kernel/` C
implementation (`archive/wavelock.c`, `src/wavelock_evolve.c`,
`crypto/blake3.c`, etc.) refer to a different repository. If you are
auditing the C layer, audit it in its own repo. The Python layer above
is independent and self-contained.

Group III drift-detection apparatus (Claims 21–27) is **not implemented
in this repo**. See `docs/PATENT_SCOPE.md` for the scope decision.
