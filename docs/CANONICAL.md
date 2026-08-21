# Canonical Reference Implementation Map

This document identifies the current **engineering reference implementations**
used by the WaveLock repository. It is not a patent claim chart, prosecution
statement, legal opinion, or limitation on any pending or issued patent right.
See `PATENT_NOTICE.md`, `LICENSE`, and `docs/PATENT_SCOPE.md`.

The repository is intentionally redundant: multiple implementations exist for
performance, reproducibility, archival, and adversarial-testing purposes. This
map identifies which implementation is currently used as the software reference
for each layer.

---

## 1. Curvature-Regulated PDE Operator F

| Layer | Reference file | Engineering status |
| --- | --- | --- |
| Reference (CPU) | `wavelock/chain/Wavelock_numpy.py` | Canonical CPU reference |
| Production-oriented (GPU) | `wavelock/chain/WaveLock.py` | GPU implementation |
| Test harness | `tests/scientific/test_wavelock_jacob.py` | Research surrogate |

The intended operator form is:

```text
F(ψ) = α · Δψ / D(ψ) − θ · ψ · Δlog(ψ² + δ) − μ · ψ
D(ψ) = ψ + ε · exp(−β · ψ²)
```

The discrete update is of the form:

```text
ψ_(t+1) = ψ_t + Δt · F(ψ_t)
```

The centered discrete Laplacian uses the four-neighbor periodic stencil in two
dimensions (`np.roll` / `cp.roll`, shifts ±1 along axes 0 and 1).

The PyTorch implementation in `tests/scientific/test_wavelock_jacob.py` is a
research surrogate for gradient-based attack experiments. Differences between
a research surrogate and the canonical implementation should be documented as
engineering differences rather than treated as statements about patent scope.

### Reproducibility rule

When deterministic cross-implementation commitments are required, the
reference path should use byte-stable state initialization, canonical
serialization, and explicit kernel metadata. GPU reduction order can vary
across hardware generations, so consensus-style commitments should be generated
through a byte-stable reference path or verified against published tolerances
where the protocol expressly permits tolerances.

---

## 2. Initial-State Derivation

| Mode | Reference file | Intended use |
| --- | --- | --- |
| SHAKE-256 XOF | `wavelock/chain/xof_init.py` | Cross-implementation deterministic initialization |
| Legacy backend RNG | `np.random` / `cp.random` seeded by integer | Historical/local tests |

SHAKE-256 is used with a WaveLock domain-separation tag to derive deterministic
`ψ₀` values from seed material. The legacy backend RNG path remains for
backward-compatible testing and historical vectors.

---

## 3. Hash Families and Dual-Hash Commitment

Canonical implementation: `wavelock/chain/hash_families.py`.

- SHA-256: `hashlib.sha256`.
- SHA3-256: `hashlib.sha3_256`.
- BLAKE3: official `blake3` Python package when installed.

The former BLAKE2b fallback under a BLAKE3 label was removed because it produced
digests from a different algorithm while presenting them as BLAKE3. The current
implementation fails closed with `RuntimeError` if BLAKE3 is selected but the
BLAKE3 package is unavailable.

This is an interoperability and correctness rule. It is not a legal conclusion
about any patent application.

---

## 4. Ledger Record Merkle Root

Canonical implementation: `wavelock/chain/ledger_merkle.py`.

A ledger record's Merkle root binds, in fixed order, the repository's current
record fields including:

1. wavefield commitment;
2. operator parameters;
3. kernel descriptor;
4. curvature invariants;
5. timestamp; and
6. linkage to the prior record hash.

Each leaf is derived from canonical JSON and hashed before construction of the
binary Merkle tree.

This record-level Merkle root is distinct from
`Block.calculate_merkle_root()` in `wavelock/chain/Block.py`, which binds the
ordered `messages` list of a chain block. A record may be carried as a message
inside a chain block, producing an intentional layered binding.

---

## 5. CurvaChain Block Structure

Current reference files:

- `wavelock/chain/Block.py`
- `wavelock/chain/CurvaChain.py`
- `wavelock/chain/chain_utils.py`

The current block representation includes fields such as `index`, `timestamp`,
`previous_hash`, `merkle_root`, `difficulty`, `nonce`, `block_type`, `meta`, and
`block_hash`.

---

## 6. One-Time Signatures, Replay State, and Protocol Binding

Current implementation work is located primarily under:

- `wavelock/crypto/`
- `wavelock/network/server.py`
- `docs/WAVELOCK_OTS_DESIGN.md`
- `docs/WAVELOCK_ENCRYPT_SECURITY_NOTE.md`

These modules experiment with public verification, one-time-use identities,
canonical block-body binding, durable replay rejection, accepted-chain replay
reconstruction, and authenticated protocol context.

Security status labels in those files describe the tested software only. They
do not narrow or disclaim the separate patent disclosure.

---

## 7. Drift Detection and Attestation

The pending WaveLock patent application includes drift-detection and attestation
embodiments. This public Python repository is not an exhaustive implementation
of every embodiment described in the filed application.

Any drift-detection implementation or validation artifact in this or another
repository should be documented in terms of its actual code, inputs, tests, and
measured behavior. The presence, absence, maturity, success, or failure of a
particular repository implementation is an engineering fact and is not, by
itself, a legal conclusion about patent support, validity, enforceability, or
scope.

---

## 8. Repository Boundaries

This repository contains the Python research/reference layer of WaveLock.
References to separate C, kernel, hardware, or experimental repositories refer
to distinct implementation trees and should be audited in their own context.

No file in this engineering map grants a patent license or changes the rights
reservation in `LICENSE` and `PATENT_NOTICE.md`.
