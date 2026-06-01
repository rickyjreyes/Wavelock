# WaveLock-Merkle Roadmap (many-signature support)

> **Status: PARTIAL.** The *many-signature* (Merkle tree of OTS keys) design in
> this document is still DESIGN / NOT IMPLEMENTED. The deployment-layer
> duplicate-use rejection it calls for **is now implemented and wired into block
> acceptance** for the single-key case: a durable replay ledger
> (`PersistentOTSReplayLedger`) rejects a reused `one_time_key_id`/leaf at
> `server.try_accept_block` (see `tests/test_ots_consensus.py`). What remains is
> cross-node consensus replication of that ledger and the multi-signature tree.
> Do not treat WaveLock-OTS as safe for repeated signing or value transfer.

## Why this is needed

WaveLock-OTS is plain Lamport-style OTS. Two security facts drive this roadmap:

- **Finding C (inherent).** Signing two different messages with the same key
  reveals secret slices for both bit patterns; after enough signatures an
  attacker holds *both* halves of every digest bit and can forge any message.
  This is inherent to Lamport-style OTS and is **not** fixed by the A/B
  hardening (strict canonical fields, fingerprint binding). The only defenses
  are: never reuse a key, and reject duplicate key/leaf usage at the verifier.

- **Finding D (deployment).** The local `used=true` flag and the host-local
  atomic key-state registry only protect a single host. A secret key copied to
  another host, or a wiped registry, bypasses them. Cryptographic enforcement
  must live where consensus lives.

## WaveLock-Merkle construction (XMSS/LMS-style)

```
                         merkle_root  (the long-lived PUBLIC KEY)
                        /            \
                  node               node
                 /     \            /     \
            leaf_0   leaf_1     leaf_2   leaf_3      ...   leaf_{2^h - 1}
              |        |          |        |
           OTS pk_0  OTS pk_1  OTS pk_2  OTS pk_3   (one WaveLock-OTS key each)
```

- **Merkle-root public key.** One published 32-byte root authorizes `2^h`
  one-time signatures. The root is the stable identity; individual OTS public
  keys are *not* published up front.
- **Leaf index.** Each signature consumes exactly one leaf index `j`
  (`0 ≤ j < 2^h`), strictly increasing / never repeated for a given signer
  (stateful, like XMSS), or chosen and recorded (stateless variants use a
  hyper-tree + few-time keys; out of scope here).
- **OTS public key per leaf.** Leaf `j` is the WaveLock-OTS public key
  `pk_j` (its canonical fingerprint) derived from the seed and the leaf index.
- **Authentication path.** A signature carries `(j, OTS_sig_j, pk_j,
  auth_path_j)` where `auth_path_j` is the `h` sibling hashes from `leaf_j` up
  to the root.
- **Verifier checks leaf under root.** The verifier (1) verifies the
  WaveLock-OTS signature against `pk_j` using the strict canonical checks
  already implemented in `verify_ots`; (2) recomputes `leaf_j = H(pk_j)` and
  folds it up the `auth_path` to a candidate root; (3) requires that candidate
  to equal the published `merkle_root`. All three must pass, fail-closed.
- **Ledger/server records consumed leaves and rejects duplicates.** Consensus
  records each consumed `(merkle_root, leaf_index)` (and/or `one_time_key_id`).
  A second signature reusing a leaf index — or, today, a duplicate
  `one_time_key_id` — is rejected as a replay. This is the load-bearing control
  that turns "one-time" from advisory file state into an enforced invariant.

## What exists today vs. what is missing

| Piece | Status |
|-------|--------|
| Single WaveLock-OTS key (strict, fingerprint-bound) | **implemented** (`wavelock/crypto/wavelock_ots.py`) |
| Merkle root over one key's `pk_commitments` | implemented (binds that key's commitments) |
| Domain-separated leaf/node hashing | implemented (`WL-OTS-MERKLE-LEAF-v1`, `WL-OTS-MERKLE-NODE-v1`) |
| Tree of many OTS keys (root → leaves) | **NOT implemented** |
| Leaf index + authentication path in signatures | **NOT implemented** |
| Stateful signer that never reuses a leaf | **NOT implemented** |
| Durable, reconstructable consumed-key/leaf ledger | **implemented** (`wavelock/crypto/ots_ledger.py`, `PersistentOTSReplayLedger`: append-only + fsync, rebuildable from accepted blocks; the in-memory `OTSReplayLedger` model is retained for the standalone entry point) |
| OTS verification wired into block acceptance | **implemented** (`server.try_accept_block` → `_verify_ots_block`; durable replay rejection; legacy SIGv2 refused on the OTS path) |
| Multi-node consensus replication of the ledger | **NOT implemented** (the ledger is canonical *per node*; cross-node agreement is the remaining gap) |

## Deployment requirement (restate, loudly)

The durable replay ledger now rejects a duplicate `one_time_key_id`/leaf at
block acceptance on a node, so a copied secret key cannot get a second OTS block
accepted there — Finding D is **fixed at the ledger/consensus layer** for that
node. The honest caveats that remain:

- Finding D is only *fully* closed when **every** accepting node runs this
  rejection against a ledger derived from agreed chain state. A single isolated
  node with no shared chain state still only protects itself.
- The host-local signing registry (`_claim_one_time_key`) is **defense-in-depth
  only** — never the primary control.
- WaveLock-OTS is still single-time Lamport (Finding C is inherent) and is
  **not** safe for value transfer or repeated signing. Do not treat it as
  production-ready.
