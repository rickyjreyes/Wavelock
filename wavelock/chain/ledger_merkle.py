"""
Merkle-root ledger-record binding (Claim 15).

A WaveLock ledger record binds:
  (i)   the wavefield commitment (schema:primary_hex:secondary_hex)
  (ii)  operator parameters (alpha, beta, theta, epsilon, delta)
  (iii) kernel descriptor (kernel_version, kernel_hash)
  (iv)  curvature invariants (E_grad, E_fb, E_ent, E_tot)
  (v)   timestamp / monotonic counter
  +     a hash of the prior record (chain linkage)

Claim 15 specifies a Merkle root computed over fields (i)–(v) AND over a
hash of the prior record. This module computes that Merkle root explicitly
so the implementation matches the claim language verbatim.

The existing Block.calculate_merkle_root() in Block.py merkles the
`messages` list; that is a different leaf set and serves a different
purpose. This module is layered on top — a record's Merkle root is one of
the messages stored in a Block, and the Block's own Merkle covers the
ordered messages.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence
import hashlib
import json


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _h(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _merkle_root(leaves: Sequence[bytes]) -> bytes:
    """
    Standard binary Merkle root over an ordered list of pre-hashed leaves.

    Empty leaf list returns SHA256(""). Odd levels duplicate the last node,
    matching Bitcoin's convention (and Block.calculate_merkle_root).
    """
    if not leaves:
        return _h(b"")
    layer: List[bytes] = list(leaves)
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [_h(layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return layer[0]


def record_leaves(
    *,
    commitment: str,
    operator_params: dict,
    kernel_descriptor: dict,
    curvature_invariants: dict,
    timestamp: str,
    prior_record_hash: Optional[str],
) -> List[bytes]:
    """
    Build the canonical ordered leaf list for a ledger record.

    Field order is fixed (matches Claim 15 ordering i..v + prior-record
    hash). Each leaf is SHA-256 of the canonical-JSON encoding of a tagged
    field, so any reorder or substitution shows up as a different Merkle
    root.
    """
    if prior_record_hash is None:
        prior_record_hash = ""

    fields = [
        ("commitment", commitment),
        ("operator_params", operator_params),
        ("kernel_descriptor", kernel_descriptor),
        ("curvature_invariants", curvature_invariants),
        ("timestamp", str(timestamp)),
        ("prior_record_hash", prior_record_hash),
    ]
    return [_h(_canonical_json({"field": name, "value": value})) for name, value in fields]


def compute_record_merkle_root(
    *,
    commitment: str,
    operator_params: dict,
    kernel_descriptor: dict,
    curvature_invariants: dict,
    timestamp: str,
    prior_record_hash: Optional[str],
) -> str:
    """
    Compute the Merkle root binding ledger-record fields (i)-(v) + prior
    record hash. Returns hex-encoded SHA-256 of the root node.
    """
    leaves = record_leaves(
        commitment=commitment,
        operator_params=operator_params,
        kernel_descriptor=kernel_descriptor,
        curvature_invariants=curvature_invariants,
        timestamp=timestamp,
        prior_record_hash=prior_record_hash,
    )
    return _merkle_root(leaves).hex()


def verify_record_merkle_root(
    *,
    commitment: str,
    operator_params: dict,
    kernel_descriptor: dict,
    curvature_invariants: dict,
    timestamp: str,
    prior_record_hash: Optional[str],
    expected_root_hex: str,
) -> bool:
    """Constant-prefix verifier: recompute and compare."""
    computed = compute_record_merkle_root(
        commitment=commitment,
        operator_params=operator_params,
        kernel_descriptor=kernel_descriptor,
        curvature_invariants=curvature_invariants,
        timestamp=timestamp,
        prior_record_hash=prior_record_hash,
    )
    return computed == expected_root_hex


__all__ = [
    "compute_record_merkle_root",
    "verify_record_merkle_root",
    "record_leaves",
]


if __name__ == "__main__":
    # Smoke test
    args = dict(
        commitment="WLv2:abc:def",
        operator_params={"alpha": 1.5, "beta": 0.0026, "theta": 1e-5,
                         "epsilon": 1e-12, "delta": 1e-12},
        kernel_descriptor={"kernel_version": "WL-psi-001",
                           "kernel_hash": "deadbeef"},
        curvature_invariants={"E_grad": 1.0, "E_fb": 2.0,
                              "E_ent": 3.0, "E_tot": 6.0},
        timestamp="1700000000.0",
        prior_record_hash=None,
    )
    root1 = compute_record_merkle_root(**args)
    root2 = compute_record_merkle_root(**args)
    assert root1 == root2, "non-deterministic Merkle root"
    assert verify_record_merkle_root(**args, expected_root_hex=root1)

    args2 = {**args, "prior_record_hash": "00" * 32}
    root3 = compute_record_merkle_root(**args2)
    assert root1 != root3, "prior-record-hash linkage must change root"

    args3 = {**args, "operator_params": {**args["operator_params"], "alpha": 1.51}}
    root4 = compute_record_merkle_root(**args3)
    assert root1 != root4, "operator-param mutation must change root"

    print("ledger_merkle smoke test passed")
    print("root with no prior:", root1)
    print("root with prior   :", root3)
