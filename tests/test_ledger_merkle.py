"""
Tests for the Claim-15 ledger-record Merkle root.

These tests pin the binding behavior: any mutation to a bound field, or to
the prior-record-hash linkage, must change the Merkle root.
"""

import pytest

from wavelock.chain.ledger_merkle import (
    compute_record_merkle_root,
    verify_record_merkle_root,
    record_leaves,
)


BASE = dict(
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


def test_root_is_deterministic():
    r1 = compute_record_merkle_root(**BASE)
    r2 = compute_record_merkle_root(**BASE)
    assert r1 == r2


def test_root_verifies():
    r = compute_record_merkle_root(**BASE)
    assert verify_record_merkle_root(**BASE, expected_root_hex=r)


def test_root_changes_with_prior_record_hash():
    r_none = compute_record_merkle_root(**BASE)
    r_prior = compute_record_merkle_root(**{**BASE, "prior_record_hash": "00" * 32})
    assert r_none != r_prior


@pytest.mark.parametrize("field,mutation", [
    ("commitment", "WLv2:xxx:yyy"),
    ("timestamp", "1700000001.0"),
])
def test_root_changes_when_field_changes(field, mutation):
    r_base = compute_record_merkle_root(**BASE)
    r_mut = compute_record_merkle_root(**{**BASE, field: mutation})
    assert r_base != r_mut


def test_root_changes_when_operator_param_changes():
    r_base = compute_record_merkle_root(**BASE)
    mutated_params = {**BASE["operator_params"], "alpha": 1.51}
    r_mut = compute_record_merkle_root(**{**BASE, "operator_params": mutated_params})
    assert r_base != r_mut


def test_root_changes_when_kernel_hash_changes():
    r_base = compute_record_merkle_root(**BASE)
    mutated_kd = {**BASE["kernel_descriptor"], "kernel_hash": "cafebabe"}
    r_mut = compute_record_merkle_root(**{**BASE, "kernel_descriptor": mutated_kd})
    assert r_base != r_mut


def test_leaf_count_matches_claim_15_fields():
    # Claim 15 binds five record fields plus the prior-record-hash linkage.
    leaves = record_leaves(**BASE)
    assert len(leaves) == 6
