"""
Tests for the hardened BLAKE3 path.

The previous implementation silently fell back to BLAKE2b when the blake3
package was missing — that fallback masquerades as BLAKE3 and is a §112
enablement gap for any patent claim that names BLAKE3 specifically.

This test exercises both branches:
  - blake3 installed: hash matches the official implementation.
  - blake3 missing  : hash_data(..., HashFamily.BLAKE3) raises RuntimeError.
"""

import importlib
import sys

import pytest

from wavelock.chain.hash_families import (
    HashFamily,
    hash_data,
    is_blake3_available,
)
import wavelock.chain.hash_families as hf


def test_sha256_and_sha3_always_work():
    digest_sha256 = hash_data(b"x", HashFamily.SHA256)
    digest_sha3 = hash_data(b"x", HashFamily.SHA3_256)
    assert len(digest_sha256) == 32
    assert len(digest_sha3) == 32
    assert digest_sha256 != digest_sha3


def test_blake3_when_available_matches_reference():
    if not is_blake3_available():
        pytest.skip("blake3 package not installed")
    import blake3 as ref
    digest = hash_data(b"hello world", HashFamily.BLAKE3)
    assert digest == ref.blake3(b"hello world").digest()


def test_blake3_missing_raises_not_silently_falls_back(monkeypatch):
    # Force the cached availability flag to False and clear the module
    # reference, simulating an environment where blake3 is uninstalled.
    monkeypatch.setattr(hf, "_BLAKE3_AVAILABLE", False, raising=False)
    monkeypatch.setattr(hf, "_BLAKE3_MODULE", None, raising=False)

    with pytest.raises(RuntimeError, match="BLAKE3"):
        hash_data(b"anything", HashFamily.BLAKE3)


def test_blake3_end_to_end_through_keypair_commitment():
    # End-to-end: construct a consensus-grade keypair with BLAKE3 as the
    # secondary family, compute the commitment, and verify both halves
    # round-trip. This is the test that lifts BLAKE3 from "primitive only"
    # to "exercised in the commitment-generation path the patent describes."
    if not is_blake3_available():
        pytest.skip("blake3 package not installed")

    from wavelock.chain.Wavelock_numpy import CurvatureKeyPairV3
    from wavelock.chain.hash_families import hash_hex

    kp = CurvatureKeyPairV3(
        n=4,
        seed=42,
        primary_family=HashFamily.SHA256,
        secondary_family=HashFamily.BLAKE3,
    )

    primary_ok, secondary_ok = kp.verify_commitment()
    assert primary_ok, "SHA-256 primary half failed to verify"
    assert secondary_ok, "BLAKE3 secondary half failed to verify"

    # The secondary hash inside the commitment string must equal the
    # BLAKE3 of the serialized ψ★ payload — not BLAKE2b, not SHA3.
    parts = kp.commitment.split(":")
    assert len(parts) == 3, f"expected dual-hash commitment, got {kp.commitment!r}"
    secondary_hex_in_commitment = parts[2]
    expected_blake3 = hash_hex(kp._serialized, HashFamily.BLAKE3)
    assert secondary_hex_in_commitment == expected_blake3

    # Same seed must reproduce the same BLAKE3-bound commitment.
    kp2 = CurvatureKeyPairV3(
        n=4,
        seed=42,
        primary_family=HashFamily.SHA256,
        secondary_family=HashFamily.BLAKE3,
    )
    assert kp.commitment == kp2.commitment


def test_blake3_dual_signature_through_keypair():
    # The signing path also has to flow through BLAKE3 when it's selected
    # as the primary or secondary family — otherwise BLAKE3 is "selectable
    # for commitments but never actually signed with."
    if not is_blake3_available():
        pytest.skip("blake3 package not installed")

    from wavelock.chain.Wavelock_numpy import CurvatureKeyPairV3

    kp = CurvatureKeyPairV3(
        n=4,
        seed=7,
        primary_family=HashFamily.SHA256,
        secondary_family=HashFamily.BLAKE3,
    )

    sig_primary, sig_secondary = kp.sign_dual("integration-test-message")
    assert sig_primary != sig_secondary
    assert kp.verify_strict("integration-test-message", sig_primary, sig_secondary)
    assert not kp.verify_strict("tampered-message", sig_primary, sig_secondary)
