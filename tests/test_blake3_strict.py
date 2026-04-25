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
