"""
WaveLock Golden Vector Tests
============================

FROZEN CONSENSUS TEST VECTORS
----------------------------
These vectors lock the expected output of the NumPy reference implementation.
Any change to physics parameters, serialization, or evolution logic will
cause these tests to fail.

PURPOSE:
- Regression detection
- Cross-implementation validation
- Audit verification

WARNING:
If these tests fail after a code change, you have broken consensus.
Either revert the change or document it as a breaking protocol upgrade.
"""

import os
import sys
import hashlib
import pytest
import numpy as np

# ---------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------
# Import NumPy reference implementation (CONSENSUS BACKEND)
# ---------------------------------------------------------------------
from wavelock.chain.Wavelock_numpy import CurvatureKeyPairV3


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def psi_star_hash(kp: CurvatureKeyPairV3) -> str:
    """Hash the ψ★ array for comparison."""
    psi = np.asarray(kp.psi_star, dtype=np.float64, order="C")
    return sha256_hex(psi.tobytes())


def commitment_primary(kp: CurvatureKeyPairV3) -> str:
    """Extract primary hash from commitment string."""
    parts = kp.commitment.split(":")
    return parts[1] if len(parts) >= 2 else ""


# =============================================================================
# FROZEN GOLDEN VECTORS
# =============================================================================
# 
# These hashes were generated from the canonical NumPy implementation.
# They represent the AUTHORITATIVE expected output for each (n, seed) pair.
#
# FORMAT:
#   (n, seed): {
#       "psi_star_hash": SHA256 of ψ★ bytes (float64, C-order),
#       "commitment_primary": Primary hash from commitment string,
#       "schema": Expected schema version,
#   }
#
# TO REGENERATE (only if intentionally changing protocol):
#   python -c "from test_golden_vectors import generate_vectors; generate_vectors()"
#
# =============================================================================

GOLDEN_VECTORS = {
    (4, 42): {
        "psi_star_hash": "91d9e34e9eb25005e924e3606bfb46ae8a198b5d55ac7e1bf56804c7bfa220bc",
        "commitment_primary": "4d34fef529b5f7cd71da2cf0ca06b7bbe0a55d8b8000118389207cc0fe000185",
        "schema": "WLv2",
    },
    (6, 99): {
        "psi_star_hash": "f5eb4d5a474e6653b49e60284aed4194f9cf0951adcc3927fb83c8eb4c6ae64f",
        "commitment_primary": "c51b7122c644e07e595f8ce35ef45c8e6d41a1ffe0c38b9a9dcb5f3b0b42cffb",
        "schema": "WLv2",
    },
    (8, 1234): {
        "psi_star_hash": "bb4d8ef23f9ce20adcb8155f31cb11a67070a94c316517cbfad42bc0d341d3a3",
        "commitment_primary": "10c0adcc701df3ebbcbc3f63b0fc38b3eff8cd5aafb1c5cf04a286ec989f5bd0",
        "schema": "WLv2",
    },
    (4, 0): {
        "psi_star_hash": "63631cb458cf56fb28e2308b177729abc68f22959576cb09486bf9656c1bd133",
        "commitment_primary": "1129fd799ccfc4054925dfa7cb304c012338a40944667a92b3feac7edaaf2bf1",
        "schema": "WLv2",
    },
    (6, 12345): {
        "psi_star_hash": "8116f52ef6e9e14209241fb28dd8e83a51f5abe48a44b97d0925ed8b12dae51a",
        "commitment_primary": "ee3a48770e218b2382abf3d347998b1a21779b542f2e6d9d3c74a999f300bb36",
        "schema": "WLv2",
    },
}


# =============================================================================
# Vector Generation (run once to populate GOLDEN_VECTORS)
# =============================================================================

def generate_vectors():
    """
    Generate golden vectors from current implementation.
    
    Run this ONCE after freezing the implementation, then paste
    the output into GOLDEN_VECTORS above.
    """
    print("=" * 70)
    print("GOLDEN VECTOR GENERATION")
    print("=" * 70)
    print()
    print("Copy the following into GOLDEN_VECTORS:")
    print()
    print("GOLDEN_VECTORS = {")
    
    test_cases = [
        (4, 42),
        (6, 99),
        (8, 1234),
        (4, 0),
        (6, 12345),
    ]
    
    for n, seed in test_cases:
        kp = CurvatureKeyPairV3(n=n, seed=seed)
        psi_hash = psi_star_hash(kp)
        primary = commitment_primary(kp)
        schema = kp.commitment.split(":")[0]
        
        print(f"    ({n}, {seed}): {{")
        print(f'        "psi_star_hash": "{psi_hash}",')
        print(f'        "commitment_primary": "{primary}",')
        print(f'        "schema": "{schema}",')
        print("    },")
    
    print("}")
    print()
    print("=" * 70)


# =============================================================================
# Golden Vector Tests
# =============================================================================

@pytest.mark.parametrize("key", GOLDEN_VECTORS.keys())
def test_golden_vector_psi_star(key):
    """
    CONSENSUS TEST: ψ★ must match frozen golden vector.
    
    Failure indicates a consensus-breaking change.
    """
    n, seed = key
    expected = GOLDEN_VECTORS[key]
    
    if expected["psi_star_hash"] == "PLACEHOLDER_GENERATE_ME":
        pytest.skip("Golden vectors not yet generated. Run generate_vectors() first.")
    
    kp = CurvatureKeyPairV3(n=n, seed=seed)
    actual_hash = psi_star_hash(kp)
    
    assert actual_hash == expected["psi_star_hash"], (
        f"GOLDEN VECTOR MISMATCH (n={n}, seed={seed})\n"
        f"Expected ψ★ hash: {expected['psi_star_hash']}\n"
        f"Actual ψ★ hash:   {actual_hash}\n"
        f"This indicates a consensus-breaking change!"
    )


@pytest.mark.parametrize("key", GOLDEN_VECTORS.keys())
def test_golden_vector_commitment(key):
    """
    CONSENSUS TEST: Commitment primary hash must match frozen golden vector.
    
    Failure indicates a serialization or hashing change.
    """
    n, seed = key
    expected = GOLDEN_VECTORS[key]
    
    if expected["commitment_primary"] == "PLACEHOLDER_GENERATE_ME":
        pytest.skip("Golden vectors not yet generated. Run generate_vectors() first.")
    
    kp = CurvatureKeyPairV3(n=n, seed=seed)
    actual_primary = commitment_primary(kp)
    
    assert actual_primary == expected["commitment_primary"], (
        f"COMMITMENT MISMATCH (n={n}, seed={seed})\n"
        f"Expected: {expected['commitment_primary']}\n"
        f"Actual:   {actual_primary}\n"
        f"This indicates a serialization or hashing change!"
    )


@pytest.mark.parametrize("key", GOLDEN_VECTORS.keys())
def test_golden_vector_schema(key):
    """
    CONSENSUS TEST: Schema version must match frozen golden vector.
    """
    n, seed = key
    expected = GOLDEN_VECTORS[key]
    
    if expected["schema"] == "PLACEHOLDER_GENERATE_ME":
        pytest.skip("Golden vectors not yet generated.")
    
    kp = CurvatureKeyPairV3(n=n, seed=seed)
    actual_schema = kp.commitment.split(":")[0]
    
    assert actual_schema == expected["schema"], (
        f"SCHEMA MISMATCH (n={n}, seed={seed})\n"
        f"Expected: {expected['schema']}\n"
        f"Actual:   {actual_schema}"
    )


# =============================================================================
# Cross-Run Consistency Test
# =============================================================================

def test_golden_vectors_self_consistent():
    """
    Verify that generating vectors twice produces identical results.
    
    This catches any non-determinism in the implementation.
    """
    for (n, seed) in [(4, 42), (6, 99)]:
        kp1 = CurvatureKeyPairV3(n=n, seed=seed)
        kp2 = CurvatureKeyPairV3(n=n, seed=seed)
        
        h1 = psi_star_hash(kp1)
        h2 = psi_star_hash(kp2)
        
        assert h1 == h2, (
            f"NON-DETERMINISM DETECTED (n={n}, seed={seed})\n"
            f"Run 1: {h1}\n"
            f"Run 2: {h2}"
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "generate":
        generate_vectors()
    else:
        print("Usage:")
        print("  python test_golden_vectors.py generate  - Generate golden vectors")
        print("  pytest test_golden_vectors.py           - Run tests")
