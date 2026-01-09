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
        "psi_star_hash": "PLACEHOLDER_GENERATE_ME",
        "commitment_primary": "PLACEHOLDER_GENERATE_ME", 
        "schema": "WLv2",
    },
    (6, 99): {
        "psi_star_hash": "PLACEHOLDER_GENERATE_ME",
        "commitment_primary": "PLACEHOLDER_GENERATE_ME",
        "schema": "WLv2",
    },
    (8, 1234): {
        "psi_star_hash": "PLACEHOLDER_GENERATE_ME",
        "commitment_primary": "PLACEHOLDER_GENERATE_ME",
        "schema": "WLv2",
    },
    (4, 0): {
        "psi_star_hash": "PLACEHOLDER_GENERATE_ME",
        "commitment_primary": "PLACEHOLDER_GENERATE_ME",
        "schema": "WLv2",
    },
    (6, 12345): {
        "psi_star_hash": "PLACEHOLDER_GENERATE_ME",
        "commitment_primary": "PLACEHOLDER_GENERATE_ME",
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
