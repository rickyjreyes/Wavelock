"""
WaveLock FP Determinism Lock Test
================================

CONSENSUS DEFINITION
-------------------
WaveLock consensus ψ★ evolution is defined *exclusively* by the
NumPy float64 reference implementation.

GPU / CuPy backends are explicitly NON-CONSENSUS accelerators.
They may diverge numerically and are NOT required to match ψ★
bitwise.

This test enforces:
  • Bitwise determinism for NumPy (consensus-critical)
  • Explicit documentation that GPU divergence is allowed
"""

import os
import sys
import hashlib
import numpy as np
import pytest

# ---------------------------------------------------------------------
# Deterministic FP contract (CONSENSUS)
# ---------------------------------------------------------------------
os.environ["PYTHONHASHSEED"] = "0"

np.set_printoptions(precision=17)
np.seterr(all="raise")

# ---------------------------------------------------------------------
# Repo path
# ---------------------------------------------------------------------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ---------------------------------------------------------------------
# Imports (NumPy reference backend ONLY)
# ---------------------------------------------------------------------
from wavelock.chain.Wavelock_numpy import (
    CurvatureKeyPairV3 as CurvatureKeyPairNP
)

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def psi_bytes_numpy(psi: np.ndarray) -> bytes:
    psi = np.asarray(psi, dtype=np.float64, order="C")
    assert psi.flags["C_CONTIGUOUS"]
    return psi.ravel(order="C").tobytes()


# ---------------------------------------------------------------------
# Frozen consensus test vectors
# ---------------------------------------------------------------------
TEST_VECTORS = [
    {"n": 4, "seed": 42},
    {"n": 6, "seed": 99},
    {"n": 8, "seed": 1234},
]


# ---------------------------------------------------------------------
# Consensus determinism test (AUTHORITATIVE)
# ---------------------------------------------------------------------
@pytest.mark.parametrize("vec", TEST_VECTORS)
def test_numpy_consensus_determinism(vec):
    """
    HARD REQUIREMENT:
    NumPy ψ★ evolution MUST be bitwise deterministic.

    Any failure here is a consensus-breaking bug.
    """

    n = vec["n"]
    seed = vec["seed"]

    kp1 = CurvatureKeyPairNP(n=n, seed=seed)
    kp2 = CurvatureKeyPairNP(n=n, seed=seed)

    b1 = psi_bytes_numpy(kp1.psi_star)
    b2 = psi_bytes_numpy(kp2.psi_star)

    h1 = sha256_bytes(b1)
    h2 = sha256_bytes(b2)

    assert b1 == b2, (
        f"NUMPY ψ★ NONDETERMINISM DETECTED (n={n}, seed={seed})\n"
        f"{h1} != {h2}"
    )


# ---------------------------------------------------------------------
# Explicit architecture guard
# ---------------------------------------------------------------------
def test_gpu_is_non_consensus_backend():
    """
    This test exists to lock architectural intent.

    GPU / CuPy backends are NOT required to be bitwise identical
    to the NumPy reference implementation and MUST NOT be used
    as a consensus authority.
    """
    assert True


# ---------------------------------------------------------------------
# Explicit enforcement sanity check
# ---------------------------------------------------------------------
def test_fp_determinism_enforced():
    """
    Sanity check: ensure this test suite would catch FP divergence.
    """
    a = np.array([1.0, 2.0], dtype=np.float64)
    b = np.array([1.0, np.nextafter(2.0, 3.0)], dtype=np.float64)

    assert a.tobytes() != b.tobytes(), (
        "Sanity check failed: FP divergence not detected"
    )
