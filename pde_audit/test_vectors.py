"""Pinned deterministic test vectors for WaveLock-PDE-256-v0.

Both implementations must reproduce the frozen vectors in vectors.json
byte-for-byte. A change here means the primitive changed (version bump).
"""

import json
import os

import pytest

from wavelock.pde_hash import reference, optimized
from wavelock.pde_hash.state import initial_state

VECTORS_PATH = os.path.join(os.path.dirname(__file__), "vectors.json")

with open(VECTORS_PATH) as f:
    VECTORS = json.load(f)

IMPLS = {"reference": reference, "optimized": optimized}


def test_version_tag():
    assert VECTORS["version"] == "WaveLock-PDE-256-v0"


def test_iv_cells():
    iv = initial_state()
    assert iv.cells[0] == VECTORS["iv_cells"]["0_0"]
    assert iv.cells[1] == VECTORS["iv_cells"]["0_1"]
    assert iv.cells[255] == VECTORS["iv_cells"]["15_15"]


@pytest.mark.parametrize("impl_name", list(IMPLS))
@pytest.mark.parametrize("vec_name", list(VECTORS["final"]))
def test_final_vectors(impl_name, vec_name):
    impl = IMPLS[impl_name]
    entry = VECTORS["final"][vec_name]
    msg = bytes.fromhex(entry["input_hex"])
    assert impl.pde_hash(msg).hex() == entry["digest_hex"]


@pytest.mark.parametrize("impl_name", list(IMPLS))
def test_intermediate_snapshots(impl_name):
    impl = IMPLS[impl_name]
    snaps = impl.trace(b"abc")
    for key, expected_hex in VECTORS["intermediate_abc"].items():
        assert snaps[key].hex() == expected_hex, f"{impl_name}: {key} drifted"
