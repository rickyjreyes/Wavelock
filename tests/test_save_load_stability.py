import pytest
import os
from wavelock.chain.WaveLock import CurvatureKeyPair
from wavelock.chain.WaveLock import save_quantum_keys, load_quantum_keys

def test_save_load_roundtrip(tmp_path):
    kp = CurvatureKeyPair(n=8)
    file = tmp_path / "keys.json"

    save_quantum_keys(str(file), { "kp": kp })
    data = load_quantum_keys(str(file))

    kp2 = data["kp"]

    assert kp2.commitment == kp.commitment
    assert kp2.schema == kp.schema
    assert kp2.primary_family == kp.primary_family
    assert kp2.secondary_family == kp.secondary_family
