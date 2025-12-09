import pytest
from wavelock.chain.WaveLock import CurvatureKeyPair
from wavelock.chain.hash_families import HashFamily, hash_hex
from wavelock.chain.WaveLock import parse_commitment

def test_commitment_format_v3():
    kp = CurvatureKeyPair(n=8)
    schema, h1, h2 = parse_commitment(kp.commitment)

    assert schema == kp.schema
    assert len(h1) == 64
    assert len(h2) == 64

    raw = kp._curvature_hash_raw()
    assert h1 == hash_hex(raw, HashFamily.SHA256)
    assert h2 == hash_hex(raw, HashFamily.SHA3_256)
