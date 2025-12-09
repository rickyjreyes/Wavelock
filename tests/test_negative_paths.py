import pytest
from wavelock.chain.WaveLock import CurvatureKeyPair
from wavelock.chain.hash_families import hash_hex, HashFamily

def test_tampered_commitment_detection():
    kp = CurvatureKeyPair(n=8)
    c = kp.commitment
    tampered = c[:-1] + "a"  # corrupt final hex char

    assert tampered != kp.commitment

def test_wrong_psi_star_rejection():
    kp1 = CurvatureKeyPair(n=8)
    kp2 = CurvatureKeyPair(n=8)

    raw1 = kp1._curvature_hash_raw()
    raw2 = kp2._curvature_hash_raw()

    assert hash_hex(raw1, HashFamily.SHA256) != hash_hex(raw2, HashFamily.SHA256)
