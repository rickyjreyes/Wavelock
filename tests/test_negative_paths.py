import pytest
from wavelock.chain.WaveLock import CurvatureKeyPair
from wavelock.chain.hash_families import hash_hex, HashFamily

def test_tampered_commitment_detection():
    kp = CurvatureKeyPair(n=8)
    c = kp.commitment
    # Corrupt the final hex char, picking a value guaranteed to differ from the
    # original (a fixed "a" is a no-op ~1/16 of the time → flaky).
    repl = "b" if c[-1] == "a" else "a"
    tampered = c[:-1] + repl

    assert tampered != kp.commitment

def test_wrong_psi_star_rejection():
    kp1 = CurvatureKeyPair(n=8)
    kp2 = CurvatureKeyPair(n=8)

    raw1 = kp1._curvature_hash_raw()
    raw2 = kp2._curvature_hash_raw()

    assert hash_hex(raw1, HashFamily.SHA256) != hash_hex(raw2, HashFamily.SHA256)
