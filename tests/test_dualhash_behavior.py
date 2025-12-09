import pytest
from wavelock.chain.WaveLock import CurvatureKeyPair

def test_dualhash_all_modes():
    kp = CurvatureKeyPair(n=8)
    raw = kp._curvature_hash_raw()
    dh = kp.dual_hash

    # Case A: both correct
    p_ok, s_ok = dh.verify(raw)
    assert p_ok and s_ok

    # Case B: primary wrong
    corrupted = raw + b"x"
    p2, s2 = dh.verify(corrupted)
    assert p2 == False

    # Case C: secondary wrong
    truncated = raw[:-1]
    p3, s3 = dh.verify(truncated)
    assert s3 == False

    # Case D: both wrong
    reversed_raw = raw[::-1]
    assert dh.verify(reversed_raw) == (False, False)
