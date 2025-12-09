import pytest
from wavelock.chain.WaveLock import CurvatureKeyPair

def test_dual_signature_strict_and_survivability():
    kp = CurvatureKeyPair(n=8)
    msg = b"hello world"

    sig_p, sig_s = kp.sign_dual(msg)

    # Survivability checks
    assert kp.verify(msg, sig_p)
    assert kp.verify(msg, sig_s)

    # Strict mode = both required
    assert kp.verify_strict(msg, sig_p, sig_s)

    # Wrong order should fail strict check
    assert not kp.verify_strict(msg, sig_s, sig_p)

def test_dual_signature_tamper():
    kp = CurvatureKeyPair(n=8)
    msg = b"test123"
    sig_p, sig_s = kp.sign_dual(msg)

    bad = sig_p[:-2] + "aa"  # tamper primary
    assert not kp.verify(msg, bad)
