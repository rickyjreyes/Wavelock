import pytest
from wavelock.chain.WaveLock import CurvatureKeyPair, parse_commitment
from wavelock.chain.migrate import upgrade_commitment_with_psi
from wavelock.chain.hash_families import HashFamily, hash_hex

def test_commitment_upgrade_v2_to_v3():
    kp = CurvatureKeyPair(n=8)

    raw = kp._curvature_hash_raw()
    old_primary = hash_hex(raw, HashFamily.SHA256)

    # Construct old-style commitment
    old_commit = f"{kp.schema}:{old_primary}"

    upgraded = upgrade_commitment_with_psi(old_commit, raw)
    schema, h1, h2 = parse_commitment(upgraded)

    assert schema == kp.schema
    assert h1 == old_primary
    assert len(h2) == 64
    assert h2 == hash_hex(raw, HashFamily.SHA3_256)

def test_upgrade_rejects_wrong_data():
    kp = CurvatureKeyPair(n=8)

    raw = kp._curvature_hash_raw()
    wrong_raw = raw + b"x"

    old_primary = hash_hex(raw, HashFamily.SHA256)
    old_commit = f"{kp.schema}:{old_primary}"

    with pytest.raises(ValueError):
        upgrade_commitment_with_psi(old_commit, wrong_raw)
