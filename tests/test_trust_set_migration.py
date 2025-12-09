import pytest
from wavelock.chain.WaveLock import CurvatureKeyPair
from wavelock.chain.migrate import upgrade_commitment_with_psi

def test_trust_set_migration():
    # simulate a network trust set
    trust = {}

    for i in range(4):
        kp = CurvatureKeyPair(n=8)
        trust[kp.commitment] = kp

    # now upgrade all commitments
    migrated = {}
    for old_commit, kp in trust.items():
        raw = kp._curvature_hash_raw()
        new_commit = upgrade_commitment_with_psi(old_commit, raw)
        migrated[new_commit] = kp

    # ensure no entries lost
    assert len(migrated) == len(trust)

    # ensure all are dual-hash now
    for c in migrated.keys():
        assert c.count(":") == 2  # schema:h1:h2
