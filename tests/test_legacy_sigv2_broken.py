"""
Regression tests that PROVE the legacy WaveLock SIGv2 design is dead.

The legacy attacks must keep succeeding against legacy SIGv2 (documenting the
break), and the companion test_ots_security.py proves the same attacks fail
against WaveLock-OTS. If anyone "fixes" legacy SIGv2 by merely hiding ψ★ while
still requiring the verifier to recompute the MAC, these tests still describe
the real situation: the scheme is symmetric.
"""

from attacks.forge_from_snapshot import (
    make_legacy_victim,
    forge_legacy_signature,
    legacy_forgery_succeeds,
)
from attacks.seed_bruteforce import (
    make_legacy_commitment,
    bruteforce_legacy_seed,
    ots_seed_search_is_infeasible,
)


def test_legacy_forge_from_snapshot_succeeds():
    """Holding only ψ★ lets an attacker forge any message — by design (broken)."""
    assert legacy_forgery_succeeds(
        attacker_message="TRANSFER 1000000 TO ATTACKER"
    ) is True


def test_legacy_verifier_equals_forger():
    """Whoever can verify (has ψ★) can forge: capability sets are identical."""
    victim, snapshot = make_legacy_victim(seed=7)
    forged = forge_legacy_signature(snapshot, n=4,
                                    attacker_message="arbitrary attacker text")
    assert victim.verify("arbitrary attacker text", forged) is True


def test_legacy_small_seed_bruteforceable():
    commitment = make_legacy_commitment(seed=123)
    recovered = bruteforce_legacy_seed(commitment, max_seed=500)
    assert recovered == 123


def test_ots_seed_space_is_not_bruteforceable():
    info = ots_seed_search_is_infeasible(entropy_bits=256)
    assert info["feasible"] is False
    assert info["search_space"] == 2 ** 256
