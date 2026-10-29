"""Protect the frozen executable CC-Core A/B constants against decimal drift.

Earlier documentation and the first Lean round model used incorrect
decimal expansions of correct hexadecimal/modulus expressions. These
tests bind the Python implementation to its actual literal values.
The corresponding *Lean* equalities live in NormativeConstants.lean.
"""

from __future__ import annotations

from math import gcd

from wavelock.curvature_capacity import spec as candidate_a
from wavelock.curvature_capacity_v1 import spec as candidate_b


def test_candidate_b_literal_values() -> None:
    assert candidate_b.P == 2**31 - 1
    assert candidate_b.RHO0 == 0x57434330 == 1_464_025_904
    assert candidate_b.RHO1 == 2_654_435_761 % candidate_b.P == 506_952_114
    assert candidate_b.D_TAG == 0x57434332 == 1_464_025_906
    assert candidate_b.G == 7
    assert candidate_b.RATE == 64
    assert candidate_b.BYTES_PER_BLOCK == 192


def test_candidate_a_shares_round_constants_but_not_final_tag() -> None:
    assert candidate_a.P == candidate_b.P
    assert candidate_a.RHO0 == candidate_b.RHO0
    assert candidate_a.RHO1 == candidate_b.RHO1
    assert candidate_a.D_TAG == 0x57434331 == 1_464_025_905
    assert candidate_a.D_TAG != candidate_b.D_TAG


def test_affine_round_schedule_is_not_multiplicative_group_period() -> None:
    p = candidate_b.P
    slope = candidate_b.RHO1
    assert gcd(slope, p - 1) == 186
    assert gcd(slope, p) == 1
    for t in (0, 1, 31, 32, 63, 127, 4096):
        expected = (1_464_025_904 + 506_952_114 * t) % p
        assert candidate_b.round_constant(t) == expected
        assert candidate_b.round_constant(t + p) == expected
        assert candidate_b.round_constant(t + (p - 1)) != expected
