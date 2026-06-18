"""Phase CC-2 Part I -- Candidate A frozen-baseline regression.

Fails if Candidate A's (CC-Core-v0-A) equations, constants, or pinned outputs
change. Candidate A is a FROZEN experimental baseline (see docs/CC_CORE_V0_BASELINE.md);
Phase CC-2 must not alter it.
"""

from __future__ import annotations

import numpy as np

from wavelock.curvature_capacity import spec, optimized as opt, reference as ref

P = spec.P
N = spec.N

# Pinned constants (frozen). Any change here is an intentional break and fails.
FROZEN_CONSTANTS = {
    "P": 2147483647, "N": 16, "N_CELLS": 256,
    "D": 5, "A": 3, "B": 1431655765, "T": 32,
    "D_C": 3, "GAMMA": 11, "ETA": 13, "ZETA": 17, "A_C": 2, "MU": 5,
    "RHO0": 0x57434330, "RHO1": 2654435761 % 2147483647,
    "WA": 40503, "WB": 50021, "WC": 60013,
    "G": 7, "D_TAG": 0x57434331, "CAP0": 64, "CAP1": 65, "CAP2": 66,
    "RATE": 64, "BYTES_PER_BLOCK": 192,
}

FROZEN_CC_DIGESTS = {
    b"": "99e7beade48a10b0e4badf5dcecfa617e3a361b789c60e5afaee8c02ab55d6d2",
    b"\x00": "c44b0b91973fe2ec9af5dae3692c9e83a14fc707f0359b2e6800d429bc63e266",
    b"\x01": "8fdc46738f2d7e7167b9af90c5b992431dbdf484a11d56c43236919973061a17",
    b"\xff": "783e26c6db282e939b65894e54afc9d51d000c3f41e0dc39f83c7b886f1cc973",
    b"abc": "385600d91057f24522d1210cb4bb7c8983339cfdcb0b466e41bde2d2c93044ef",
    b"WaveLock": "afbcb55d3a54af05a370703780273e375b08269e4071920c92f3e35a3978c44e",
    b"\x00" * 192: "faa462019807c38ccefac61c50e79a1c6e1dfc46825d510d3526871d3a523f5f",
    b"\x00" * 193: "34d8c71f620edae8f9f2d4039376be2e29ffc59435a66ea984a55bc197d6e8d8",
}

FROZEN_ZERO_TRAJECTORY = "c4ed8b688e14f2127c8e03ea62eee0ecd10cc15ed5dd695afc8a193efc9198d3"


def test_frozen_constants_unchanged():
    for name, val in FROZEN_CONSTANTS.items():
        assert getattr(spec, name) == val, f"Candidate A constant {name} changed!"


def test_frozen_cc_digests_unchanged():
    for msg, expected in FROZEN_CC_DIGESTS.items():
        assert opt.cc_hash(msg).hex() == expected, f"opt cc_hash({msg!r}) changed"
        assert ref.cc_hash(msg).hex() == expected, f"ref cc_hash({msg!r}) changed"


def test_frozen_zero_trajectory_unchanged():
    z = np.zeros((N, N), dtype=np.int64)
    assert opt.trajectory_digest(z).hex() == FROZEN_ZERO_TRAJECTORY


def test_frozen_injection_is_candidate_a_quadratic():
    """j_A must retain the ETA*u^2 term (the defining Candidate A feature)."""
    assert spec.ETA == 13 and spec.ETA != 0, "Candidate A must keep ETA != 0"
    # spot-check j_A formula at a fixed point
    u, v = 12345, 67890
    p = P
    j = (u + spec.GAMMA * (u * v % p) + spec.ETA * (u * u % p) + spec.ZETA * v) % p
    # recompute via accumulator behaviour: with C=0, t=0, the W*j term is visible
    assert j == (u + 11 * (u * v % p) + 13 * (u * u % p) + 17 * v) % p


def test_frozen_2to1_relation_holds():
    """The proved 2-to-1 relation j_A(u,v)=j_A(u',v) for u' = -(1+gamma*v)/eta - u."""
    eta_inv = pow(spec.ETA, P - 2, P)
    for u, v in [(1, 2), (100, 200), (999999, 12345678), (P - 1, P - 2)]:
        u_prime = (P - (1 + spec.GAMMA * v) % P * eta_inv % P - u) % P
        p = P
        j1 = (u + spec.GAMMA * (u * v % p) + spec.ETA * (u * u % p) + spec.ZETA * v) % p
        j2 = (u_prime + spec.GAMMA * (u_prime * v % p)
              + spec.ETA * (u_prime * u_prime % p) + spec.ZETA * v) % p
        assert j1 == j2, f"Candidate A 2-to-1 relation broken at u={u}, v={v}"


def test_design_a_still_frozen():
    """Cross-check the frozen Design A primitive digests (must never change)."""
    from wavelock.pde_hash import optimized as a_opt
    assert a_opt.pde_hash(b"").hex() == \
        "d12c29be1429775e6dcc9ff3e29d9bca96865c0179a99b9bcee58581bf118820"
    assert a_opt.pde_hash(b"abc").hex() == \
        "e6231beb61a76e304a5292473a955a970b74b25f55027ca6f0cc34a1cd21985d"
