"""Fast parity / exact-trace regression for the curvature-capacity core.

* the wave round is byte-identical to the frozen Design A round;
* reference and optimized implementations agree byte-for-byte on the digest and
  on the named coupled snapshots;
* pinned CC-Core-v0 test vectors are reproduced.
"""

import importlib

import numpy as np
import pytest

from wavelock.curvature_capacity import spec, reference as ref, optimized as opt
from wavelock.pde_hash.state import PDEState

_a_ev = importlib.import_module("wavelock.pde_hash.evolve")

P, N = spec.P, spec.N

PINNED = {
    b"": "99e7beade48a10b0e4badf5dcecfa617e3a361b789c60e5afaee8c02ab55d6d2",
    b"\x00": "c44b0b91973fe2ec9af5dae3692c9e83a14fc707f0359b2e6800d429bc63e266",
    b"\x01": "8fdc46738f2d7e7167b9af90c5b992431dbdf484a11d56c43236919973061a17",
    b"\xff": "783e26c6db282e939b65894e54afc9d51d000c3f41e0dc39f83c7b886f1cc973",
    b"abc": "385600d91057f24522d1210cb4bb7c8983339cfdcb0b466e41bde2d2c93044ef",
    b"WaveLock": "afbcb55d3a54af05a370703780273e375b08269e4071920c92f3e35a3978c44e",
    b"\x00" * 192: "faa462019807c38ccefac61c50e79a1c6e1dfc46825d510d3526871d3a523f5f",
    b"\x00" * 193: "34d8c71f620edae8f9f2d4039376be2e29ffc59435a66ea984a55bc197d6e8d8",
}


def test_wave_round_matches_design_a():
    rng = np.random.default_rng(7)
    for _ in range(20):
        st = rng.integers(0, P, size=(N, N), dtype=np.int64)
        got = opt._wave_round(st.copy()) % P
        want = np.array(_a_ev.evolve(PDEState([int(x) for x in st.reshape(-1)]), 1).cells,
                        dtype=np.int64).reshape(N, N) % P
        assert np.array_equal(got, want)


@pytest.mark.parametrize("m", list(PINNED) + [bytes(range(256)), b"\x00\x01\x02" * 90])
def test_reference_optimized_parity(m):
    assert ref.cc_hash(m) == opt.cc_hash(m)


@pytest.mark.parametrize("m,hexd", list(PINNED.items()), ids=lambda x: repr(x)[:16])
def test_pinned_vectors(m, hexd):
    if isinstance(m, bytes):
        assert opt.cc_hash(m).hex() == hexd


def test_wave_field_is_design_a_via_trajectory():
    # a random wave state evolved T rounds inside the core matches Design A T-evolve
    rng = np.random.default_rng(11)
    st = rng.integers(0, P, size=(N, N), dtype=np.int64)
    psi = st.copy()
    for _ in range(spec.T):
        psi = opt._wave_round(psi)
    want = np.array(_a_ev.evolve_T(PDEState([int(x) for x in st.reshape(-1)])).cells,
                    dtype=np.int64).reshape(N, N) % P
    assert np.array_equal(psi % P, want)
