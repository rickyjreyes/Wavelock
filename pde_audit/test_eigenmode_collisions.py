"""Regression tests for the structured Laplacian-eigenmode collision family.

These pin EXACT internal-state collisions in the normative WaveLock-PDE-256-v0
one-round map and evolve_T (T=32). Verification is by element-by-element state
equality only -- no floating point, no tolerance, no digest-of-state.

A failure here means the constructive collision claim is wrong and the analysis
must stop.
"""

import importlib

import numpy as np
import pytest

from wavelock.pde_hash import spec, optimized as opt
from wavelock.pde_hash.state import PDEState
from pde_audit._harness import PDEVariant, PDEParams
from pde_audit import eigenmode_collisions as ec

ev = importlib.import_module("wavelock.pde_hash.evolve")
P, N = spec.P, spec.N
V = PDEVariant(PDEParams())
ZERO = np.zeros((N, N), dtype=np.int64)
ii, jj = np.indices((N, N))


# ---------------------------------------------------------------------------
# algebraic derivation
# ---------------------------------------------------------------------------
def test_modular_inverse_of_a():
    assert ec.inv_mod(spec.A) == 1431655765
    assert (spec.A * ec.inv_mod(spec.A)) % P == 1


@pytest.mark.parametrize("r,expected_s2,is_qr,roots", [
    (0, 715827883, False, None),
    (1, 1431655762, True, (1217065103, 930418544)),
    (2, 2147483641, True, (1395627816, 751855831)),
    (3, 715827873, False, None),
    (4, 1431655752, True, (151946369, 1995537278)),
])
def test_amplitude_per_r(r, expected_s2, is_qr, roots):
    amp = ec.amplitude_for_r(r)
    assert amp["s_squared"] == expected_s2
    assert amp["is_quadratic_residue"] == is_qr
    if is_qr:
        assert tuple(amp["roots"]) == roots
        for s in amp["roots"]:
            assert (s * s) % P == expected_s2


# ---------------------------------------------------------------------------
# exact internal-state collisions (both implementations, both amplitudes)
# ---------------------------------------------------------------------------
def _ref_one(arr):
    return np.array(ev.evolve(PDEState([int(x) for x in arr.reshape(-1)]), 1).cells,
                    dtype=np.int64).reshape(N, N) % P


def _ref_T(arr):
    return np.array(ev.evolve_T(PDEState([int(x) for x in arr.reshape(-1)])).cells,
                    dtype=np.int64).reshape(N, N) % P


SIGNS = {
    "checkerboard_r4": ((-1) ** (ii + jj), 4),
    "stripe_rows_r2": ((-1) ** ii, 2),
    "stripe_cols_r2": ((-1) ** jj, 2),
    "period4_cols_r1": (np.array([1, -1, -1, 1])[jj % 4], 1),
}


@pytest.mark.parametrize("name", list(SIGNS))
@pytest.mark.parametrize("amp_idx", [0, 1])      # +s and -s
def test_structured_state_maps_to_zero_exactly(name, amp_idx):
    sig, r = SIGNS[name]
    amp = ec.amplitude_for_r(r)
    assert amp["is_quadratic_residue"]
    s = amp["roots"][amp_idx]
    state = ((s * sig) % P).astype(np.int64)
    assert not np.array_equal(state, ZERO)       # it's a nonzero state

    # one round -> exactly zero in all three implementations
    assert np.array_equal(V.one_round(state), ZERO)
    assert np.array_equal(_ref_one(state), ZERO)
    assert np.array_equal(opt._one_round(state.copy()) % P, ZERO)

    # evolve_T -> exactly zero in reference and harness
    assert np.array_equal(V.evolve_T(state), ZERO)
    assert np.array_equal(_ref_T(state), ZERO)
    assert np.array_equal(opt._evolve_T(state.copy()) % P, ZERO)


def test_sign_fields_are_laplacian_eigenvectors():
    for name, (sig, r) in SIGNS.items():
        ok, lam, r_eig = ec.is_laplacian_eigenvector(sig)
        assert ok, name
        assert lam == (-2 * r) % P, (name, lam)


def test_zero_is_fixed_point_exact():
    assert np.array_equal(V.one_round(ZERO), ZERO)
    assert np.array_equal(V.evolve_T(ZERO), ZERO)
    assert np.array_equal(_ref_one(ZERO), ZERO)
    assert np.array_equal(_ref_T(ZERO), ZERO)


def test_distinct_states_collide():
    # checkerboard(+s), checkerboard(-s), and zero are three DISTINCT states
    # all mapping to zero in one round.
    sig = (-1) ** (ii + jj)
    amp = ec.amplitude_for_r(4)
    sp = ((amp["roots"][0] * sig) % P).astype(np.int64)
    sm = ((amp["roots"][1] * sig) % P).astype(np.int64)
    assert not np.array_equal(sp, sm)
    assert not np.array_equal(sp, ZERO)
    for st in (sp, sm, ZERO):
        assert np.array_equal(V.one_round(st), ZERO)


def test_zero_state_digest_is_all_zero_bytes():
    # squeeze of the all-zero state: every comparison is a tie -> all-zero bits
    assert V.squeeze(ZERO) == b"\x00" * 32
