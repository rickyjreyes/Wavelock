"""Regression: the Design A eigenmode collisions and their non-collision under
the path commitment.

These pin the central claims of the curvature-capacity work:
  * Design A property PRESERVED: every eigenmode state F-collapses to terminal 0;
  * REPAIR: those terminal-state collisions are NOT digest collisions under the
    trajectory commitment (all distinct, large pairwise Hamming distance);
  * the accumulator has no trivial zero-preimage in a small budget;
  * no toroidal symmetry preserves the trajectory digest;
  * the zero-wave digest is NOT all-zero bytes (non-degeneracy).
"""

import numpy as np

from wavelock.curvature_capacity import spec, optimized as opt
from . import _common as C

P, N = spec.P, spec.N
_ZERO = np.zeros((N, N), dtype=np.int64)


def test_design_a_eigenmodes_still_collapse_to_zero():
    for name, st in C.eigenmode_states().items():
        psi = st.copy()
        for _ in range(spec.T):
            psi = opt._wave_round(psi)
        assert np.array_equal(psi % P, _ZERO), name


def test_trajectory_commitment_separates_eigenmodes():
    eig = C.eigenmode_states()
    digs = {k: opt.trajectory_digest(v) for k, v in eig.items()}
    hexes = [d.hex() for d in digs.values()]
    assert len(set(hexes)) == len(hexes)  # all distinct
    items = list(digs.values())
    min_hd = min(C.hamming_bytes(items[i], items[j])
                 for i in range(len(items)) for j in range(i + 1, len(items)))
    assert min_hd > 64  # well separated (ideal ~128)


def test_accumulator_has_no_trivial_zero_preimage():
    rng = np.random.default_rng(123)
    for _ in range(500):
        Cf = rng.integers(0, P, size=(N, N), dtype=np.int64)
        nxt = opt._accumulator_step(Cf, _ZERO, _ZERO, 0)
        assert not np.array_equal(nxt % P, _ZERO)


def test_symmetry_does_not_preserve_digest():
    rng = np.random.default_rng(321)
    base = rng.integers(0, P, size=(N, N), dtype=np.int64)
    d0 = opt.trajectory_digest(base)
    for v in (np.roll(base, 1, 0), base[::-1, :].copy(), base.T.copy(),
              (P - base) % P):
        assert opt.trajectory_digest(v) != d0


def test_zero_wave_digest_is_nondegenerate():
    d = opt.trajectory_digest(_ZERO)
    assert d != b"\x00" * 32
