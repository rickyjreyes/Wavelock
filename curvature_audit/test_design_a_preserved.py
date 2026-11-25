"""Guard: the frozen Design A primitive is untouched by this branch.

Pins the Design A digests and the eigenmode collision theorem so a regression in
wavelock/pde_hash would fail here too.
"""

import numpy as np

from wavelock.pde_hash import optimized as a_opt, reference as a_ref
from wavelock.curvature_capacity import spec
from . import _common as C

P, N = spec.P, spec.N

DESIGN_A_PINNED = {
    b"": "d12c29be1429775e6dcc9ff3e29d9bca96865c0179a99b9bcee58581bf118820",
    b"abc": "e6231beb61a76e304a5292473a955a970b74b25f55027ca6f0cc34a1cd21985d",
    b"WaveLock": "5109e4c0d3effe338c4b1b35555aac8db35f2754753afea961cd768a04937cb2",
}


def test_design_a_digests_unchanged():
    for m, h in DESIGN_A_PINNED.items():
        assert a_opt.pde_hash(m).hex() == h
        assert a_ref.pde_hash(m).hex() == h


def test_design_a_eigenmode_collision_theorem_holds():
    # F(s*sigma) == 0 for the enumerated family (one Design A round).
    import importlib
    ev = importlib.import_module("wavelock.pde_hash.evolve")
    from wavelock.pde_hash.state import PDEState
    zero = np.zeros((N, N), dtype=np.int64)
    for name, st in C.eigenmode_states().items():
        if name == "zero":
            continue
        out = np.array(ev.evolve(PDEState([int(x) for x in st.reshape(-1)]), 1).cells,
                       dtype=np.int64).reshape(N, N) % P
        assert np.array_equal(out, zero), name
