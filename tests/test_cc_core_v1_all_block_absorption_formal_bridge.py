"""All-block CC-Core-v1-B absorption parity against the normative reference.

This independently reconstructs the Fin-256 finite-field branch cases in
WaveLockProofs.BlockAbsorption.lean and compares the Python implementation
for rate cells, counter capacities, finalization and untouched sites.
It is a deterministic implementation *test*, not a formal cross-language
semantics proof, a coupled-round parity theorem, or a hardness claim.
"""

from __future__ import annotations

import pytest

from wavelock.curvature_capacity_v1 import reference, spec
from wavelock.curvature_capacity_v1.state import initial_state


def scalar_absorb(before: list[int], block: list[int], k: int,
                  is_last: bool, rate_gain: int) -> list[int]:
    p = spec.P
    q = k + 1
    lo = q % p
    hi = q // p
    result = list(before)
    for i in range(spec.N_CELLS):
        if i < spec.RATE:
            result[i] = (before[i] + rate_gain * block[i]) % p
        elif i == spec.CAP0:
            result[i] = (before[i] + spec.G * lo) % p
        elif i == spec.CAP2:
            result[i] = (before[i] + spec.G * hi) % p
        elif i == spec.CAP1 and is_last:
            result[i] = (before[i] + spec.D_TAG) % p
    return result


@pytest.mark.parametrize("k", [0, 1, 33, spec.P - 2, spec.P - 1, spec.P, spec.P + 1])
@pytest.mark.parametrize("is_last", [False, True])
def test_all_block_absorption_parity(k: int, is_last: bool) -> None:
    state = initial_state()
    # Exercise wraparound at wave and accumulator rate cells without
    # rewriting or freezing the initial-state vectors.
    state.psi[0] = spec.P - 3
    state.C[0] = spec.P - 2
    state.psi[spec.CAP0] = spec.P - 4
    state.C[spec.CAP2] = spec.P - 5

    block = [((i * 123457 + 991) ^ (i << 12)) % (1 << 24)
             for i in range(spec.RATE)]
    block[0] = (1 << 24) - 1
    block[-1] = 0

    psi_before = state.psi.copy()
    c_before = state.C.copy()
    expected_psi = scalar_absorb(psi_before, block, k, is_last, 1)
    expected_c = scalar_absorb(c_before, block, k, is_last, spec.G)

    reference._inject_block(state, block, k, is_last)

    assert state.psi == expected_psi
    assert state.C == expected_c
    assert state.psi[67:] == psi_before[67:]
    assert state.C[67:] == c_before[67:]
    if not is_last:
        assert state.psi[spec.CAP1] == psi_before[spec.CAP1]
        assert state.C[spec.CAP1] == c_before[spec.CAP1]
