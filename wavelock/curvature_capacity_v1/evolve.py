"""Wave round F (reference) and the coupled (psi, C) round for Candidate B.

The wave round F is the *exact, unmodified* Design A round (verbatim, asserted
byte-for-byte against wavelock.pde_hash by tests). Only the accumulator step is
Candidate B's (commit.py).
"""

from __future__ import annotations

from . import spec
from .state import CCStateV1
from .commit import accumulator_step

_P = spec.P
_N = spec.N
_D = spec.D
_A = spec.A
_B = spec.B
_PM4 = (_P - 4) % _P


def wave_one_round(cells):
    """One Design A wave round on a flat list of residues -> new flat list."""
    p = _P
    n = _N
    new = [0] * spec.N_CELLS
    for i in range(n):
        row = i * n
        up = ((i - 1) % n) * n
        down = ((i + 1) % n) * n
        for j in range(n):
            idx = row + j
            psi = cells[idx]
            t = (psi * psi) % p
            bm = (_B + (p - t)) % p
            react = (_A * ((psi * bm) % p)) % p
            jl = (j - 1) % n
            jr = (j + 1) % n
            lap = (
                cells[down + j] + cells[up + j]
                + cells[row + jr] + cells[row + jl]
                + (_PM4 * psi) % p
            ) % p
            new[idx] = (psi + (_D * lap) % p + react) % p
    return new


def coupled_one_round(state: CCStateV1, round_index: int) -> CCStateV1:
    """One coupled round at global round index ``round_index`` (Candidate B)."""
    psi_t = state.psi
    psi_next = wave_one_round(psi_t)
    C_next = accumulator_step(state.C, psi_t, psi_next, round_index)
    return CCStateV1(psi_next, C_next)


def coupled_evolve_T(state: CCStateV1, start_round: int, rounds: int | None = None):
    """Apply ``rounds`` (default T) coupled rounds; return (new_state, next_round_index)."""
    r = spec.T if rounds is None else rounds
    s = state
    ri = start_round
    for _ in range(r):
        s = coupled_one_round(s, ri)
        ri += 1
    return s, ri
