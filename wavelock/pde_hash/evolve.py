"""Phi: the finite-field Allen-Cahn reaction-diffusion PDE round (reference).

Pure-Python, exact modular integer arithmetic over F_p (p = 2**31 - 1). This is
the readable reference; ``optimized.py`` reimplements the same map in NumPy
without sharing this code, and Phase-7 parity tests require byte-identical
agreement including intermediate-round snapshots.

One round (spec §1.1, §5), Jacobi update from pre-update psi only:

    t       = psi^2 mod p
    react   = a * (psi * ((b + (p - t)) mod p) mod p) mod p
    lap     = (psi[i+1,j] + psi[i-1,j] + psi[i,j+1] + psi[i,j-1]
               + (p-4)*psi[i,j]) mod p
    psi'    = (psi + (D*lap mod p) + react) mod p
"""

from __future__ import annotations

from . import spec
from .state import PDEState

_P = spec.P
_N = spec.N
_D = spec.D
_A = spec.A
_B = spec.B
_PM4 = (_P - 4) % _P     # (p - 4), the Laplacian centre coefficient


def _one_round(cells):
    """Apply one PDE round to a flat list of residues; return a new list."""
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

            # reaction: a * psi * (b - psi^2)
            t = (psi * psi) % p
            bm = (_B + (p - t)) % p          # b - psi^2  (non-negative)
            react = (_A * ((psi * bm) % p)) % p

            # diffusion: 5-point toroidal Laplacian
            jl = (j - 1) % n
            jr = (j + 1) % n
            lap = (
                cells[down + j]
                + cells[up + j]
                + cells[row + jr]
                + cells[row + jl]
                + (_PM4 * psi) % p
            ) % p

            new[idx] = (psi + (_D * lap) % p + react) % p
    return new


def evolve(state: PDEState, rounds: int) -> PDEState:
    """Apply ``rounds`` PDE rounds and return a new PDEState."""
    if rounds < 0:
        raise ValueError("rounds must be non-negative")
    cells = list(state.cells)
    for _ in range(rounds):
        cells = _one_round(cells)
    return PDEState(cells)


def permute(state: PDEState) -> PDEState:
    """Apply exactly Phi_P^T (T rounds) — one permutation call."""
    return evolve(state, spec.T)
