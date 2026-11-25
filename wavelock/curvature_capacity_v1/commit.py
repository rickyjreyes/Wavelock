"""The path-binding accumulator update Phi_t^(B) (Candidate B, reference).

Candidate B uses a LINEAR trajectory injection:

    j_B(u, v) = u + GAMMA*u*v  =  u*(1 + GAMMA*v)        (mod p)

(compared to Candidate A's  j_A = u + GAMMA*u*v + ETA*u^2 + ZETA*v).

Everything else is identical to Candidate A:

    Cd[x]      = ( C + D_C * Laplacian(C) )[x]                   mod p
    W_t(x)     = ( 1 + (t+1)*WA + (x+1)*WB + (t+1)*(x+1)*WC )    mod p
    rho_t      = ( RHO0 + RHO1*t )                              mod p
    C'[x]      = ( MU*Cd[x] + A_C*Cd[x]^2 + W_t(x)*j_B(u,v) + rho_t )  mod p

Structural consequence (analyzed in docs/CC_CORE_V1_ALGEBRA.md):
  * For fixed v, j_B is LINEAR in u with slope (1 + GAMMA*v). It is injective in
    u whenever (1 + GAMMA*v) != 0 mod p, removing Candidate A's generic 2-to-1.
  * BUT when v == V_STAR = -GAMMA^{-1} mod p, the slope is 0 and j_B(u, V_STAR)=0
    for every u: the injection erases all u-information at that coordinate. This
    singular hyperplane is the new risk Phase CC-2 must audit (Part IV).

No conventional cryptographic primitive is used.
"""

from __future__ import annotations

from typing import List

from . import spec

_P = spec.P
_N = spec.N
_PM4 = (_P - 4) % _P
_DC = spec.D_C
_GAMMA = spec.GAMMA
_AC = spec.A_C
_MU = spec.MU
_WA = spec.WA
_WB = spec.WB
_WC = spec.WC


def _laplacian(cells: List[int]) -> List[int]:
    p = _P
    n = _N
    out = [0] * spec.N_CELLS
    for i in range(n):
        row = i * n
        up = ((i - 1) % n) * n
        down = ((i + 1) % n) * n
        for j in range(n):
            idx = row + j
            jl = (j - 1) % n
            jr = (j + 1) % n
            out[idx] = (
                cells[down + j] + cells[up + j]
                + cells[row + jr] + cells[row + jl]
                + (_PM4 * cells[idx]) % p
            ) % p
    return out


def weight(t: int, x: int) -> int:
    return (1 + (t + 1) * _WA + (x + 1) * _WB + (t + 1) * (x + 1) * _WC) % _P


def injection(u: int, v: int) -> int:
    """j_B(u, v) = u + GAMMA*u*v = u*(1 + GAMMA*v) mod p."""
    p = _P
    return (u + _GAMMA * ((u * v) % p)) % p


def accumulator_step(C: List[int], psi_t: List[int], psi_next: List[int],
                     t: int) -> List[int]:
    """Apply one Candidate B accumulator round Phi_t^(B); return a new flat list."""
    p = _P
    rho = spec.round_constant(t)
    Cd = _laplacian(C)
    out = [0] * spec.N_CELLS
    t1 = t + 1
    for x in range(spec.N_CELLS):
        u = psi_t[x]
        v = psi_next[x]
        j = (u + _GAMMA * ((u * v) % p)) % p          # j_B (linear injection)
        cd = (C[x] + (_DC * Cd[x]) % p) % p
        w = (1 + t1 * _WA + (x + 1) * _WB + t1 * (x + 1) * _WC) % p
        out[x] = (_MU * cd + (_AC * ((cd * cd) % p)) % p + (w * j) % p + rho) % p
    return out
