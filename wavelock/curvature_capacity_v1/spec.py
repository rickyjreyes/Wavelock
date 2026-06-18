"""WaveLock Curvature-Capacity Core CC-Core-v1 / Candidate B -- spec constants.

Candidate B is the *linear-injection* revision of Candidate A (CC-Core-v0,
``wavelock.curvature_capacity``). The ONLY algebraic change is the trajectory
injection:

    Candidate A:  j_A(u, v) = u + GAMMA*u*v + ETA*u^2 + ZETA*v
    Candidate B:  j_B(u, v) = u + GAMMA*u*v                       =  u*(1 + GAMMA*v)

i.e. ETA = 0 AND ZETA = 0, leaving the single multiplicative factor (1 + GAMMA*v).
This removes Candidate A's *proved generic 2-to-1* relation (the symmetric ETA*u^2
term) but introduces a *singular multiplicative hyperplane*: when 1 + GAMMA*v == 0
mod p the injection collapses to 0 for every u. That singular value is

    v_star = -GAMMA^{-1} mod p = 195225786

and is analyzed adversarially in docs/CC_CORE_V1_ALGEBRA.md before any claim is made.

Status: EXPERIMENTAL research candidate. **No** security claim. This package never
overwrites Candidate A; the two are domain-separated by VERSION and D_TAG. The
wave field psi still evolves by the *unmodified Design A round* (frozen primitive).

Design choice (controlled comparison): the lattice, field, wave coefficients,
IVs, weights, round constants, rate/capacity layout, and squeeze are IDENTICAL
to Candidate A so that an A/B comparison from the same initial wave state isolates
the single change (the injection). Message-level domain separation is provided by
a distinct finalization tag D_TAG and a distinct VERSION string.
"""

from __future__ import annotations

# --- shared field / lattice (identical to Design A and Candidate A) -----
P = (1 << 31) - 1            # 2**31 - 1, Mersenne prime
N = 16                       # lattice side; N*N = 256 cells
N_CELLS = N * N

# --- wave-field (psi) PDE coefficients: EXACTLY Design A ----------------
D = 5
A = 3
B = 1431655765
T = 32

# --- accumulator-field (C) coefficients ---------------------------------
# Candidate B differs from Candidate A ONLY in the injection: ETA = 0, ZETA = 0.
D_C = 3                      # accumulator self-diffusion (same as A)
GAMMA = 11                   # cross term psi_t * psi_{t+1}  (same as A)
ETA = 0                      # *** Candidate B: removed quadratic u^2 term ***
ZETA = 0                     # *** Candidate B: removed linear-in-v term  ***
A_C = 2                      # accumulator self-square (same as A)
MU = 5                       # accumulator carry multiplier (same as A)
RHO0 = 0x57434330            # round-constant base (same as A)
RHO1 = 2654435761 % P        # round-constant slope (same as A)
WA = 40503
WB = 50021
WC = 60013

# --- sponge-like / injection constants ----------------------------------
G = 7
D_TAG = 0x57434332           # ASCII "WCC2" = 1463898162 ; DISTINCT from A's WCC1
CAP0 = 64
CAP1 = 65
CAP2 = 66

RATE = 64
BYTES_PER_ELEM = 3
BYTES_PER_BLOCK = RATE * BYTES_PER_ELEM   # 192

MAX_INPUT_BITS = (1 << 64) - 1

# --- domain tags --------------------------------------------------------
# Shared with Candidate A *deliberately* so trajectory_digest from the same psi0
# isolates the injection change. cc_hash() domain separation comes from D_TAG.
IV_TAG_PSI = b"WaveLock-CC-Core-v0:psi"
IV_TAG_C = b"WaveLock-CC-Core-v0:acc"

# --- output -------------------------------------------------------------
OUTPUT_BITS = 256
SQUEEZE_BITS_PER_ROUND = 64

VERSION = "WaveLock-CC-Core-v1"     # DISTINCT from Candidate A's "WaveLock-CC-Core-v0"

# --- the singular hyperplane (Candidate B's new structural risk) --------
# 1 + GAMMA*v == 0 mod p  <=>  v == V_STAR.  j_B(u, V_STAR) == 0 for ALL u.
GAMMA_INV = pow(GAMMA, P - 2, P)
V_STAR = (-GAMMA_INV) % P           # == 195225786

# internal consistency checks (run at import; cheap)
assert P == 2147483647
assert ETA == 0 and ZETA == 0, "Candidate B uses a linear injection (ETA=ZETA=0)"
assert (1 + GAMMA * V_STAR) % P == 0, "V_STAR must satisfy 1 + GAMMA*v == 0 mod p"
assert V_STAR == 195225786
assert N_CELLS == 256 and BYTES_PER_BLOCK == 192
assert len({CAP0, CAP1, CAP2}) == 3 and min(CAP0, CAP1, CAP2) >= RATE
assert D_TAG != 0x57434331, "D_TAG must differ from Candidate A's finalization tag"
assert VERSION != "WaveLock-CC-Core-v0"


def encode_block_counter(k: int):
    """Injective two-digit base-P encoding of 0-based block index k (as Design A)."""
    if k < 0:
        raise ValueError("block index must be non-negative")
    q = k + 1
    return q % P, q // P


def squeeze_pairs():
    """Disjoint comparison pairs (t, t+128) for t in 0..63 (read from C)."""
    return [(t, t + 128) for t in range(SQUEEZE_BITS_PER_ROUND)]


def position_weights(t: int):
    """W_t(x) = (1 + (t+1)*WA + (x+1)*WB + (t+1)*(x+1)*WC) mod P (same as Candidate A)."""
    import numpy as _np
    x = _np.arange(N_CELLS, dtype=_np.int64)
    return (1 + (t + 1) * WA + (x + 1) * WB + (t + 1) * (x + 1) * WC) % P


def round_constant(t: int) -> int:
    """rho_t = (RHO0 + RHO1 * t) mod P (same as Candidate A)."""
    return (RHO0 + RHO1 * t) % P
