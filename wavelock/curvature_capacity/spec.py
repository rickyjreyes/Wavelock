"""WaveLock Curvature-Capacity Core (CC-Core-v0) -- frozen specification constants.

This module is the single source of truth for every constant of the *new*
candidate. It contains NO conventional cryptographic primitive (no SHA / SHAKE /
BLAKE / MD5 / AES / ChaCha / SipHash / HMAC / HKDF / Argon2). Every constant is a
small fixed integer or a deterministic finite-field recurrence; none is derived
from a library digest or XOF.

Status: EXPERIMENTAL research candidate. It makes **no** security claim. See
``docs/WAVELOCK_CURVATURE_CAPACITY_SPEC.md`` for the full specification and
``docs/WAVELOCK_CURVATURE_CAPACITY_RESULTS.md`` for the adversarial audit.

Relationship to Design A (WaveLock-PDE-256-v0)
----------------------------------------------
The *wave field* psi evolves under the **exact, unmodified Design A round** F
(the finite-field Allen-Cahn discretization). Design A is frozen; this package
imports/reproduces its round byte-for-byte and never edits it. The new content
is a **path-binding accumulator field** C that is co-evolved with psi so the
digest commits to the ordered *trajectory* (the wake), not only the terminal
state. The known Design-A eigenmode collisions (psi = s*sigma -> 0) collapse the
terminal *state*; the explicit purpose of C is to test whether the *trajectory*
still separates them.
"""

from __future__ import annotations

# --- shared field / lattice (identical to Design A) ---------------------
P = (1 << 31) - 1            # 2**31 - 1, Mersenne prime
N = 16                       # lattice side; N*N = 256 cells
N_CELLS = N * N

# --- wave-field (psi) PDE coefficients: EXACTLY Design A ----------------
# These mirror wavelock.pde_hash.spec; duplicated here so the package is
# self-contained, and asserted equal to Design A by the parity test.
D = 5                        # diffusion coefficient
A = 3                        # reaction gain
B = 1431655765               # bistable offset in [0, P)
T = 32                       # rounds per T-round transform (provisional)

# --- accumulator-field (C) coefficients: NEW, domain-separated ----------
# All chosen as small fixed integers; none derived from a hash. The design
# intent of each is documented in docs/WAVELOCK_CURVATURE_CAPACITY_SPEC.md.
D_C = 3                      # accumulator self-diffusion coefficient
GAMMA = 11                   # cross term  psi_t * psi_{t+1}   (couples both states)
ETA = 13                     # psi_t^2 term (depends on the *earlier* state only)
ZETA = 17                    # psi_{t+1} linear term (depends on *later* state only)
A_C = 2                      # accumulator self-square (nonlinearity in C)
MU = 5                       # accumulator carry multiplier
# round-constant schedule rho_t = (RHO0 + RHO1 * t) mod P
RHO0 = 0x57434330            # ASCII "WCC0" = 1463898160 ; < P
RHO1 = 2654435761 % P        # an odd multiplier, reduced into [0, P)
# position-weight schedule W_t(x) = (1 + (t+1)*WA + (x+1)*WB + (t+1)*(x+1)*WC) mod P
WA = 40503                   # small fixed odd constants
WB = 50021
WC = 60013

# --- sponge-like / injection constants (analogous to Design A) ----------
G = 7                        # counter generator
D_TAG = 0x57434331           # ASCII "WCC1" = 1463898161 ; finalization tag, < P
CAP0 = 64                    # block-counter low digit (capacity cell, psi & C)
CAP1 = 65                    # finalization injection
CAP2 = 66                    # block-counter high digit

RATE = 64                    # rate cells are flat indices 0 .. RATE-1
BYTES_PER_ELEM = 3
BYTES_PER_BLOCK = RATE * BYTES_PER_ELEM   # 192

MAX_INPUT_BITS = (1 << 64) - 1

# --- domain tags (domain separation; direct base-256 injection, no hash) -
IV_TAG_PSI = b"WaveLock-CC-Core-v0:psi"     # wave-field IV tag
IV_TAG_C = b"WaveLock-CC-Core-v0:acc"       # accumulator IV tag (distinct)

# --- output -------------------------------------------------------------
OUTPUT_BITS = 256
SQUEEZE_BITS_PER_ROUND = 64

VERSION = "WaveLock-CC-Core-v0"

# internal consistency checks (run at import; cheap)
assert P == 2147483647
assert all(0 <= c < P for c in (B, D_C, GAMMA, ETA, ZETA, A_C, MU, RHO0, RHO1,
                                WA, WB, WC, D_TAG))
assert N_CELLS == 256 and BYTES_PER_BLOCK == 192
assert len({CAP0, CAP1, CAP2}) == 3 and min(CAP0, CAP1, CAP2) >= RATE
assert IV_TAG_PSI != IV_TAG_C


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
    """Round-and-position dependent weight vector W_t(x), x = 0..N_CELLS-1.

    W_t(x) = (1 + (t+1)*WA + (x+1)*WB + (t+1)*(x+1)*WC) mod P.

    Always >= 1 contribution from the constant ``1`` before reduction, but the
    reduction can land on 0 for isolated (t, x); such zeros are measure-~1/P and
    are documented, not patched, in the spec. The ``+1`` and the cross term
    (t+1)*(x+1)*WC make the schedule round-dependent (order sensitivity) and
    position-dependent (breaks lattice translation/sign symmetry).
    """
    import numpy as _np
    x = _np.arange(N_CELLS, dtype=_np.int64)
    return (1 + (t + 1) * WA + (x + 1) * WB + (t + 1) * (x + 1) * WC) % P


def round_constant(t: int) -> int:
    """rho_t = (RHO0 + RHO1 * t) mod P."""
    return (RHO0 + RHO1 * t) % P
