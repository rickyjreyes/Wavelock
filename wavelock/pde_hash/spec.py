"""WaveLock-PDE-256-v0 — frozen specification constants.

This module is the single source of truth for every constant in the primitive.
It contains NO cryptographic primitive of any kind. The values here are
normative and mirror ``docs/PDE_HASH_SPEC.md`` (§7). Changing any of them is a
version bump.

The primitive is a finite-field Allen-Cahn reaction-diffusion PDE
(``dpsi/dt = D * laplacian(psi) + a * psi * (b - psi^2)``) discretized over
F_p with p = 2**31 - 1, run as a sponge.
"""

from __future__ import annotations

# --- field --------------------------------------------------------------
P = (1 << 31) - 1            # 2**31 - 1, Mersenne prime field modulus
N = 16                       # lattice side; N*N = 256 cells

# --- PDE / reaction-diffusion coefficients ------------------------------
D = 5                        # diffusion coefficient
A = 3                        # reaction gain (spec symbol: a)
B = 1431655765               # bistable offset in [0, P) (spec symbol: b)
T = 32                       # PDE rounds per permutation Phi_P^T (provisional)

# --- sponge / injection constants ---------------------------------------
G = 7                        # counter generator for inject_counter
D_TAG = 0x574C5044           # ASCII "WLPD" = 1464619076; already < P
CAP0 = 64                    # capacity cell for block-counter injection
CAP1 = 65                    # capacity cell for finalization injection

# --- rate / capacity regions (flat indices) -----------------------------
RATE = 64                    # rate cells are flat indices 0 .. RATE-1
N_CELLS = N * N              # 256
BYTES_PER_ELEM = 3           # 3 message bytes per field element (24 < 31 bits)
BYTES_PER_BLOCK = RATE * BYTES_PER_ELEM   # 192 message bytes per block

# --- IV domain tag ------------------------------------------------------
IV_TAG = b"WaveLock-PDE-256-v0"           # 19 bytes

# --- output -------------------------------------------------------------
DEFAULT_OUTPUT_BITS = 256
SQUEEZE_BITS_PER_ROUND = 64               # one comparison per SQUEEZE_PAIRS entry

VERSION = "WaveLock-PDE-256-v0"

# Sanity checks that the frozen constants are internally consistent. These
# are cheap and run at import; they protect against a typo silently changing
# the primitive.
assert P == 2147483647
assert 0 <= B < P
assert 0 <= D_TAG < P
assert 0 <= D < P and 0 <= A < P and 0 <= G < P
assert N_CELLS == 256
assert BYTES_PER_BLOCK == 192
assert CAP0 >= RATE and CAP1 >= RATE      # named cells live in the capacity region


def squeeze_pairs():
    """Return the normative SQUEEZE_PAIRS list: (t, t+128) for t in 0..63.

    Cell ``t`` (rate region) is compared against cell ``t+128`` (capacity
    region); all 128 referenced cells are distinct (spec §4).
    """
    return [(t, t + 128) for t in range(SQUEEZE_BITS_PER_ROUND)]
