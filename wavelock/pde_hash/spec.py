"""WaveLock-PDE-256-v0 — frozen specification constants.

This module is the single source of truth for every constant in the primitive.
It contains NO cryptographic primitive of any kind. The values here are
normative and mirror ``docs/PDE_HASH_SPEC.md`` (§7). Changing any of them is a
version bump.

The primitive is a **finite-field polynomial dynamical system derived from the
algebraic form of the Allen-Cahn reaction-diffusion equation**
(``dpsi/dt = D * laplacian(psi) + a * psi * (b - psi^2)``), discretized over
F_p with p = 2**31 - 1 and iterated as a sponge-like absorb / evolve / squeeze
construction. The map preserves a discrete-Laplacian-plus-cubic-reaction
algebraic structure; it is NOT known to be bijective, and standard real-valued
Allen-Cahn analytical results (energy decay, diffusion, stability) do not
transfer to arithmetic over F_p. Injectivity, preimage multiplicity, state
collapse, and cycle structure are unresolved properties under test.
"""

from __future__ import annotations

# --- field --------------------------------------------------------------
P = (1 << 31) - 1            # 2**31 - 1, Mersenne prime field modulus
N = 16                       # lattice side; N*N = 256 cells

# --- PDE / reaction-diffusion coefficients ------------------------------
D = 5                        # diffusion coefficient
A = 3                        # reaction gain (spec symbol: a)
B = 1431655765               # bistable offset in [0, P) (spec symbol: b)
T = 32                       # T-round state-transformation count (provisional)

# --- sponge-like / injection constants ----------------------------------
G = 7                        # counter generator for the block-counter injection
D_TAG = 0x574C5044           # ASCII "WLPD" = 1464619076; already < P
CAP0 = 64                    # capacity cell for block-counter digit q0
CAP1 = 65                    # capacity cell for finalization injection
CAP2 = 66                    # capacity cell for block-counter digit q1

# --- rate / capacity regions (flat indices) -----------------------------
# "rate" and "capacity" are STATE-REGION NAMES ONLY. They do NOT imply any
# proven generic sponge security: the capacity is not shown to deliver
# ~(192*31) bits of collision/preimage resistance, because the T-round state
# transformation is not known to be bijective.
RATE = 64                    # rate cells are flat indices 0 .. RATE-1
N_CELLS = N * N              # 256
BYTES_PER_ELEM = 3           # 3 message bytes per field element (24 < 31 bits)
BYTES_PER_BLOCK = RATE * BYTES_PER_ELEM   # 192 message bytes per block

# --- input bound --------------------------------------------------------
# The length field is 64-bit; messages longer than this are rejected.
MAX_INPUT_BITS = (1 << 64) - 1

# --- IV domain tag ------------------------------------------------------
IV_TAG = b"WaveLock-PDE-256-v0"           # 19 bytes

# --- output -------------------------------------------------------------
OUTPUT_BITS = 256                         # fixed-output digest (NOT an XOF)
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
assert CAP0 >= RATE and CAP1 >= RATE and CAP2 >= RATE   # named cells in capacity
assert len({CAP0, CAP1, CAP2}) == 3                     # distinct named cells


def encode_block_counter(k: int):
    """Injective two-digit base-P encoding of 0-based block index ``k``.

    Returns ``(q0, q1)`` where ``q = k + 1``, ``q0 = q mod P``, ``q1 = q // P``.
    The single-digit counter ``(k+1) mod P`` repeats every P blocks; two
    base-P digits give an injective encoding for all ``q`` with ``q < P**2``.

    Coverage proof (see spec §3.5): the maximum block index under the
    ``MAX_INPUT_BITS = 2**64 - 1`` input bound is bounded by the number of
    192-byte padded blocks, which is < 2**56 << P**2 = (2**31 - 1)**2 ~ 2**62,
    so ``q1 < P`` always and the pair ``(q0, q1)`` never aliases in range.

    For ordinary messages ``q < P`` so ``q1 == 0`` and the encoding reduces to
    the original single-digit counter (existing vectors are preserved).
    """
    if k < 0:
        raise ValueError("block index must be non-negative")
    q = k + 1
    return q % P, q // P


def squeeze_pairs():
    """Return the normative SQUEEZE_PAIRS list: (t, t+128) for t in 0..63.

    Cell ``t`` (rate region) is compared against cell ``t+128`` (capacity
    region); all 128 referenced cells are distinct (spec §4).
    """
    return [(t, t + 128) for t in range(SQUEEZE_BITS_PER_ROUND)]

