"""Message absorption A: bytes -> padded blocks -> evolved state (reference).

No cryptographic primitive is used. Padding is an injective 10*-style rule plus
a dedicated trailing length block (spec §3.3). Bytes are packed 3-per-field-
element (spec §3.2). Absorption is additive into the rate cells, with a
two-digit base-P per-block counter injection and a finalization domain
injection (spec §3.4/§3.5), inside a sponge-like absorb/evolve/squeeze
construction (the T-round evolution is not a proven permutation).
"""

from __future__ import annotations

from typing import List

from . import spec
from .state import PDEState, initial_state
from .evolve import evolve_T


def pad(message: bytes) -> bytes:
    """Apply the injective padding of spec §3.3.

    1. append 0x01
    2. append 0x00 until length is a multiple of 192
    3. append one 192-byte length block: first 8 bytes = bit length (big-endian)

    Rejects messages whose bit length exceeds MAX_INPUT_BITS (2**64 - 1).
    """
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("message must be bytes")
    m = bytes(message)
    lbits = len(m) * 8
    if lbits > spec.MAX_INPUT_BITS:
        raise ValueError(
            f"message too long: {lbits} bits exceeds MAX_INPUT_BITS "
            f"({spec.MAX_INPUT_BITS})"
        )

    out = bytearray(m)
    out.append(0x01)
    while len(out) % spec.BYTES_PER_BLOCK != 0:
        out.append(0x00)

    length_block = bytearray(spec.BYTES_PER_BLOCK)
    length_block[0:8] = lbits.to_bytes(8, "big")
    out.extend(length_block)
    return bytes(out)


def pack_block(padded: bytes, k: int) -> List[int]:
    """Convert byte-block k of the padded stream into 64 field elements.

    elem(b0,b1,b2) = b0 + 256*b1 + 65536*b2   (spec §3.2)
    """
    base = k * spec.BYTES_PER_BLOCK
    elems = []
    for c in range(spec.RATE):
        off = base + 3 * c
        b0 = padded[off]
        b1 = padded[off + 1]
        b2 = padded[off + 2]
        elems.append(b0 + (b1 << 8) + (b2 << 16))
    return elems


def absorb(message: bytes) -> PDEState:
    """Run the full sponge absorption loop and return the pre-squeeze state."""
    padded = pad(message)
    n_blocks = len(padded) // spec.BYTES_PER_BLOCK

    state = initial_state()
    cells = state.cells  # mutate in place; permute() returns fresh lists
    p = spec.P

    for k in range(n_blocks):
        block = pack_block(padded, k)
        # additive rate write (cells 0 .. RATE-1)
        for c in range(spec.RATE):
            cells[c] = (cells[c] + block[c]) % p
        # two-digit base-P block-counter injection (cap0 = low digit, cap2 = high)
        q0, q1 = spec.encode_block_counter(k)
        cells[spec.CAP0] = (cells[spec.CAP0] + q0 * spec.G) % p
        cells[spec.CAP2] = (cells[spec.CAP2] + q1 * spec.G) % p
        # finalization injection (cap1) on the last block only
        if k == n_blocks - 1:
            cells[spec.CAP1] = (cells[spec.CAP1] + spec.D_TAG) % p
        # T-round state transformation
        state = evolve_T(PDEState(cells))
        cells = state.cells

    return state
