"""Squeeze Q: evolved state -> output bits (reference).

PDE-native squeeze by canonical cell-pair comparison (sign of a field
difference), with intermediate T-round state transformations between squeeze
rounds (spec §4). No cryptographic primitive is used.

The normative digest is fixed at spec.OUTPUT_BITS (256). The ``output_bits``
parameter of ``squeeze`` is retained ONLY as an internal audit knob (e.g. for
truncated-collision experiments); it does NOT make WaveLock-PDE-256-v0 an XOF
and is not exposed by the public ``pde_hash`` digest.
"""

from __future__ import annotations

from . import spec
from .state import PDEState
from .evolve import evolve_T

_PAIRS = spec.squeeze_pairs()


def _pack_msb_first(bits) -> bytes:
    """Pack 0/1 ints MSB-first: bits[0] is bit 7 of byte 0 (spec §4)."""
    if len(bits) % 8 != 0:
        raise ValueError("bit count must be a multiple of 8")
    out = bytearray(len(bits) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(out)


def squeeze(state: PDEState, output_bits: int = spec.OUTPUT_BITS) -> bytes:
    """Read ``output_bits`` bits from the evolved state (spec §4).

    ``output_bits`` is an internal audit knob; the normative public digest
    always uses spec.OUTPUT_BITS (256) via ``reference.pde_hash``.
    """
    if output_bits <= 0 or output_bits % spec.SQUEEZE_BITS_PER_ROUND != 0:
        raise ValueError(
            f"output_bits must be a positive multiple of "
            f"{spec.SQUEEZE_BITS_PER_ROUND}"
        )

    cells = state.cells
    bits = []
    while len(bits) < output_bits:
        for a_cell, b_cell in _PAIRS:
            bits.append(1 if cells[a_cell] > cells[b_cell] else 0)
        if len(bits) < output_bits:
            # re-evolve only when more bits are needed
            state = evolve_T(state)
            cells = state.cells

    return _pack_msb_first(bits[:output_bits])
