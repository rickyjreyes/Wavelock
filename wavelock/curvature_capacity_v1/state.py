"""CCStateV1: coupled (wave, accumulator) lattice state and IVs for Candidate B.

Identical structure to Candidate A's state; the IVs share Candidate A's byte
values deliberately (controlled comparison). No hash is used.
"""

from __future__ import annotations

from typing import List

from . import spec


class CCStateV1:
    """A coupled state: two flat ``list[int]`` of length N_CELLS in [0, P)."""

    __slots__ = ("psi", "C")

    def __init__(self, psi: List[int], C: List[int]):
        if len(psi) != spec.N_CELLS or len(C) != spec.N_CELLS:
            raise ValueError(f"expected {spec.N_CELLS} cells in each field")
        self.psi = [int(c) % spec.P for c in psi]
        self.C = [int(c) % spec.P for c in C]

    def copy(self) -> "CCStateV1":
        return CCStateV1(list(self.psi), list(self.C))

    def __eq__(self, other) -> bool:
        return (isinstance(other, CCStateV1)
                and self.psi == other.psi and self.C == other.C)

    def snapshot_bytes(self) -> bytes:
        out = bytearray(spec.N_CELLS * 8)
        for idx, v in enumerate(self.psi):
            out[4 * idx:4 * idx + 4] = int(v).to_bytes(4, "big")
        off = spec.N_CELLS * 4
        for idx, v in enumerate(self.C):
            out[off + 4 * idx:off + 4 * idx + 4] = int(v).to_bytes(4, "big")
        return bytes(out)


def _iv_from_tag(tag: bytes) -> List[int]:
    cells = []
    for idx in range(spec.N_CELLS):
        cells.append((1 + idx + tag[idx % len(tag)]) % spec.P)
    return cells


def initial_psi() -> List[int]:
    return _iv_from_tag(spec.IV_TAG_PSI)


def initial_C() -> List[int]:
    return _iv_from_tag(spec.IV_TAG_C)


def initial_state() -> CCStateV1:
    return CCStateV1(initial_psi(), initial_C())
