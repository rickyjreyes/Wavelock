"""PDEState: the lattice state container and the public IV.

Pure-Python representation: a flat list of 256 integers in [0, P), indexed by
flat index ``idx = i*N + j``. No cryptographic primitive is used to build the
IV; it is direct base-256 injection of the domain tag (spec §2.1).
"""

from __future__ import annotations

from typing import List

from . import spec


class PDEState:
    """A WaveLock-PDE lattice state: N*N residues in [0, P).

    Stored as a flat ``list[int]`` of length ``N_CELLS`` in row-major order.
    This is the reference (pure-Python) representation. The optimized NumPy
    implementation carries its own array representation but agrees on the same
    integer residues at every snapshot point.
    """

    __slots__ = ("cells",)

    def __init__(self, cells: List[int]):
        if len(cells) != spec.N_CELLS:
            raise ValueError(f"expected {spec.N_CELLS} cells, got {len(cells)}")
        # Defensive copy as plain ints; callers may pass any iterable of ints.
        self.cells = [int(c) % spec.P for c in cells]

    def copy(self) -> "PDEState":
        return PDEState(list(self.cells))

    def __eq__(self, other) -> bool:
        return isinstance(other, PDEState) and self.cells == other.cells

    def snapshot_bytes(self) -> bytes:
        """Serialize the 256 residues as 256 big-endian uint32 (1024 bytes).

        Used ONLY for test-vector bookkeeping (spec §9). Never fed back into
        the primitive.
        """
        out = bytearray(spec.N_CELLS * 4)
        for idx, v in enumerate(self.cells):
            out[4 * idx + 0] = (v >> 24) & 0xFF
            out[4 * idx + 1] = (v >> 16) & 0xFF
            out[4 * idx + 2] = (v >> 8) & 0xFF
            out[4 * idx + 3] = v & 0xFF
        return bytes(out)


def initial_state() -> PDEState:
    """Build the public IV by direct base-256 injection of the domain tag.

    IV[i,j] = (1 + i*N + j + tag[(i*N + j) mod len(tag)]) mod P    (spec §2.1)
    """
    tag = spec.IV_TAG
    n = spec.N
    cells = []
    for idx in range(spec.N_CELLS):
        # idx == i*N + j directly, so we use idx for the linear term.
        v = (1 + idx + tag[idx % len(tag)]) % spec.P
        cells.append(v)
    return PDEState(cells)
