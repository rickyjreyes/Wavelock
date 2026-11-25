"""CCState: the coupled (wave, accumulator) lattice state and the public IVs.

The curvature-capacity core carries two N*N fields over F_p:

  * ``psi`` -- the wave field, evolved by the *unmodified* Design A round F;
  * ``C``   -- the path-binding accumulator, co-evolved by Phi_t (commit.py).

Both IVs are built by direct base-256 injection of distinct domain tags (no
hash), so they are domain-separated public constants.
"""

from __future__ import annotations

from typing import List

from . import spec


class CCState:
    """A coupled state: two flat ``list[int]`` of length N_CELLS in [0, P)."""

    __slots__ = ("psi", "C")

    def __init__(self, psi: List[int], C: List[int]):
        if len(psi) != spec.N_CELLS or len(C) != spec.N_CELLS:
            raise ValueError(f"expected {spec.N_CELLS} cells in each field")
        self.psi = [int(c) % spec.P for c in psi]
        self.C = [int(c) % spec.P for c in C]

    def copy(self) -> "CCState":
        return CCState(list(self.psi), list(self.C))

    def __eq__(self, other) -> bool:
        return (isinstance(other, CCState)
                and self.psi == other.psi and self.C == other.C)

    def snapshot_bytes(self) -> bytes:
        """256 BE-uint32 of psi followed by 256 BE-uint32 of C (2048 bytes).

        Test-vector bookkeeping only; never fed back into the primitive.
        """
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
    """Wave-field IV by base-256 injection of IV_TAG_PSI (no hash)."""
    return _iv_from_tag(spec.IV_TAG_PSI)


def initial_C() -> List[int]:
    """Accumulator IV by base-256 injection of IV_TAG_C (no hash)."""
    return _iv_from_tag(spec.IV_TAG_C)


def initial_state() -> CCState:
    return CCState(initial_psi(), initial_C())
