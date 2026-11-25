"""Implementation-side regression for the first-block field facts proved in Lean.

This checks the *frozen Candidate B Python reference* against the exact
integer/mod-p embedding in WaveLockProofs.ExactField.lean. It is not a
cross-language equivalence proof, a later-round reachability theorem, or
a cryptographic security test.
"""

from __future__ import annotations

import pytest

from wavelock.curvature_capacity_v1 import reference, spec
from wavelock.curvature_capacity_v1.state import initial_state


@pytest.mark.parametrize("length", [0, 1, 2, 190, 191, 192, 193, 383, 384, 1025])
def test_exact_padding_length_block_and_nonfinal_first_block(length: int) -> None:
    message = bytes((73 * i + 19) % 256 for i in range(length))
    padded = reference.pad(message)
    count = len(padded) // spec.BYTES_PER_BLOCK

    assert len(padded) % spec.BYTES_PER_BLOCK == 0
    assert count >= 2
    assert count == (length + spec.BYTES_PER_BLOCK) // spec.BYTES_PER_BLOCK + 1
    assert padded[:length] == message
    assert padded[length] == 1
    assert all(b == 0 for b in padded[length + 1 : -spec.BYTES_PER_BLOCK])
    length_block = padded[-spec.BYTES_PER_BLOCK :]
    assert length_block[:8] == (8 * length).to_bytes(8, "big")
    assert length_block[8:] == bytes(spec.BYTES_PER_BLOCK - 8)


@pytest.mark.parametrize(
    "message",
    [
        b"",
        b"\x00",
        b"\xff",
        bytes(range(192)),
        bytes([255] * 192),
        bytes([0] * 193),
        bytes((i * 41) % 256 for i in range(400)),
    ],
)
def test_reference_first_block_matches_bounded_field_embedding(message: bytes) -> None:
    padded = reference.pad(message)
    elements = reference.pack_block(padded, 0)
    state = initial_state()
    psi_iv, c_iv = state.psi.copy(), state.C.copy()

    # The first block is never final, including the empty message.
    assert len(padded) >= 2 * spec.BYTES_PER_BLOCK
    reference._inject_block(state, elements, 0, last=False)

    for cell in range(spec.RATE):
        offset = 3 * cell
        triple = padded[offset : offset + 3]
        packed = triple[0] + (triple[1] << 8) + (triple[2] << 16)
        assert elements[cell] == packed
        assert 0 <= packed < (1 << 24)
        assert 0 <= psi_iv[cell] < 375

        integer_absorption = psi_iv[cell] + packed
        assert integer_absorption < spec.P
        assert integer_absorption < spec.V_STAR
        assert state.psi[cell] == integer_absorption % spec.P
        assert state.C[cell] == (c_iv[cell] + spec.G * packed) % spec.P

    # Capacity index 65 only receives the domain tag on the LAST block.
    assert state.psi[spec.CAP1] == psi_iv[spec.CAP1]
    assert state.C[spec.CAP1] == c_iv[spec.CAP1]

    # The 0-based first-block counter is (1, 0) in base p.
    assert spec.encode_block_counter(0) == (1, 0)
    assert state.psi[spec.CAP0] == (psi_iv[spec.CAP0] + spec.G) % spec.P
    assert state.C[spec.CAP0] == (c_iv[spec.CAP0] + spec.G) % spec.P
    assert state.psi[spec.CAP2] == psi_iv[spec.CAP2]
    assert state.C[spec.CAP2] == c_iv[spec.CAP2]

    # Absorption never directly writes wave coordinates 67..255.
    assert state.psi[67:] == psi_iv[67:]
    assert all(v != spec.V_STAR for v in state.psi)
