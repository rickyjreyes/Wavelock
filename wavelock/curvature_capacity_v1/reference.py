"""Readable pure-Python reference for CC-Core-v1 / Candidate B.

Identical message/sponge structure to Candidate A; only the accumulator
injection differs (see commit.py). ``optimized.py`` must agree byte-for-byte.
No conventional cryptographic primitive is imported.
"""

from __future__ import annotations

from typing import List, Optional

from . import spec
from .state import CCStateV1, initial_state, initial_C
from .evolve import coupled_evolve_T


def pad(message: bytes) -> bytes:
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("message must be bytes")
    m = bytes(message)
    lbits = len(m) * 8
    if lbits > spec.MAX_INPUT_BITS:
        raise ValueError("message too long")
    out = bytearray(m)
    out.append(0x01)
    while len(out) % spec.BYTES_PER_BLOCK != 0:
        out.append(0x00)
    lb = bytearray(spec.BYTES_PER_BLOCK)
    lb[0:8] = lbits.to_bytes(8, "big")
    out.extend(lb)
    return bytes(out)


def pack_block(padded: bytes, k: int) -> List[int]:
    base = k * spec.BYTES_PER_BLOCK
    elems = []
    for c in range(spec.RATE):
        off = base + 3 * c
        elems.append(padded[off] + (padded[off + 1] << 8) + (padded[off + 2] << 16))
    return elems


def _inject_block(state: CCStateV1, block: List[int], k: int, last: bool) -> None:
    p = spec.P
    for c in range(spec.RATE):
        state.psi[c] = (state.psi[c] + block[c]) % p
        state.C[c] = (state.C[c] + spec.G * block[c]) % p
    q0, q1 = spec.encode_block_counter(k)
    state.psi[spec.CAP0] = (state.psi[spec.CAP0] + q0 * spec.G) % p
    state.psi[spec.CAP2] = (state.psi[spec.CAP2] + q1 * spec.G) % p
    state.C[spec.CAP0] = (state.C[spec.CAP0] + q0 * spec.G) % p
    state.C[spec.CAP2] = (state.C[spec.CAP2] + q1 * spec.G) % p
    if last:
        state.psi[spec.CAP1] = (state.psi[spec.CAP1] + spec.D_TAG) % p
        state.C[spec.CAP1] = (state.C[spec.CAP1] + spec.D_TAG) % p


def absorb(message: bytes):
    padded = pad(message)
    nb = len(padded) // spec.BYTES_PER_BLOCK
    state = initial_state()
    ri = 0
    for k in range(nb):
        block = pack_block(padded, k)
        _inject_block(state, block, k, last=(k == nb - 1))
        state, ri = coupled_evolve_T(state, ri)
    return state, ri


def squeeze(state: CCStateV1, ri: int, output_bits: int = spec.OUTPUT_BITS) -> bytes:
    pairs = spec.squeeze_pairs()
    bits: List[int] = []
    s = state
    while len(bits) < output_bits:
        for (a, b) in pairs:
            bits.append(1 if s.C[a] > s.C[b] else 0)
        if len(bits) < output_bits:
            s, ri = coupled_evolve_T(s, ri)
    bits = bits[:output_bits]
    out = bytearray((output_bits + 7) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(out)


def cc_hash(message: bytes) -> bytes:
    state, ri = absorb(message)
    return squeeze(state, ri, spec.OUTPUT_BITS)


def trajectory_digest(psi0: List[int], C0: Optional[List[int]] = None,
                      rounds: Optional[int] = None,
                      output_bits: int = spec.OUTPUT_BITS) -> bytes:
    C = list(C0) if C0 is not None else initial_C()
    state = CCStateV1(list(psi0), C)
    r = spec.T if rounds is None else rounds
    state, ri = coupled_evolve_T(state, 0, r)
    return squeeze(state, ri, output_bits)


def trace(message: bytes) -> dict:
    padded = pad(message)
    nb = len(padded) // spec.BYTES_PER_BLOCK
    snaps = {}
    state = initial_state()
    snaps["S_iv"] = state.snapshot_bytes()
    ri = 0
    for k in range(nb):
        block = pack_block(padded, k)
        _inject_block(state, block, k, last=(k == nb - 1))
        if k == 0:
            snaps["S_absorb0"] = state.snapshot_bytes()
        state, ri = coupled_evolve_T(state, ri)
        if k == 0:
            snaps["S_round0"] = state.snapshot_bytes()
    snaps["S_final"] = state.snapshot_bytes()
    snaps["digest"] = squeeze(state, ri)
    return snaps


__all__ = ["cc_hash", "trajectory_digest", "trace", "absorb", "squeeze", "pad"]
