"""Readable pure-Python reference for the WaveLock Curvature-Capacity Core.

Slow, obviously-correct implementation. ``optimized.py`` must agree byte-for-
byte (parity test). No conventional cryptographic primitive is imported.

Pipeline:  absorb (multi-block, coupled psi+C) -> squeeze (from accumulator C).

A separate ``trajectory_digest`` entry point starts from an arbitrary wave state
psi0 and the accumulator IV; it is the *state-level* commitment used to test
whether the Design A eigenmode collisions (psi = s*sigma, all mapping to the
terminal state 0) remain digest collisions once the trajectory is bound.
"""

from __future__ import annotations

from typing import List, Optional

from . import spec
from .state import CCState, initial_state, initial_C
from .evolve import coupled_evolve_T, coupled_one_round


# --- message padding / packing (identical structure to Design A) --------
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


def _inject_block(state: CCState, block: List[int], k: int, last: bool) -> None:
    """Additive rate write + counter/finalization into BOTH fields (mutates)."""
    p = spec.P
    for c in range(spec.RATE):
        state.psi[c] = (state.psi[c] + block[c]) % p
        # the accumulator rate cells also receive the message (domain separated
        # by a fixed multiplier so psi and C never see the identical injection)
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
    """Run the coupled sponge absorption; return (CCState, next_round_index)."""
    padded = pad(message)
    nb = len(padded) // spec.BYTES_PER_BLOCK
    state = initial_state()
    ri = 0
    for k in range(nb):
        block = pack_block(padded, k)
        _inject_block(state, block, k, last=(k == nb - 1))
        state, ri = coupled_evolve_T(state, ri)
    return state, ri


def squeeze(state: CCState, ri: int, output_bits: int = spec.OUTPUT_BITS) -> bytes:
    """Read comparison bits from the accumulator C; re-evolve the coupled
    system between squeeze rounds (so the squeeze is part of the dynamics)."""
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
    """Compute the fixed 256-bit curvature-capacity digest."""
    state, ri = absorb(message)
    return squeeze(state, ri, spec.OUTPUT_BITS)


def trajectory_digest(psi0: List[int], C0: Optional[List[int]] = None,
                      rounds: Optional[int] = None,
                      output_bits: int = spec.OUTPUT_BITS) -> bytes:
    """State-level commitment: evolve the coupled system from (psi0, C0) for
    ``rounds`` (default T) rounds, then squeeze from C.

    With C0 = accumulator IV, two wave states that share a terminal wave state
    (e.g. the Design A eigenmode collisions, all -> 0) generically yield
    *different* trajectory digests, because their accumulator paths differ. That
    is exactly the property tested in curvature_audit/eigenmode_attacks.py.
    """
    C = list(C0) if C0 is not None else initial_C()
    state = CCState(list(psi0), C)
    r = spec.T if rounds is None else rounds
    state, ri = coupled_evolve_T(state, 0, r)
    return squeeze(state, ri, output_bits)


def trace(message: bytes) -> dict:
    """Named coupled snapshots for parity testing."""
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
