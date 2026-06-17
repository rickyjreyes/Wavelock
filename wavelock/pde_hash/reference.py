"""Readable pure-Python reference implementation of WaveLock-PDE-256-v0.

This is the canonical, slow, obviously-correct implementation. The NumPy
``optimized.py`` must agree with it byte-for-byte (Phase 7). No cryptographic
primitive is imported anywhere in this package.
"""

from __future__ import annotations

from . import spec
from .absorb import absorb, pad, pack_block
from .squeeze import squeeze
from .state import PDEState, initial_state
from .evolve import permute


def pde_hash(message: bytes, *, output_bits: int = spec.DEFAULT_OUTPUT_BITS) -> bytes:
    """Compute H_PDE(message) = Q(Phi_P^T(A(message)))."""
    state = absorb(message)
    return squeeze(state, output_bits)


def trace(message: bytes) -> dict:
    """Return named round-granular snapshots for Phase-7 parity (spec §9).

    Re-runs the sponge using the public building blocks while capturing the
    snapshots S_iv, S_absorb0, S_perm0, S_final, S_squeeze1, plus the digest.
    All snapshots are encoded with PDEState.snapshot_bytes (256 BE uint32).
    """
    p = spec.P
    padded = pad(message)
    n_blocks = len(padded) // spec.BYTES_PER_BLOCK

    snaps = {}
    state = initial_state()
    snaps["S_iv"] = state.snapshot_bytes()

    for k in range(n_blocks):
        cells = list(state.cells)
        block = pack_block(padded, k)
        for c in range(spec.RATE):
            cells[c] = (cells[c] + block[c]) % p
        cells[spec.CAP0] = (cells[spec.CAP0] + ((k + 1) % p) * spec.G) % p
        if k == n_blocks - 1:
            cells[spec.CAP1] = (cells[spec.CAP1] + spec.D_TAG) % p
        pre = PDEState(cells)
        if k == 0:
            snaps["S_absorb0"] = pre.snapshot_bytes()
        state = permute(pre)
        if k == 0:
            snaps["S_perm0"] = state.snapshot_bytes()

    snaps["S_final"] = state.snapshot_bytes()
    snaps["S_squeeze1"] = permute(state).snapshot_bytes()
    snaps["digest"] = squeeze(state, spec.DEFAULT_OUTPUT_BITS)
    return snaps


__all__ = ["pde_hash", "trace", "absorb", "squeeze", "PDEState"]
