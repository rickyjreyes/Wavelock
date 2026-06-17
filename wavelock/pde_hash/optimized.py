"""Independent NumPy implementation of WaveLock-PDE-256-v0.

This implementation deliberately shares NO core evolution code with the
pure-Python reference (``evolve.py``/``absorb.py``/``squeeze.py``). It only
imports frozen *constants* from ``spec`` (data, not algorithm) and reimplements
padding, packing, the PDE round, the sponge loop, and the squeeze from scratch
in NumPy. Phase-7 parity tests require byte-identical agreement with the
reference, including round-granular intermediate snapshots.

Exact arithmetic: F_p with p = 2**31 - 1. All products of two residues are
< 2**62 < 2**63, so int64 holds every intermediate exactly; reduction is plain
``% P``. No floating point, no cryptographic primitive.
"""

from __future__ import annotations

import numpy as np

from . import spec

_P = np.int64(spec.P)
_N = spec.N
_D = np.int64(spec.D)
_A = np.int64(spec.A)
_B = np.int64(spec.B)
_PM4 = np.int64((spec.P - 4) % spec.P)


# ---------------------------------------------------------------------------
# byte layer (reimplemented independently of absorb.py)
# ---------------------------------------------------------------------------
def _pad(message: bytes) -> bytes:
    m = bytes(message)
    if len(m) * 8 > spec.MAX_INPUT_BITS:
        raise ValueError("message too long: exceeds MAX_INPUT_BITS")
    out = bytearray(m)
    out.append(1)
    block = spec.BYTES_PER_BLOCK
    pad_to = (-len(out)) % block
    out.extend(b"\x00" * pad_to)
    lblock = bytearray(block)
    lblock[0:8] = (len(m) * 8).to_bytes(8, "big")
    out.extend(lblock)
    return bytes(out)


def _blocks_to_elems(padded: bytes) -> np.ndarray:
    """Return an (n_blocks, RATE) int64 array of packed field elements."""
    arr = np.frombuffer(padded, dtype=np.uint8).astype(np.int64)
    n_blocks = arr.size // spec.BYTES_PER_BLOCK
    arr = arr.reshape(n_blocks, spec.RATE, 3)
    elems = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)
    return elems  # (n_blocks, RATE), int64


# ---------------------------------------------------------------------------
# state / IV (independent of state.py)
# ---------------------------------------------------------------------------
def _iv() -> np.ndarray:
    tag = np.frombuffer(spec.IV_TAG, dtype=np.uint8).astype(np.int64)
    idx = np.arange(spec.N_CELLS, dtype=np.int64)
    flat = (1 + idx + tag[idx % tag.size]) % _P
    return flat.reshape(_N, _N).copy()


# ---------------------------------------------------------------------------
# PDE round (independent NumPy reimplementation of evolve.py)
# ---------------------------------------------------------------------------
def _one_round(psi: np.ndarray) -> np.ndarray:
    p = _P
    # reaction: a * psi * (b - psi^2)
    sq = (psi * psi) % p
    bm = (_B + (p - sq)) % p
    react = (_A * ((psi * bm) % p)) % p
    # diffusion: 5-point toroidal Laplacian via roll
    lap = (
        np.roll(psi, -1, axis=0)
        + np.roll(psi, 1, axis=0)
        + np.roll(psi, -1, axis=1)
        + np.roll(psi, 1, axis=1)
        + (_PM4 * psi) % p
    ) % p
    return (psi + (_D * lap) % p + react) % p


def _evolve_T(psi: np.ndarray) -> np.ndarray:
    """T-round state transformation (not a proven permutation)."""
    for _ in range(spec.T):
        psi = _one_round(psi)
    return psi


# ---------------------------------------------------------------------------
# sponge (independent of absorb.py / squeeze.py)
# ---------------------------------------------------------------------------
_PAIRS_A = np.array([a for a, _ in spec.squeeze_pairs()], dtype=np.int64)
_PAIRS_B = np.array([b for _, b in spec.squeeze_pairs()], dtype=np.int64)


def _absorb(message: bytes) -> np.ndarray:
    elems = _blocks_to_elems(_pad(message))
    n_blocks = elems.shape[0]
    psi = _iv()
    p = _P
    for k in range(n_blocks):
        flat = psi.reshape(-1)
        flat[: spec.RATE] = (flat[: spec.RATE] + elems[k]) % p
        q0, q1 = spec.encode_block_counter(k)
        flat[spec.CAP0] = (flat[spec.CAP0] + np.int64(q0) * np.int64(spec.G)) % p
        flat[spec.CAP2] = (flat[spec.CAP2] + np.int64(q1) * np.int64(spec.G)) % p
        if k == n_blocks - 1:
            flat[spec.CAP1] = (flat[spec.CAP1] + np.int64(spec.D_TAG)) % p
        psi = _evolve_T(flat.reshape(_N, _N))
    return psi


def _squeeze(psi: np.ndarray, output_bits: int) -> bytes:
    flat = psi.reshape(-1)
    bits = []
    while len(bits) < output_bits:
        cmp = (flat[_PAIRS_A] > flat[_PAIRS_B]).astype(np.uint8)
        bits.extend(int(x) for x in cmp)
        if len(bits) < output_bits:
            psi = _evolve_T(psi)
            flat = psi.reshape(-1)
    bits = bits[:output_bits]
    out = bytearray(output_bits // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(out)


def _snapshot_bytes(psi: np.ndarray) -> bytes:
    """Serialize 256 residues as 256 big-endian uint32 (matches state.py)."""
    flat = psi.reshape(-1).astype(">u4")
    return flat.tobytes()


def pde_hash(message: bytes) -> bytes:
    """Fixed-output 256-bit digest (NOT an XOF). Always returns 32 bytes."""
    if not isinstance(message, (bytes, bytearray)):
        raise TypeError("message must be bytes")
    psi = _absorb(bytes(message))
    return _squeeze(psi, spec.OUTPUT_BITS)


def trace(message: bytes) -> dict:
    """Independent NumPy reproduction of reference.trace (Phase-7 parity)."""
    p = _P
    elems = _blocks_to_elems(_pad(bytes(message)))
    n_blocks = elems.shape[0]

    snaps = {}
    psi = _iv()
    snaps["S_iv"] = _snapshot_bytes(psi)

    for k in range(n_blocks):
        flat = psi.reshape(-1).copy()
        flat[: spec.RATE] = (flat[: spec.RATE] + elems[k]) % p
        q0, q1 = spec.encode_block_counter(k)
        flat[spec.CAP0] = (flat[spec.CAP0] + np.int64(q0) * np.int64(spec.G)) % p
        flat[spec.CAP2] = (flat[spec.CAP2] + np.int64(q1) * np.int64(spec.G)) % p
        if k == n_blocks - 1:
            flat[spec.CAP1] = (flat[spec.CAP1] + np.int64(spec.D_TAG)) % p
        pre = flat.reshape(_N, _N)
        if k == 0:
            snaps["S_absorb0"] = _snapshot_bytes(pre)
        psi = _evolve_T(pre)
        if k == 0:
            snaps["S_perm0"] = _snapshot_bytes(psi)

    snaps["S_final"] = _snapshot_bytes(psi)
    snaps["S_squeeze1"] = _snapshot_bytes(_evolve_T(psi))
    snaps["digest"] = _squeeze(psi, spec.OUTPUT_BITS)
    return snaps


__all__ = ["pde_hash", "trace"]
