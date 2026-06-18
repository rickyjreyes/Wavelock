"""NumPy implementation of the WaveLock Curvature-Capacity Core.

Independent of ``reference.py`` (shares only ``spec``); parity tests require
byte-for-byte agreement, including the wave round matching the *Design A*
optimized round. No conventional cryptographic primitive is imported.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import spec

_P = spec.P
_N = spec.N
_PM4 = (_P - 4) % _P


def _wave_round(psi: np.ndarray) -> np.ndarray:
    """One Design A wave round (NumPy), identical to wavelock.pde_hash."""
    p = _P
    sq = (psi * psi) % p
    bm = (spec.B + (p - sq)) % p
    react = (spec.A * ((psi * bm) % p)) % p
    lap = (
        np.roll(psi, -1, 0) + np.roll(psi, 1, 0)
        + np.roll(psi, -1, 1) + np.roll(psi, 1, 1)
        + (_PM4 * psi) % p
    ) % p
    return (psi + (spec.D * lap) % p + react) % p


def _lap_C(C: np.ndarray) -> np.ndarray:
    p = _P
    return (
        np.roll(C, -1, 0) + np.roll(C, 1, 0)
        + np.roll(C, -1, 1) + np.roll(C, 1, 1)
        + (_PM4 * C) % p
    ) % p


# precompute the (t-independent) position-weight components as flat vectors
_X = np.arange(spec.N_CELLS, dtype=np.int64)


def _weights(t: int) -> np.ndarray:
    t1 = t + 1
    return (1 + t1 * spec.WA + (_X + 1) * spec.WB + t1 * (_X + 1) * spec.WC) % _P


def _accumulator_step(C: np.ndarray, psi_t: np.ndarray, psi_next: np.ndarray,
                      t: int) -> np.ndarray:
    p = _P
    rho = spec.round_constant(t)
    u = psi_t.reshape(-1)
    v = psi_next.reshape(-1)
    j = (u + spec.GAMMA * ((u * v) % p) + spec.ETA * ((u * u) % p) + spec.ZETA * v) % p
    Cd = _lap_C(C).reshape(-1)
    cd = (C.reshape(-1) + (spec.D_C * Cd) % p) % p
    w = _weights(t)
    out = (spec.MU * cd + (spec.A_C * ((cd * cd) % p)) % p + (w * j) % p + rho) % p
    return out.reshape(_N, _N)


def _coupled_round(psi: np.ndarray, C: np.ndarray, t: int):
    psi_next = _wave_round(psi)
    C_next = _accumulator_step(C, psi, psi_next, t)
    return psi_next, C_next


def _coupled_evolve_T(psi, C, start_round, rounds=None):
    r = spec.T if rounds is None else rounds
    ri = start_round
    for _ in range(r):
        psi, C = _coupled_round(psi, C, ri)
        ri += 1
    return psi, C, ri


# --- IVs ----------------------------------------------------------------
def _iv(tag: bytes) -> np.ndarray:
    t = np.frombuffer(tag, dtype=np.uint8).astype(np.int64)
    idx = np.arange(spec.N_CELLS, dtype=np.int64)
    return ((1 + idx + t[idx % t.size]) % _P).reshape(_N, _N).copy()


def iv_psi() -> np.ndarray:
    return _iv(spec.IV_TAG_PSI)


def iv_C() -> np.ndarray:
    return _iv(spec.IV_TAG_C)


# --- message layer ------------------------------------------------------
def _pad(message: bytes) -> bytes:
    m = bytes(message)
    if len(m) * 8 > spec.MAX_INPUT_BITS:
        raise ValueError("message too long")
    out = bytearray(m)
    out.append(1)
    out.extend(b"\x00" * ((-len(out)) % spec.BYTES_PER_BLOCK))
    lb = bytearray(spec.BYTES_PER_BLOCK)
    lb[0:8] = (len(m) * 8).to_bytes(8, "big")
    out.extend(lb)
    return bytes(out)


def absorb(message: bytes):
    p = _P
    padded = _pad(message)
    arr = np.frombuffer(padded, dtype=np.uint8).astype(np.int64)
    nb = arr.size // spec.BYTES_PER_BLOCK
    arr = arr.reshape(nb, spec.RATE, 3)
    elems = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)
    psi = iv_psi()
    C = iv_C()
    ri = 0
    for k in range(nb):
        pf = psi.reshape(-1)
        cf = C.reshape(-1)
        pf[:spec.RATE] = (pf[:spec.RATE] + elems[k]) % p
        cf[:spec.RATE] = (cf[:spec.RATE] + spec.G * elems[k]) % p
        q = k + 1
        q0, q1 = q % p, q // p
        for f in (pf, cf):
            f[spec.CAP0] = (f[spec.CAP0] + q0 * spec.G) % p
            f[spec.CAP2] = (f[spec.CAP2] + q1 * spec.G) % p
        if k == nb - 1:
            pf[spec.CAP1] = (pf[spec.CAP1] + spec.D_TAG) % p
            cf[spec.CAP1] = (cf[spec.CAP1] + spec.D_TAG) % p
        psi, C, ri = _coupled_evolve_T(psi.reshape(_N, _N), C.reshape(_N, _N), ri)
    return psi, C, ri


def squeeze(psi, C, ri, output_bits: int = spec.OUTPUT_BITS) -> bytes:
    pa = np.array([a for a, _ in spec.squeeze_pairs()], dtype=np.int64)
    pb = np.array([b for _, b in spec.squeeze_pairs()], dtype=np.int64)
    bits = []
    while len(bits) < output_bits:
        cf = C.reshape(-1)
        cmp = (cf[pa] > cf[pb]).astype(np.uint8)
        bits.extend(int(x) for x in cmp)
        if len(bits) < output_bits:
            psi, C, ri = _coupled_evolve_T(psi, C, ri)
    bits = bits[:output_bits]
    out = bytearray((output_bits + 7) // 8)
    for i, bit in enumerate(bits):
        if bit:
            out[i >> 3] |= 1 << (7 - (i & 7))
    return bytes(out)


def cc_hash(message: bytes) -> bytes:
    psi, C, ri = absorb(message)
    return squeeze(psi, C, ri, spec.OUTPUT_BITS)


def trajectory_digest(psi0: np.ndarray, C0: Optional[np.ndarray] = None,
                      rounds: Optional[int] = None,
                      output_bits: int = spec.OUTPUT_BITS) -> bytes:
    psi = (np.asarray(psi0, dtype=np.int64) % _P).reshape(_N, _N).copy()
    C = iv_C() if C0 is None else (np.asarray(C0, dtype=np.int64) % _P).reshape(_N, _N).copy()
    r = spec.T if rounds is None else rounds
    psi, C, ri = _coupled_evolve_T(psi, C, 0, r)
    return squeeze(psi, C, ri, output_bits)
