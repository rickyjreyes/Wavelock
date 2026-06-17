"""Shared audit harness for Phase 8.

Provides:
  * environment / provenance metadata capture;
  * deterministic message generators;
  * artifact JSON I/O under pde_audit/artifacts/;
  * a PARAMETERIZED audit-only reimplementation of the WaveLock-PDE construction
    (PDEVariant) that can vary T, D, a, b, G, N, p WITHOUT touching the
    normative primitive in wavelock/pde_hash/. With v0 parameters it reproduces
    the normative optimized.pde_hash byte-for-byte (asserted by self_test()).
  * modular linear algebra (rank over F_p) for Jacobian analysis.

Nothing here is a conventional cryptographic primitive. The harness operates on
raw PDE state and raw 256-bit output only; it never routes candidate output
through SHA/SHAKE/BLAKE.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np

from wavelock.pde_hash import spec as _spec
from wavelock.pde_hash import optimized as _opt

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def env_metadata() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
    }


def save_artifact(name: str, obj: dict) -> str:
    path = os.path.join(ARTIFACT_DIR, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
    return path


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")


def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def random_messages(seed: int, count: int, min_len: int = 0, max_len: int = 400):
    g = rng(seed)
    out = []
    for _ in range(count):
        n = int(g.integers(min_len, max_len + 1))
        out.append(g.integers(0, 256, size=n, dtype=np.uint8).tobytes())
    return out


# ---------------------------------------------------------------------------
# parameterized audit variant
# ---------------------------------------------------------------------------
@dataclass
class PDEParams:
    N: int = _spec.N
    p: int = _spec.P
    D: int = _spec.D
    a: int = _spec.A
    b: int = _spec.B
    G: int = _spec.G
    T: int = _spec.T

    def tag(self) -> str:
        return f"N{self.N}_p{self.p}_D{self.D}_a{self.a}_b{self.b}_G{self.G}_T{self.T}"


class PDEVariant:
    """Audit-only NumPy reimplementation with tunable parameters.

    The state is an (N, N) int64 array of residues in [0, p). The message layer
    (packing / padding / counter / squeeze) mirrors the spec; only the round
    constants and round count are tunable. For toy primes p the byte packing can
    exceed p, so message-level helpers are intended for the full field
    (p = 2**31 - 1); state-level helpers (one_round, evolve_T) work for any p.
    """

    def __init__(self, params: PDEParams | None = None):
        self.P = PDEParams() if params is None else params
        p = self.P.p
        self.pm4 = (p - 4) % p
        self._pairs_a = np.array([a for a, _ in _spec.squeeze_pairs()], dtype=np.int64)
        self._pairs_b = np.array([b for _, b in _spec.squeeze_pairs()], dtype=np.int64)

    # --- state-level -----------------------------------------------------
    def iv(self) -> np.ndarray:
        tag = np.frombuffer(_spec.IV_TAG, dtype=np.uint8).astype(np.int64)
        n_cells = self.P.N * self.P.N
        idx = np.arange(n_cells, dtype=np.int64)
        flat = (1 + idx + tag[idx % tag.size]) % self.P.p
        return flat.reshape(self.P.N, self.P.N).copy()

    def one_round(self, psi: np.ndarray) -> np.ndarray:
        p = self.P.p
        sq = (psi * psi) % p
        bm = (self.P.b + (p - sq)) % p
        react = (self.P.a * ((psi * bm) % p)) % p
        lap = (
            np.roll(psi, -1, 0) + np.roll(psi, 1, 0)
            + np.roll(psi, -1, 1) + np.roll(psi, 1, 1)
            + (self.pm4 * psi) % p
        ) % p
        return (psi + (self.P.D * lap) % p + react) % p

    def evolve_T(self, psi: np.ndarray, rounds: int | None = None) -> np.ndarray:
        r = self.P.T if rounds is None else rounds
        for _ in range(r):
            psi = self.one_round(psi)
        return psi

    # --- message-level (full field only) --------------------------------
    def _pad(self, message: bytes) -> bytes:
        m = bytes(message)
        out = bytearray(m)
        out.append(1)
        block = _spec.BYTES_PER_BLOCK
        out.extend(b"\x00" * ((-len(out)) % block))
        lb = bytearray(block)
        lb[0:8] = (len(m) * 8).to_bytes(8, "big")
        out.extend(lb)
        return bytes(out)

    def absorb(self, message: bytes) -> np.ndarray:
        p = self.P.p
        padded = self._pad(message)
        arr = np.frombuffer(padded, dtype=np.uint8).astype(np.int64)
        nb = arr.size // _spec.BYTES_PER_BLOCK
        arr = arr.reshape(nb, _spec.RATE, 3)
        elems = arr[:, :, 0] + (arr[:, :, 1] << 8) + (arr[:, :, 2] << 16)
        psi = self.iv()
        for k in range(nb):
            flat = psi.reshape(-1)
            flat[: _spec.RATE] = (flat[: _spec.RATE] + elems[k]) % p
            q = k + 1
            flat[_spec.CAP0] = (flat[_spec.CAP0] + (q % p) * self.P.G) % p
            flat[_spec.CAP2] = (flat[_spec.CAP2] + (q // p) * self.P.G) % p
            if k == nb - 1:
                flat[_spec.CAP1] = (flat[_spec.CAP1] + _spec.D_TAG) % p
            psi = self.evolve_T(flat.reshape(self.P.N, self.P.N))
        return psi

    def squeeze(self, psi: np.ndarray, output_bits: int = 256) -> bytes:
        flat = psi.reshape(-1)
        bits: List[int] = []
        while len(bits) < output_bits:
            cmp = (flat[self._pairs_a] > flat[self._pairs_b]).astype(np.uint8)
            bits.extend(int(x) for x in cmp)
            if len(bits) < output_bits:
                psi = self.evolve_T(psi)
                flat = psi.reshape(-1)
        bits = bits[:output_bits]
        out = bytearray((output_bits + 7) // 8)
        for i, bit in enumerate(bits):
            if bit:
                out[i >> 3] |= 1 << (7 - (i & 7))
        return bytes(out)

    def squeeze_bits(self, psi: np.ndarray, output_bits: int = 256) -> np.ndarray:
        """Return the raw 0/1 bit array (no byte packing) for analysis."""
        flat = psi.reshape(-1)
        bits: List[int] = []
        while len(bits) < output_bits:
            cmp = (flat[self._pairs_a] > flat[self._pairs_b]).astype(np.uint8)
            bits.extend(int(x) for x in cmp)
            if len(bits) < output_bits:
                psi = self.evolve_T(psi)
                flat = psi.reshape(-1)
        return np.array(bits[:output_bits], dtype=np.uint8)

    def hash(self, message: bytes, output_bits: int = 256) -> bytes:
        return self.squeeze(self.absorb(message), output_bits)


def self_test() -> bool:
    """v0-parameter variant must match the normative optimized primitive."""
    v = PDEVariant()
    for m in [b"", b"abc", b"WaveLock", b"\x00" * 193, bytes(range(256))]:
        assert v.hash(m) == _opt.pde_hash(m), f"variant mismatch on {m[:8]!r}"
    return True


# ---------------------------------------------------------------------------
# modular linear algebra
# ---------------------------------------------------------------------------
def mod_rank(matrix: np.ndarray, p: int) -> int:
    """Rank of an integer matrix over F_p via modular Gaussian elimination."""
    M = (np.asarray(matrix, dtype=object) % p)
    M = [list(row) for row in M]
    rows = len(M)
    cols = len(M[0]) if rows else 0
    rank = 0
    pivot_row = 0
    for col in range(cols):
        # find a pivot in this column at or below pivot_row
        piv = None
        for r in range(pivot_row, rows):
            if M[r][col] % p != 0:
                piv = r
                break
        if piv is None:
            continue
        M[pivot_row], M[piv] = M[piv], M[pivot_row]
        inv = pow(int(M[pivot_row][col]), p - 2, p)  # Fermat inverse (p prime)
        M[pivot_row] = [(x * inv) % p for x in M[pivot_row]]
        for r in range(rows):
            if r != pivot_row and M[r][col] % p != 0:
                factor = M[r][col] % p
                M[r] = [(M[r][c] - factor * M[pivot_row][c]) % p for c in range(cols)]
        pivot_row += 1
        rank += 1
        if pivot_row == rows:
            break
    return rank


def hamming_bytes(a: bytes, b: bytes) -> int:
    return sum(bin(x ^ y).count("1") for x, y in zip(a, b))


def bits_of(data: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


if __name__ == "__main__":
    t0 = time.perf_counter()
    assert self_test()
    print("harness self-test OK", f"{time.perf_counter()-t0:.2f}s")
    print(json.dumps(env_metadata(), indent=2))
