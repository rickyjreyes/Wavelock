"""Part III: weak non-cryptographic baselines + external reference points.

Compares the WaveLock-PDE candidate against deliberately weak controls (direct
truncation, XOR folding, a linear GF(2) cellular automaton, simple modular
linear mixing, and a reduced-round T=1 PDE) on avalanche, output bit bias, and
throughput. Conventional hashes (SHA-256, BLAKE2b) are included ONLY as external
statistical / performance reference columns.

IMPORTANT: the conventional hashes here are imported in audit TOOLING only. They
never touch the candidate's state or output (the candidate is computed entirely
by PDEVariant). Similarity to SHA-256 on avalanche/randomness is NOT evidence of
equivalent security and is not claimed as such.

Run:
    python -m pde_audit.baselines
"""

from __future__ import annotations

import hashlib          # EXTERNAL REFERENCE ONLY -- not part of the candidate
import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams


def _b256(x: bytes) -> bytes:
    return (x + b"\x00" * 32)[:32]


def trunc_hash(m: bytes) -> bytes:
    return _b256(m)


def xor_fold(m: bytes) -> bytes:
    acc = bytearray(32)
    for i, byte in enumerate(m):
        acc[i % 32] ^= byte
    return bytes(acc)


def linear_ca(m: bytes, steps: int = 64) -> bytes:
    bits = np.unpackbits(np.frombuffer(_seed_bits(m), dtype=np.uint8)).astype(np.uint8)
    for _ in range(steps):
        bits = (np.roll(bits, 1) ^ np.roll(bits, -1))   # rule-90-like, linear over GF(2)
    return np.packbits(bits).tobytes()


def mod_linear(m: bytes) -> bytes:
    words = np.zeros(4, dtype=np.uint64)
    arr = np.frombuffer(_b256_pad(m), dtype=np.uint8)
    for i, byte in enumerate(arr):
        words[i % 4] = (words[i % 4] * np.uint64(131) + np.uint64(int(byte))) % np.uint64(4294967291)
    return words.astype("<u8").tobytes()


def _b256_pad(m):
    if len(m) % 32:
        m = m + b"\x00" * (32 - len(m) % 32)
    return m if m else b"\x00" * 32


def _seed_bits(m):
    return xor_fold(m)


def sha256_ref(m: bytes) -> bytes:
    return hashlib.sha256(m).digest()


def blake2b_ref(m: bytes) -> bytes:
    return hashlib.blake2b(m, digest_size=32).digest()


_V1 = PDEVariant(PDEParams(T=1))
_V32 = PDEVariant(PDEParams(T=32))


def pde_T1(m: bytes) -> bytes:
    return _V1.hash(m)


def pde_T32(m: bytes) -> bytes:
    return _V32.hash(m)


FUNCS = {
    "truncation": trunc_hash,
    "xor_fold": xor_fold,
    "linear_ca": linear_ca,
    "mod_linear": mod_linear,
    "pde_T1_reduced": pde_T1,
    "pde_T32_candidate": pde_T32,
    "sha256_external_ref": sha256_ref,
    "blake2b_external_ref": blake2b_ref,
}
EXTERNAL = {"sha256_external_ref", "blake2b_external_ref"}


def _avalanche(fn, seed, n_msgs=120, flips_per=8):
    g = H.rng(seed)
    hds = []
    for _ in range(n_msgs):
        L = int(g.integers(8, 80))
        m = g.integers(0, 256, size=L, dtype=np.uint8).tobytes()
        base = H.bits_of(fn(m))
        for _ in range(flips_per):
            bit = int(g.integers(0, L * 8))
            ba = bytearray(m)
            ba[bit >> 3] ^= 1 << (7 - (bit & 7))
            hds.append(int((base ^ H.bits_of(fn(bytes(ba)))).sum()))
    hds = np.array(hds)
    return float(hds.mean()), float(hds.std())


def _bit_bias(fn, seed, n=3000):
    g = H.rng(seed)
    acc = np.zeros(256)
    for _ in range(n):
        L = int(g.integers(1, 200))
        m = g.integers(0, 256, size=L, dtype=np.uint8).tobytes()
        acc += H.bits_of(fn(m))
    p = acc / n
    return float(np.max(np.abs(p - 0.5)))


def _throughput(fn, seed, n=400):
    g = H.rng(seed)
    msgs = [g.integers(0, 256, size=256, dtype=np.uint8).tobytes() for _ in range(n)]
    t0 = time.perf_counter()
    for m in msgs:
        fn(m)
    dt = time.perf_counter() - t0
    return n / dt


def main(seed: int = 80090) -> dict:
    t0 = time.perf_counter()
    rows = {}
    for name, fn in FUNCS.items():
        amean, astd = _avalanche(fn, seed)
        bias = _bit_bias(fn, seed + 1)
        thr = _throughput(fn, seed + 2)
        rows[name] = {
            "external_reference": name in EXTERNAL,
            "avalanche_mean_hd": round(amean, 2),
            "avalanche_std_hd": round(astd, 2),
            "max_bit_bias_from_half": round(bias, 4),
            "hashes_per_sec": round(thr, 1),
        }
        tag = " [EXT REF]" if name in EXTERNAL else ""
        print(f"  {name:24s} avalancheHD={amean:6.1f}+/-{astd:4.1f}  "
              f"bias={bias:.4f}  {thr:8.0f} h/s{tag}")

    results = {
        "phase": "PartIII_baselines",
        "metadata": H.env_metadata(),
        "seed": seed,
        "ideal_avalanche_hd": 128.0,
        "rows": rows,
        "discipline_note": "External SHA-256/BLAKE2b columns are reference only. "
                           "Matching their avalanche/bias is NOT evidence of "
                           "equivalent cryptographic security.",
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    path = H.save_artifact("partIII_baselines.json", results)
    print("  saved", path, f"({results['runtime_s']}s)")
    return results


if __name__ == "__main__":
    main()
