"""
XOF-based initial wavefield derivation (Claim 9 best mode).

Provides deterministic ψ₀ derivation from a seed via an extendable-output
function (XOF). The default XOF is SHAKE-256, the NIST-standardized variant
of Keccak. The interface is generic so other XOFs can be substituted without
changing the call site.

Why this exists
---------------
Per-backend RNGs (numpy.random.seed, cupy.random.seed) are NOT byte-stable
across versions or backends. A consensus commitment binds ψ★ to a kernel
hash and operator parameters; the ψ₀ that produced ψ★ must be reproducible
from the seed alone. SHAKE-256 gives that reproducibility:

    seed (bytes)  ──SHAKE-256──► uniform float64 in [0, 1) ──reshape──► ψ₀

Because SHAKE-256 is a fixed standard with byte-stable output, two
independent implementations of WaveLock (one in C, one in Python; one on
CPU, one on GPU) will derive the same ψ₀ from the same seed.

This module is opt-in. Existing users of CurvatureKeyPair with integer
seeds continue to use np.random / cp.random for backward compatibility.
New consensus-grade keypairs should pass `seed_bytes=...` and select the
XOF derivation explicitly.
"""

from __future__ import annotations

from typing import Tuple, Union
import hashlib
import struct
import numpy as np


# Domain separation tag — distinguishes WaveLock ψ₀ derivation from any
# other use of SHAKE-256 with the same seed.
_DOMAIN_TAG = b"WL-PSI-INIT-v1"


def _shake256_stream(seed_bytes: bytes, n_bytes: int) -> bytes:
    """SHAKE-256 XOF stream with WaveLock domain separation."""
    h = hashlib.shake_256()
    h.update(_DOMAIN_TAG)
    h.update(struct.pack(">Q", len(seed_bytes)))
    h.update(seed_bytes)
    return h.digest(n_bytes)


def _bytes_to_uniform_float64(stream: bytes, count: int) -> np.ndarray:
    """
    Convert raw bytes to count float64 values uniform in [0, 1).

    Uses the standard 53-bit mantissa construction: take 8 bytes per sample,
    interpret as a uint64, mask to 53 bits, divide by 2**53. This produces
    the same distribution as numpy's default uniform draw, but deterministic
    across implementations because we control the byte source.
    """
    if len(stream) < count * 8:
        raise ValueError(
            f"XOF stream too short: need {count * 8} bytes, got {len(stream)}"
        )
    raw = np.frombuffer(stream[: count * 8], dtype=">u8").astype(np.uint64)
    masked = raw & np.uint64((1 << 53) - 1)
    return masked.astype(np.float64) / float(1 << 53)


def derive_psi_zero(
    seed: Union[bytes, str, int],
    shape: Tuple[int, ...],
    dtype: np.dtype = np.float64,
    xof: str = "shake_256",
) -> np.ndarray:
    """
    Deterministically derive ψ₀ from a seed using a XOF.

    Parameters
    ----------
    seed : bytes | str | int
        Seed material. Strings are encoded UTF-8. Ints are encoded as
        big-endian 8-byte uint64. Bytes are used as-is.
    shape : tuple[int, ...]
        Shape of the resulting wavefield (typically (side, side)).
    dtype : np.dtype
        Output dtype. float64 is the consensus default.
    xof : str
        Which XOF to use. Currently only "shake_256" is supported. The
        parameter exists so the patent claim can reference "an extendable-
        output function" generically and we can swap algorithms later
        without API churn.

    Returns
    -------
    np.ndarray
        ψ₀ of the requested shape and dtype, with values uniform in [0, 1).

    Notes
    -----
    Byte-exact across CPU and GPU backends — the GPU keypair should derive
    ψ₀ on the host with this function and copy to device, never call
    cupy.random.* for consensus-grade commitments.
    """
    if isinstance(seed, str):
        seed_bytes = seed.encode("utf-8")
    elif isinstance(seed, int):
        seed_bytes = struct.pack(">Q", seed & 0xFFFFFFFFFFFFFFFF)
    elif isinstance(seed, (bytes, bytearray)):
        seed_bytes = bytes(seed)
    else:
        raise TypeError(f"seed must be bytes, str, or int (got {type(seed).__name__})")

    if xof != "shake_256":
        raise ValueError(
            f"Unsupported XOF '{xof}'. Currently supported: 'shake_256'."
        )

    count = 1
    for d in shape:
        count *= int(d)

    stream = _shake256_stream(seed_bytes, count * 8)
    flat = _bytes_to_uniform_float64(stream, count)
    return flat.reshape(shape).astype(dtype, copy=False)


__all__ = ["derive_psi_zero"]


if __name__ == "__main__":
    # Smoke test: same seed -> same ψ₀; different seeds -> different ψ₀.
    a = derive_psi_zero(42, (4, 4))
    b = derive_psi_zero(42, (4, 4))
    c = derive_psi_zero(43, (4, 4))
    assert np.array_equal(a, b), "determinism failed"
    assert not np.array_equal(a, c), "seed sensitivity failed"
    assert a.dtype == np.float64
    assert (a >= 0.0).all() and (a < 1.0).all()
    print("xof_init smoke test passed")
    print("sample[0,0] =", float(a[0, 0]))
