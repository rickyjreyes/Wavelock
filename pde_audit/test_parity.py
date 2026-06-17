"""Cross-implementation parity: reference (pure-Python) vs optimized (NumPy).

Byte-for-byte agreement is required over:
  * the empty message,
  * all 256 one-byte messages,
  * lengths around every padding boundary,
  * structured inputs,
  * >= 10,000 deterministic pseudo-random messages,
and round-granular intermediate snapshots (to catch shared-bug masking that a
final-output-only comparison would miss).

The message generator uses the stdlib ``random`` module (Mersenne Twister) with
a fixed seed. That is test tooling only; it never touches the primitive, which
contains no PRNG or hash.
"""

import random

import pytest

from wavelock.pde_hash import reference, optimized
from wavelock.pde_hash import spec


def _agree(msg):
    r = reference.pde_hash(msg)
    o = optimized.pde_hash(msg)
    assert len(r) == 32
    assert r == o, f"parity break on {msg[:16]!r} (len {len(msg)}): {r.hex()} != {o.hex()}"
    return r


def test_empty_message():
    _agree(b"")


def test_all_one_byte_messages():
    for b in range(256):
        _agree(bytes([b]))


@pytest.mark.parametrize(
    "length",
    # around the 192-byte block boundary and the second boundary
    sorted(set(
        list(range(0, 8))
        + [spec.BYTES_PER_BLOCK - 2, spec.BYTES_PER_BLOCK - 1,
           spec.BYTES_PER_BLOCK, spec.BYTES_PER_BLOCK + 1, spec.BYTES_PER_BLOCK + 2,
           2 * spec.BYTES_PER_BLOCK - 1, 2 * spec.BYTES_PER_BLOCK,
           2 * spec.BYTES_PER_BLOCK + 1, 3 * spec.BYTES_PER_BLOCK]
    )),
)
def test_padding_boundaries(length):
    # both all-zero and a fixed nonzero fill, since padding interacts with zeros
    _agree(b"\x00" * length)
    _agree(b"\xa5" * length)


def test_structured_inputs():
    cases = [
        b"abc",
        b"WaveLock",
        b"The quick brown fox jumps over the lazy dog",
        bytes(range(256)),
        b"\x00\x00\x00\x01",
        b"\x01\x00\x00\x00",
        b"\xff" * 100,
        b"a" * 191,
        b"a" * 192,
        b"a" * 193,
        ("é你好" * 10).encode("utf-8"),
    ]
    for m in cases:
        _agree(m)


def test_repeated_block_order_matters():
    # Same block repeated vs single, and order swaps, must give distinct digests
    # (and both impls must agree on each).
    blk0 = b"\x11" * spec.BYTES_PER_BLOCK
    blk1 = b"\x22" * spec.BYTES_PER_BLOCK
    d01 = _agree(blk0 + blk1)
    d10 = _agree(blk1 + blk0)
    d00 = _agree(blk0 + blk0)
    assert d01 != d10, "block order does not affect digest"
    assert d00 != d01


@pytest.mark.slow
def test_10k_random_messages():
    rng = random.Random(20260617)  # fixed seed for deterministic messages
    n = 10_000
    for _ in range(n):
        length = rng.randint(0, 400)
        msg = bytes(rng.getrandbits(8) for _ in range(length))
        _agree(msg)


def test_intermediate_round_parity():
    rng = random.Random(424242)
    for _ in range(50):
        length = rng.randint(0, 500)
        msg = bytes(rng.getrandbits(8) for _ in range(length))
        tr = reference.trace(msg)
        to = optimized.trace(msg)
        assert tr.keys() == to.keys()
        for k in tr:
            assert tr[k] == to[k], f"intermediate {k} differs on len {length}"
