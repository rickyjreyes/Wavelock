"""Tests for the Part-I formal corrections (terminology / counter / fixed output).

These lock in:
  * the fixed 256-bit public digest (no XOF surface),
  * the injective two-digit base-P block-counter encoding and its boundaries,
  * the MAX_INPUT_BITS rejection,
  * the corrected public API names (evolve_T, no permute).
"""

import importlib

import pytest

from wavelock.pde_hash import reference, optimized, spec

# The package re-exports `evolve` and `absorb` as *functions*, which shadow the
# submodules of the same name; import the modules explicitly.
evolve_mod = importlib.import_module("wavelock.pde_hash.evolve")
absorb_mod = importlib.import_module("wavelock.pde_hash.absorb")


# ---------------------------------------------------------------------------
# Fixed-output (not an XOF)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("impl", [reference, optimized])
@pytest.mark.parametrize("msg", [b"", b"abc", b"\x00" * 200, bytes(range(256))])
def test_pde_hash_is_fixed_32_bytes(impl, msg):
    out = impl.pde_hash(msg)
    assert isinstance(out, bytes)
    assert len(out) == 32


def test_pde_hash_has_no_output_bits_kwarg():
    import inspect

    for impl in (reference, optimized):
        sig = inspect.signature(impl.pde_hash)
        assert "output_bits" not in sig.parameters, (
            f"{impl.__name__}.pde_hash must not expose output_bits"
        )


def test_no_pde_xof_defined():
    for impl in (reference, optimized):
        assert not hasattr(impl, "pde_xof")


def test_public_api_uses_evolve_T_not_permute():
    import wavelock.pde_hash as pkg

    assert hasattr(pkg, "evolve_T")
    assert not hasattr(pkg, "permute")
    assert hasattr(evolve_mod, "evolve_T")
    assert not hasattr(evolve_mod, "permute")


# ---------------------------------------------------------------------------
# Two-digit base-P block counter
# ---------------------------------------------------------------------------
def test_counter_ordinary_messages_have_zero_high_digit():
    # q < P  =>  q1 == 0, preserving the original single-digit behaviour.
    for k in [0, 1, 2, 1000, spec.P - 2]:
        q0, q1 = spec.encode_block_counter(k)
        assert q1 == 0
        assert q0 == (k + 1) % spec.P


@pytest.mark.parametrize(
    "k",
    [
        spec.P - 2,
        spec.P - 1,
        spec.P,
        spec.P + 1,
        # the maximum block count allowed by the 64-bit length field:
        # at most ceil((2**64-1)/8 / 192) + 1 padded blocks.
        ((spec.MAX_INPUT_BITS // 8) // spec.BYTES_PER_BLOCK) + 2,
    ],
)
def test_counter_boundaries_are_in_high_digit_range(k):
    q0, q1 = spec.encode_block_counter(k)
    assert 0 <= q0 < spec.P
    assert 0 <= q1 < spec.P  # high digit must stay in-range over the whole span


def test_counter_pair_is_injective_across_boundary():
    # Exercise indices straddling the base-P rollover; encoded pairs must be
    # distinct (no aliasing) for distinct k.
    ks = list(range(spec.P - 5, spec.P + 5)) + [0, 1, 2, 3]
    seen = {}
    for k in ks:
        pair = spec.encode_block_counter(k)
        assert pair not in seen, f"alias: k={k} and k={seen[pair]} share {pair}"
        seen[pair] = k


def test_counter_high_digit_covers_max_block_index():
    # Max possible block index under the 64-bit input bound stays below P**2,
    # so two base-P digits suffice (q1 < P).
    max_blocks = (spec.MAX_INPUT_BITS // 8) // spec.BYTES_PER_BLOCK + 2
    q0, q1 = spec.encode_block_counter(max_blocks)
    assert q1 < spec.P
    assert max_blocks + 1 < spec.P * spec.P


def test_counter_rejects_negative():
    with pytest.raises(ValueError):
        spec.encode_block_counter(-1)


# ---------------------------------------------------------------------------
# MAX_INPUT_BITS rejection (synthetic; no giant allocation)
# ---------------------------------------------------------------------------
def test_pad_rejects_oversize_via_monkeypatched_bound(monkeypatch):
    # Lower the bound so a tiny message trips the same guard path.
    monkeypatch.setattr(spec, "MAX_INPUT_BITS", 8)
    with pytest.raises(ValueError):
        absorb_mod.pad(b"\x00\x00")  # 16 bits > 8
    # boundary: exactly at the bound is allowed
    absorb_mod.pad(b"\x00")  # 8 bits == 8 ok


def test_optimized_rejects_oversize(monkeypatch):
    monkeypatch.setattr(spec, "MAX_INPUT_BITS", 8)
    with pytest.raises(ValueError):
        optimized.pde_hash(b"\x00\x00")
