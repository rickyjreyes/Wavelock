"""Executable High 1–5 contract and bounded replay/canonical binding checks."""
import hashlib
import json
import struct

import numpy as np
import pytest

from wavelock.chain import consensus_commitment as cc
from wavelock.chain.Wavelock_numpy import CurvatureKeyPairV3


SEED = bytes(range(32))
GOLDEN = {
    16: "9e2a16c787e5ce4ca4bde68edb8df9cbde716cc25638b1c3abf4bb3b8b826b74",
    24: "7d5924a4859661ff3a38edf0e8a7603a0587d8619084103b4ef8b3cb48b0bcc3",
    32: "aff7f2d51d2a1a45d3558eefb9a803a2f406a7ddf531271863f55e7e76a80943",
}


def reverse_dicts(value):
    if isinstance(value, dict):
        return {k: reverse_dicts(v) for k, v in reversed(list(value.items()))}
    return value


def test_high1_canonical_layout_and_dictionary_order():
    state = np.arange(16, dtype=np.float64).reshape(4, 4) / 16 + 0.1
    alternate = np.empty((8, 8), dtype=np.float64)[::2, ::2]
    alternate[:] = state
    expected = cc.canonical_serialize(state)
    for array in (state.copy(order="F"), alternate, state.astype(">f8"), state.astype("<f8")):
        assert cc.canonical_serialize(array, metadata=reverse_dicts(cc.profile_metadata())) == expected
    assert cc.validate_canonical_bytes(expected)
    body_offset = len(cc.MAGIC) + 4 + struct.unpack(">I", expected[6:10])[0]
    assert expected[body_offset:body_offset+128] == struct.pack(">16d", *state.ravel(order="C"))


@pytest.mark.parametrize("dtype", ["f4", "i8", "c16", "object"])
def test_high1_wrong_dtype_rejected(dtype):
    with pytest.raises(ValueError):
        cc.canonical_serialize(np.zeros((4, 4), dtype=dtype))


@pytest.mark.parametrize("change", ["json-whitespace", "endian", "trailing", "schema", "negative-zero"])
def test_high1_alternate_wire_encoding_rejected(change):
    state = np.arange(16, dtype=np.float64).reshape(4, 4)
    raw = cc.canonical_serialize(state)
    size = struct.unpack(">I", raw[6:10])[0]
    start = 10 + size
    if change == "json-whitespace":
        header = json.dumps(json.loads(raw[10:start]), indent=1).encode()
        raw = cc.MAGIC + struct.pack(">I", len(header)) + header + raw[start:]
    elif change == "endian":
        raw = raw[:start] + state.astype("<f8").tobytes() + raw[start+128:]
    elif change == "trailing":
        raw += b"\0"
    elif change == "schema":
        raw = b"WLCC\x00\x02" + raw[6:]
    else:
        raw = raw[:start] + struct.pack(">d", -0.0) + raw[start+8:]
    with pytest.raises((ValueError, FloatingPointError)):
        cc.validate_canonical_bytes(raw)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("position", range(16))
def test_high2_every_state_position_rejected(value, position):
    state = np.ones((4, 4), dtype=np.float64)
    state.flat[position] = value
    with pytest.raises(ValueError, match="non-finite"):
        cc.canonical_serialize(state)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", ["alpha", "beta", "theta", "epsilon", "delta", "dt", "damping"])
def test_high2_every_parameter_rejected(value, field):
    meta = cc.profile_metadata()
    meta["kernel"]["parameters"][field] = value
    with pytest.raises(ValueError, match="non-finite"):
        cc.canonical_serialize(np.zeros((4, 4)), metadata=meta)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("field", cc.INVARIANTS)
def test_high2_every_invariant_rejected(value, field):
    invariants = dict.fromkeys(cc.INVARIANTS, 0.0)
    invariants[field] = value
    with pytest.raises(ValueError, match="non-finite"):
        cc.canonical_serialize(np.zeros((4, 4)), invariants=invariants)


def test_high2_derived_overflow_and_producer_injection_rejected(monkeypatch):
    with pytest.raises((ValueError, FloatingPointError)):
        cc.canonical_serialize(np.full((4, 4), 1e308))
    monkeypatch.setattr(cc, "evolve_reference", lambda seed: np.full((4, 4), np.nan))
    with pytest.raises(ValueError):
        cc.commit_consensus_state(SEED)
    assert not cc.verify_consensus_commitment({}, SEED)


def test_high3_signed_zero_in_state_and_invariants():
    positive = np.zeros((4, 4))
    mixed = positive.copy()
    mixed[0, 0] = -0.0
    isolated = cc.canonical_serialize(mixed)
    mixed.flat[::2] = -0.0
    negative_invariants = dict.fromkeys(cc.INVARIANTS, -0.0)
    raw = cc.canonical_serialize(positive)
    assert isolated == raw == cc.canonical_serialize(mixed, invariants=negative_invariants)
    assert hashlib.sha256(raw).digest() == hashlib.sha256(cc.canonical_serialize(mixed)).digest()
    mixed[0, 0] = -1.0
    assert cc.canonical_serialize(mixed) != raw  # ordinary negative values survive


@pytest.mark.parametrize("seed", [42, 0, 2**64-1, b"", bytes(8), bytes(15),
                                 "a long string is still not a production secret", None, bytearray(32)])
def test_high4_production_seed_threshold(seed):
    with pytest.raises(ValueError):
        cc.commit_consensus_state(seed)


@pytest.mark.parametrize("length", [16, 24, 32])
def test_high4_accepted_inputs_and_pinned_reference_parity(length):
    seed = bytes(range(length))
    result = cc.commit_consensus_state(seed)
    assert result.commitment == cc.PROFILE + ":" + GOLDEN[length]
    assert cc.verify_consensus_commitment(result, seed)
    assert cc.verify_consensus_commitment(result.to_dict(), seed)
    old = CurvatureKeyPairV3(n=4, seed=seed)
    assert cc.canonical_serialize(old.psi_star) == result.canonical_bytes


@pytest.mark.parametrize("options", [{"backend": "cupy"}, {"backend": "numpy-fast"},
    {"backend": "fft"}, {"backend": "numpy"}, {"fast_math": True}, {"reassociation": True},
    {"profile": "WLv2"}, {"profile": "WLv3.1"}, {"profile": "WL-Consensus-Commitment-v2"}])
def test_high5_unsupported_backend_or_profile(options):
    with pytest.raises(ValueError):
        cc.commit_consensus_state(SEED, **options)


def test_historical_modes_cannot_masquerade_as_consensus():
    from wavelock.chain.WaveLock import CurvatureKeyPair
    from wavelock.chain.Wavelock_numpy import _serialize_commitment
    for key in (CurvatureKeyPair(4, seed=42, test_mode=True), CurvatureKeyPairV3(4, seed=42)):
        assert key.consensus_valid is False
        assert key.operating_mode == "historical/research"
        assert not cc.verify_consensus_commitment(key.commitment, SEED)
    with pytest.raises(ValueError):
        _serialize_commitment(np.zeros((4, 4)), schema=cc.PROFILE)


BOUND_FIELDS = [
    ("profile",), ("schema",), ("backend",), ("dtype",), ("byte_order",),
    ("array_order",), ("shape",), ("kernel_hash",), ("kernel", "version"),
    ("kernel", "steps"), ("kernel", "boundary"), ("kernel", "laplacian"),
    ("kernel", "update"), ("kernel", "invariants"),
    ("initialization", "xof"), ("initialization", "version"),
    ("initialization", "domain"), ("initialization", "length_encoding"),
    ("initialization", "mapping"), ("normalization",), ("hash",),
    ("fast_math",), ("reassociation",),
] + [("kernel", "parameters", name) for name in cc.profile_metadata()["kernel"]["parameters"]]


@pytest.mark.parametrize("path", BOUND_FIELDS)
def test_every_metadata_field_is_bound(path):
    artifact = cc.commit_consensus_state(SEED).to_dict()
    target = artifact["metadata"]
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = "substitution"
    assert not cc.verify_consensus_commitment(artifact, SEED)


def test_replay_rejects_wrong_input_state_invariants_and_extra_fields():
    artifact = cc.commit_consensus_state(SEED)
    state = cc.evolve_reference(SEED)
    assert not cc.verify_consensus_commitment(artifact, bytes(range(1, 33)))
    state[0, 0] += 1
    assert not cc.verify_consensus_commitment(artifact, SEED, state=state)
    values = cc.reference_invariants(cc.evolve_reference(SEED))
    values["E_tot"] += 1
    assert not cc.verify_consensus_commitment(artifact, SEED, invariants=values)
    public = artifact.to_dict()
    public["unbound"] = "field"
    assert not cc.verify_consensus_commitment(public, SEED)


def test_commitment_artifact_publishing_is_public_and_exclusive(tmp_path):
    artifact = cc.commit_consensus_state(SEED)
    path = tmp_path / "commitment.json"
    cc.write_commitment_artifact(path, artifact)
    public = json.loads(path.read_text())
    assert set(public) == {"profile", "metadata", "commitment"}
    assert cc.verify_consensus_commitment(public, SEED)
    with pytest.raises(FileExistsError):
        cc.write_commitment_artifact(path, artifact)
