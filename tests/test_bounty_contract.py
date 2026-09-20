"""Executable High 1–5 contract and bounded replay/canonical binding checks."""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from wavelock.chain import consensus_commitment as cc
from wavelock.chain.Wavelock_numpy import CurvatureKeyPairV3


SEED = bytes(range(32))
GOLDEN = {
    16: "6983241d9dee059dd82de32b996cf68fafa451a2bbe3775a675b3e791ce992eb",
    24: "0bf16e7bb05d2f3c3d572c60922465302cb2d8c30eecade6c8f2bc61e56b39d3",
    32: "f0d67235e3d0e712c31598f22a51fd303ac0334ecfa9fc0a7aec16e59fed3e46",
}
BODY_GOLDEN = {
    16: "352457d789c2f830d76899eb4405bf008bf0ce5f1a7576b8b19e00cd56ccea8c",
    24: "0a4bb86c48e7068eb83703f07488adfea34490456041a0a4ffa0f60de54ce68b",
    32: "53d7939101db324fa86ed7d0c145d25269de005224a6615e453867cd043a61ff",
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


def test_normative_descriptor_matches_hashed_header():
    path = Path(__file__).resolve().parents[1] / "audit/artifacts/bounty_profile_v1.json"
    descriptor = json.loads(path.read_text(encoding="utf-8"))
    header = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    raw = cc.commit_consensus_state(SEED).canonical_bytes
    assert raw[6:10] == struct.pack(">I", len(header))
    assert raw[10:10+len(header)] == header
    assert len(raw) == 1409


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
def test_high4_accepted_inputs_and_pinned_reference_parity(length, monkeypatch):
    seed = bytes(range(length))
    result = cc.commit_consensus_state(seed)
    assert result.commitment == cc.PROFILE + ":" + GOLDEN[length]
    assert cc.verify_consensus_commitment(result, seed)
    assert cc.verify_consensus_commitment(result.to_dict(), seed)
    offset = 10 + struct.unpack(">I", result.canonical_bytes[6:10])[0]
    # Preserve the original high-accuracy reference state/invariant body. The
    # digest changes only because the descriptor now binds fixed math rules.
    assert hashlib.sha256(result.canonical_bytes[offset:]).hexdigest() == BODY_GOLDEN[length]
    # Independent historical NumPy evolution with the declared exp/log rules.
    # Unmodified legacy libm is deliberately not claimed to be byte-identical.
    monkeypatch.setattr(np, "exp", lambda a: cc._transcendental(a))
    monkeypatch.setattr(np, "log", lambda a: cc._transcendental(a, logarithm=True))
    old = CurvatureKeyPairV3(n=4, seed=seed)
    assert cc.canonical_serialize(old.psi_star) == result.canonical_bytes


def test_reference_error_policy_does_not_inherit_ambient_numpy_settings():
    from decimal import localcontext, ROUND_DOWN, Inexact
    with localcontext() as context, np.errstate(all="raise"):
        context.prec = 3
        context.rounding = ROUND_DOWN
        context.traps[Inexact] = True
        artifact = cc.commit_consensus_state(SEED)
        assert artifact.commitment == cc.PROFILE + ":" + GOLDEN[32]
        assert cc.verify_consensus_commitment(artifact, SEED)


@pytest.mark.parametrize("disabled", ["", "AVX512F", "AVX512F,AVX2,FMA3,AVX"])
def test_reference_vectors_independent_of_simd_dispatch(disabled):
    code = """
import json
from wavelock.chain.consensus_commitment import commit_consensus_state
print(json.dumps([commit_consensus_state(bytes(range(n))).commitment for n in (16,24,32)]))
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, timeout=30,
                            env={**os.environ, "NPY_DISABLE_CPU_FEATURES": disabled})
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [cc.PROFILE + ":" + GOLDEN[n] for n in (16,24,32)]


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
    ("kernel", "update"), ("kernel", "invariants"), ("kernel", "transcendentals"),
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
