"""Normative WaveLock commitment profile; historical WLv* encodings are separate.

The producer validates secret input and executes the NumPy reference kernel.
Replay verification repeats that production path. Canonical-byte validation
alone establishes encoding and hash consistency, not provenance of a runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import hmac
import json
import math
import struct

import numpy as np

from .xof_init import derive_psi_zero

PROFILE = "WL-Consensus-Commitment-v1"
BACKEND = "numpy-reference-v1"
MAGIC = b"WLCC\x00\x01"
MIN_INPUT_BYTES = 16
DEFAULT_INPUT_BYTES = 32
INVARIANTS = ("E_grad", "E_fb", "E_ent", "E_tot")


def _float(value):
    if type(value) not in (float, int) or isinstance(value, bool):
        raise ValueError("consensus scalar must be a real number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("non-finite consensus scalar")
    return 0.0 if value == 0.0 else value


def _wire(value):
    """Metadata floats use the same big-endian IEEE-754 binary64 as the body."""
    if type(value) is float:
        return {"binary64": struct.pack(">d", _float(value)).hex()}
    if type(value) is dict:
        if not all(type(k) is str for k in value):
            raise ValueError("metadata keys must be strings")
        return {k: _wire(v) for k, v in value.items()}
    if type(value) is list:
        return [_wire(v) for v in value]
    if type(value) in (str, int, bool) or value is None:
        return value
    raise ValueError("unsupported metadata type")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def profile_metadata():
    """A fresh descriptor: no caller mutation changes the registered profile."""
    kernel = {
        "version": "WL-psi-001-reference-v1",
        "parameters": {"alpha": 1.5, "beta": 0.0026, "theta": 1e-5,
                       "epsilon": 1e-12, "delta": 1e-12, "dt": 0.1,
                       "damping": 2e-5},
        "steps": 50,
        "boundary": "periodic-roll",
        "laplacian": "((((-4*x+roll(x,+1,0))+roll(x,-1,0))+roll(x,+1,1))+roll(x,-1,1))",
        "update": "psi+(dt*(alpha*lap/(psi+epsilon*exp(-beta*psi**2))-theta*(psi*lap(log(psi**2+delta))))-damping*psi)",
        "invariants": "numpy-gradient-edge-order-1;C-order-sums;E_grad,E_fb,E_ent,E_tot",
    }
    return {
        "profile": PROFILE, "schema": 1, "backend": BACKEND,
        "dtype": "IEEE-754-binary64", "byte_order": "big", "array_order": "C",
        "shape": [4, 4], "kernel": kernel,
        "kernel_hash": hashlib.sha256(_json(_wire(kernel))).hexdigest(),
        "initialization": {"xof": "SHAKE-256", "version": 1,
                           "domain": "WL-PSI-INIT-v1", "length_encoding": "uint64-big",
                           "mapping": "uint64-big-low53-div-2**53"},
        "normalization": "finite-only;negative-zero-to-positive-zero",
        "hash": "SHA-256", "fast_math": False, "reassociation": False,
    }


def _metadata(metadata):
    expected = profile_metadata()
    candidate = expected if metadata is None else metadata
    # Encode before comparing: NaN, Inf, unsupported types and extra fields
    # never disappear through permissive equality/coercion.
    encoded = _json(_wire(candidate))
    if encoded != _json(_wire(expected)):
        raise ValueError("unknown or modified consensus profile/metadata")
    return encoded


def _state(state):
    # Do not implicitly transfer GPU arrays, invoke __array__, or round f32.
    if type(state) is not np.ndarray or state.dtype.kind != "f" or state.dtype.itemsize != 8:
        raise ValueError("state must be a NumPy binary64 array")
    if state.shape != (4, 4):
        raise ValueError("profile v1 requires a 4 by 4 state")
    if not np.isfinite(state).all():
        raise ValueError("non-finite consensus state")
    result = np.array(state, dtype=np.float64, order="C", copy=True)
    result[result == 0.0] = 0.0
    return result


def _lap(x):
    return (-4.0 * x + np.roll(x, 1, 0) + np.roll(x, -1, 0)
            + np.roll(x, 1, 1) + np.roll(x, -1, 1))


def evolve_reference(secret_input):
    """The existing NumPy reference equation, with every parameter explicit."""
    _validate_input(secret_input)
    p = profile_metadata()["kernel"]["parameters"]
    psi = derive_psi_zero(secret_input, (4, 4), dtype=np.float64, xof="shake_256")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for _ in range(50):
            lap = _lap(psi)
            fb = p["alpha"] * lap / (psi + p["epsilon"] * np.exp(-p["beta"] * psi ** 2))
            ent = p["theta"] * (psi * _lap(np.log(psi ** 2 + p["delta"])))
            dpsi = p["dt"] * (fb - ent) - p["damping"] * psi
            psi = psi + dpsi
    return _state(psi)


def reference_invariants(state):
    psi = _state(state)
    p = profile_metadata()["kernel"]["parameters"]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        gx, gy = np.gradient(psi)
        grad = float(np.sum(gx * gx) + np.sum(gy * gy))
        fb = p["alpha"] * _lap(psi) / (psi + p["epsilon"] * np.exp(-p["beta"] * psi ** 2))
        ent = p["theta"] * (psi * _lap(np.log(psi ** 2 + p["delta"])))
        feedback = float(np.sum(fb * fb))
        entropy = float(np.sum(ent * ent))
        total = float(grad + feedback + entropy)
    return dict(zip(INVARIANTS, map(_float, (grad, feedback, entropy, total))))


def canonical_serialize(state, *, metadata=None, invariants=None):
    """The sole normative preimage encoding. See BOUNTY_SECURITY_PROFILE.md."""
    header = _metadata(metadata)
    psi = _state(state)
    expected = reference_invariants(psi)
    values = expected if invariants is None else invariants
    if type(values) is not dict or set(values) != set(INVARIANTS):
        raise ValueError("invalid invariant fields")
    normalized = {k: _float(values[k]) for k in INVARIANTS}
    if normalized != expected:
        raise ValueError("invariants do not match the declared state")
    body = psi.astype(">f8").tobytes(order="C")
    energies = struct.pack(">4d", *(normalized[k] for k in INVARIANTS))
    return MAGIC + struct.pack(">I", len(header)) + header + body + energies


def _digest(preimage):
    return PROFILE + ":" + hashlib.sha256(preimage).hexdigest()


def validate_canonical_bytes(preimage):
    """Reject alternate JSON, endian, zero, trailing, schema or field encodings."""
    if type(preimage) is not bytes or not preimage.startswith(MAGIC):
        raise ValueError("unknown commitment serialization")
    offset = len(MAGIC) + 4
    if len(preimage) < offset:
        raise ValueError("truncated canonical commitment")
    length = struct.unpack(">I", preimage[len(MAGIC):offset])[0]
    header = _metadata(None)
    if length != len(header) or preimage[offset:offset+length] != header:
        raise ValueError("noncanonical profile metadata")
    offset += length
    if len(preimage) != offset + 16 * 8 + 4 * 8:
        raise ValueError("invalid canonical body length")
    state = np.frombuffer(preimage[offset:offset+128], dtype=">f8").reshape(4, 4)
    values = dict(zip(INVARIANTS, struct.unpack(">4d", preimage[offset+128:])))
    if canonical_serialize(state, invariants=values) != preimage:
        raise ValueError("noncanonical state or invariant encoding")
    return True


def _validate_input(secret_input):
    if type(secret_input) is not bytes:
        raise ValueError("production input must be bytes; integer/string/demo seeds are not supported")
    if len(secret_input) < MIN_INPUT_BYTES:
        raise ValueError("production input requires at least 16 bytes (128 bits)")


@dataclass(frozen=True)
class CommitmentArtifact:
    commitment: str
    canonical_bytes: bytes = field(repr=False)

    def to_dict(self):
        # The portable public artifact contains no seed or wave array.
        return {"profile": PROFILE, "metadata": _wire(profile_metadata()),
                "commitment": self.commitment}


def commit_consensus_state(secret_input: bytes, *, profile=PROFILE, backend=BACKEND,
                           fast_math=False, reassociation=False):
    _validate_input(secret_input)
    if profile != PROFILE or backend != BACKEND:
        raise ValueError("unsupported consensus profile or backend")
    if fast_math is not False or reassociation is not False:
        raise ValueError("fast-math/reassociation are not supported")
    raw = canonical_serialize(evolve_reference(secret_input))
    return CommitmentArtifact(_digest(raw), raw)


def verify_consensus_commitment(artifact, secret_input, *, state=None, invariants=None):
    """Replay the declared computation and bind all presented state/metadata."""
    try:
        if isinstance(artifact, CommitmentArtifact):
            validate_canonical_bytes(artifact.canonical_bytes)
            if _digest(artifact.canonical_bytes) != artifact.commitment:
                return False
            artifact = artifact.to_dict()
        if type(artifact) is not dict or set(artifact) != {"profile", "metadata", "commitment"}:
            return False
        if artifact["profile"] != PROFILE or _json(artifact["metadata"]) != _json(_wire(profile_metadata())):
            return False
        expected = commit_consensus_state(secret_input)
        if not isinstance(artifact["commitment"], str) or not hmac.compare_digest(artifact["commitment"], expected.commitment):
            return False
        if state is not None or invariants is not None:
            supplied = evolve_reference(secret_input) if state is None else state
            if canonical_serialize(supplied, invariants=invariants) != expected.canonical_bytes:
                return False
        return True
    except (ValueError, TypeError, OverflowError, FloatingPointError):
        return False


def write_commitment_artifact(path, artifact):
    """Publish an immutable public artifact after canonical consistency checks."""
    from wavelock.crypto.keyfiles import write_json
    if not isinstance(artifact, CommitmentArtifact):
        raise ValueError("expected a CommitmentArtifact from commit_consensus_state")
    validate_canonical_bytes(artifact.canonical_bytes)
    if _digest(artifact.canonical_bytes) != artifact.commitment:
        raise ValueError("artifact hash does not match canonical bytes")
    write_json(path, artifact.to_dict(), exclusive=True)
