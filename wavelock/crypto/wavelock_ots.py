"""
WaveLock-OTS — an asymmetric, public-verifier one-time signature.

Why this module exists
----------------------
The legacy WaveLock signature (``WLv2``/``SIGv2`` in
:mod:`wavelock.chain.WaveLock`) is **not** an asymmetric signature. There,

    signature = H("SIGv2" || message || header || ψ★)

and ``verify()`` recomputes exactly that hash, so the verifier must possess
ψ★ — the very secret that produces the commitment. Anyone able to verify is
therefore able to forge. That is a symmetric MAC keyed by a published secret,
not a public-key signature. See ``docs/WAVELOCK_OTS_DESIGN.md`` and
``attacks/forge_from_snapshot.py``.

WaveLock-OTS restores the minimum asymmetric property that Lamport already
satisfies, *without* abandoning the WaveLock thesis:

* The public key contains **only** commitments (hashes, a Merkle root, a
  ψ-state commitment) plus parameters/metadata. It never contains ψ★, ψ₀,
  the seed, or any raw secret slice.
* The secret key contains a high-entropy (≥128-bit, default 256-bit) seed and
  the ψ-generation parameters. The ψ-derived secret slices are *derived on
  demand* from the seed, never stored in any public artifact.
* A signature reveals only the message-selected secret slices (one of two per
  message-digest bit, Lamport-style). The unrevealed half stays secret, so the
  verifier — who only ever sees the selected halves — cannot forge a different
  message.
* Each key is **one-time** by default; reuse is detected and loudly rejected.

WaveLock binding
----------------
The "WaveLock" in WaveLock-OTS is not decoration. The Lamport secret slices
are not raw random bytes; they are derived through the WaveLock ψ pipeline:

    seed ──SHAKE256(WL-PSI-INIT)──► ψ₀ ──PDE evolution──► ψ★
    ψ★ ──quantize+SHAKE256──► psi_commitment
    sk[i][b] = SHAKE256("WL-OTS-SK" || seed || psi_commitment || params_hash || i || b)

so producing a valid signature requires regenerating the exact ψ evolution.
``psi_commitment`` is published (it is a hash, not ψ★), binding every public
key and signature to a specific curvature/ψ-state evolution while keeping ψ★
itself private.

This is experimental. It is *not* a reviewed cryptographic standard. Do not
use it for production funds. For production, use Ed25519, SLH-DSA, LMS, or
XMSS (see ``docs/MIGRATION_FROM_SIGV2.md``).
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import struct
import tempfile
import uuid
from typing import Optional

import numpy as np

# WaveLock ψ pipeline: deterministic ψ₀ derivation + PDE kernel parameters.
# Importing the kernel constants from the canonical module keeps WaveLock-OTS
# bound to the *same* physics as the rest of WaveLock; if the kernel changes,
# params_hash changes, and old keys/signatures no longer validate.
from wavelock.chain.xof_init import derive_psi_zero
from wavelock.chain.WaveLock import (
    alpha,
    beta,
    theta,
    epsilon,
    delta,
    _dt,
    _steps,
    _damping,
    KERNEL_VERSION,
    _kernel_hash,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCHEME = "WaveLock-OTS-v1"
HASH_ALG = "SHAKE256-256"

#: Wire/version integer carried in every public key and signature. Strict
#: verification rejects anything that does not equal this exact value.
VERSION = 1

#: Output length in bytes for all SHAKE256-256 outputs (digests, slices, pk).
_HASH_LEN = 32

#: Number of message-digest bits = number of Lamport bit positions.
_N_BITS = 256

#: Minimum acceptable seed entropy. 128 bits is the floor; 256 is the default.
MIN_ENTROPY_BITS = 128
DEFAULT_ENTROPY_BITS = 256

# ---------------------------------------------------------------------------
# Domain separators
# ---------------------------------------------------------------------------
# Every hash in WaveLock-OTS is domain-separated. Distinct, versioned tags make
# it impossible for a digest computed for one purpose (e.g. a Merkle leaf) to be
# reinterpreted as another (e.g. an internal node or a fingerprint).
_DOM_PK_FINGERPRINT = b"WL-OTS-PK-FINGERPRINT-v1"  # public-key fingerprint
_DOM_MERKLE_LEAF = b"WL-OTS-MERKLE-LEAF-v1"        # Merkle leaf hash
_DOM_MERKLE_NODE = b"WL-OTS-MERKLE-NODE-v1"        # Merkle internal node hash
_DOM_MERKLE_EMPTY = b"WL-OTS-MERKLE-EMPTY-v1"      # (defensive) empty tree
_DOM_MSG = b"WL-OTS-MSG-v1"                        # message digest
_DOM_SIG_TRANSCRIPT = b"WL-OTS-SIG-TRANSCRIPT-v1"  # signature transcript

# ---------------------------------------------------------------------------
# Canonical object shapes (strict verification rejects anything else)
# ---------------------------------------------------------------------------
#: The EXACT field set of a canonical WaveLock-OTS public key. Strict
#: verification and load reject a public key whose keys differ from this set
#: (no unknown fields, no missing fields). ``params`` is intentionally NOT a
#: public-key field: it is bound through ``params_hash`` (and the fingerprint),
#: and the full parameter set lives only in the secret key.
PUBLIC_KEY_FIELDS = (
    "scheme",
    "version",
    "hash_alg",
    "params_hash",
    "psi_commitment",
    "one_time_key_id",
    "pk_commitments",
    "merkle_root",
    "public_key_fingerprint",
)

#: The EXACT field set of a canonical WaveLock-OTS signature.
SIGNATURE_FIELDS = (
    "scheme",
    "version",
    "hash_alg",
    "one_time_key_id",
    "public_key_fingerprint",
    "params_hash",
    "psi_commitment",
    "message_digest",
    "revealed_slices",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class WaveLockOTSError(Exception):
    """Base class for all WaveLock-OTS failures."""


class OTSKeyReuseError(WaveLockOTSError):
    """Raised when a one-time key is used to sign more than once."""


class InsufficientEntropyError(WaveLockOTSError):
    """Raised when seed material is below the 128-bit minimum."""


# ---------------------------------------------------------------------------
# Hashing primitives (SHAKE256-256, length-prefixed domain separation)
# ---------------------------------------------------------------------------


def _h(domain: bytes, *parts: bytes, out_len: int = _HASH_LEN) -> bytes:
    """SHAKE256 of a domain-separated, length-prefixed concatenation.

    Every field is prefixed with its big-endian 8-byte length so that no two
    distinct ``(domain, parts)`` tuples can ever collide on the byte stream
    that is hashed. This is the single hashing primitive used everywhere in
    WaveLock-OTS.
    """
    x = hashlib.shake_256()
    x.update(struct.pack(">Q", len(domain)))
    x.update(domain)
    for p in parts:
        x.update(struct.pack(">Q", len(p)))
        x.update(p)
    return x.digest(out_len)


def _i2b(i: int) -> bytes:
    """Encode a non-negative integer as a fixed 4-byte big-endian field."""
    return struct.pack(">I", i)


# ---------------------------------------------------------------------------
# Parameters + ψ pipeline
# ---------------------------------------------------------------------------


def default_params(n: int = 4) -> dict:
    """Canonical, JSON-safe WaveLock-OTS parameter set.

    ``n`` controls the ψ field side length exactly as in
    :class:`wavelock.chain.WaveLock.CurvatureKeyPair` (``side = 2**(n//2)``).
    """
    side = 2 ** max(1, n // 2)
    return {
        "scheme": SCHEME,
        "hash_alg": HASH_ALG,
        "n": int(n),
        "side": int(side),
        "n_bits": _N_BITS,
        "kernel_version": KERNEL_VERSION,
        "kernel_hash": _kernel_hash(),
        "pde": {
            "alpha": float(alpha),
            "beta": float(beta),
            "theta": float(theta),
            "epsilon": float(epsilon),
            "delta": float(delta),
            "dt": float(_dt),
            "steps": int(_steps),
            "damping": float(_damping),
        },
    }


def _canon(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def params_hash(params: dict) -> bytes:
    """Hash the canonical parameter set. Binds keys/signatures to the kernel."""
    return _h(b"WL-OTS-PARAMS-v1", _canon(params))


def _laplacian_np(x: np.ndarray) -> np.ndarray:
    """5-point periodic Laplacian (NumPy reference, backend-independent).

    Mirrors :func:`wavelock.chain.WaveLock.laplacian` but is pinned to NumPy so
    WaveLock-OTS ψ evolution is byte-deterministic on every platform — the
    consensus/reference path, never a GPU RNG path.
    """
    return (
        -4.0 * x
        + np.roll(x, +1, 0)
        + np.roll(x, -1, 0)
        + np.roll(x, +1, 1)
        + np.roll(x, -1, 1)
    )


def evolve_psi_star(seed_bytes: bytes, params: dict) -> np.ndarray:
    """Run the full WaveLock ψ pipeline: seed → ψ₀ → ψ★.

    Deterministic in ``seed_bytes`` and ``params`` alone (SHAKE256 ψ₀ +
    fixed-step NumPy PDE). Reproduces the same ψ★ on CPU or GPU hosts.
    """
    side = int(params["side"])
    pde = params["pde"]
    a, b, th = float(pde["alpha"]), float(pde["beta"]), float(pde["theta"])
    eps, dl = float(pde["epsilon"]), float(pde["delta"])
    dt, steps, damping = float(pde["dt"]), int(pde["steps"]), float(pde["damping"])

    psi = derive_psi_zero(seed_bytes, (side, side)).astype(np.float64, copy=True)
    for _ in range(steps):
        lap = _laplacian_np(psi)
        fb = a * lap / (psi + eps * np.exp(-b * psi**2))
        ent = th * (psi * _laplacian_np(np.log(psi**2 + dl)))
        dpsi = dt * (fb - ent) - damping * psi
        psi = psi + dpsi
    return psi


def _quantize_psi(psi: np.ndarray) -> bytes:
    """Deterministically quantize ψ★ to bytes for commitment.

    Rounding to a fixed grid before hashing means the commitment is stable
    against the last-bit float noise that would otherwise differ between
    independent re-derivations, while still binding to the ψ-state shape.
    """
    q = np.rint(np.asarray(psi, dtype=np.float64) * 1e9).astype(np.int64)
    return q.ravel(order="C").tobytes()


def psi_commitment(psi: np.ndarray) -> bytes:
    """psi_commitment = H("WL-PSI-COMMIT" || quantized_psi_star)."""
    return _h(b"WL-PSI-COMMIT", _quantize_psi(psi))


# ---------------------------------------------------------------------------
# Lamport-over-ψ secret/public slices
# ---------------------------------------------------------------------------


def _secret_slice(seed_bytes: bytes, psi_commit: bytes, p_hash: bytes,
                  i: int, b: int) -> bytes:
    """sk[i][b] derived through the WaveLock ψ binding.

    Bound to: the high-entropy seed, the ψ-state (via ``psi_commit``), and the
    full parameter set (via ``p_hash``). Computing this requires the seed, so
    only the secret-key holder can produce signatures; ``psi_commit`` ties the
    secret to a specific ψ evolution.
    """
    return _h(b"WL-OTS-SK-v1", seed_bytes, psi_commit, p_hash, _i2b(i), bytes([b]))


def _public_slice(p_hash: bytes, i: int, b: int, sk: bytes) -> bytes:
    """pk[i][b] = H("WL-OTS-PK" || params_hash || i || b || sk[i][b])."""
    return _h(b"WL-OTS-PK-v1", p_hash, _i2b(i), bytes([b]), sk)


def _merkle_root(leaves: list[bytes]) -> bytes:
    """Binary Merkle root over leaf hashes (odd levels duplicate the last)."""
    if not leaves:
        return _h(_DOM_MERKLE_EMPTY)
    level = [_h(_DOM_MERKLE_LEAF, leaf) for leaf in leaves]
    while len(level) > 1:
        if len(level) % 2 == 1:
            level.append(level[-1])
        level = [
            _h(_DOM_MERKLE_NODE, level[i], level[i + 1])
            for i in range(0, len(level), 2)
        ]
    return level[0]


def _merkle_root_from_commitments(pk_commitments) -> bytes:
    """Recompute the Merkle root from the published ``pk_commitments`` array.

    The leaves are the per-(bit, half) public commitments, flattened in the same
    ``(i, b)`` order used at key generation: ``pk[0][0], pk[0][1], pk[1][0], …``.
    Raises on any structurally invalid commitment so callers fail closed.
    """
    leaves: list[bytes] = []
    for row in pk_commitments:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("each pk_commitments row must hold exactly 2 halves")
        for half in row:
            leaves.append(bytes.fromhex(half))
    return _merkle_root(leaves)


def _message_digest(message, p_hash: bytes) -> bytes:
    """Domain-separated, params-bound message digest (n_bits/8 bytes)."""
    if isinstance(message, str):
        msg = message.encode("utf-8")
    elif isinstance(message, (bytes, bytearray)):
        msg = bytes(message)
    else:
        raise TypeError("message must be str or bytes")
    return _h(_DOM_MSG, p_hash, msg, out_len=_N_BITS // 8)


def _digest_bits(digest: bytes) -> list[int]:
    """Expand the digest into n_bits bits, MSB-first within each byte."""
    bits = []
    for byte in digest:
        for k in range(7, -1, -1):
            bits.append((byte >> k) & 1)
    return bits


# ---------------------------------------------------------------------------
# Canonical public-key fingerprint
# ---------------------------------------------------------------------------


def _canonical_public_key_payload(public_key: dict) -> dict:
    """The canonical, fingerprint-covered view of a public key.

    Exactly the public-key fields *except* ``public_key_fingerprint`` itself.
    Binding ``pk_commitments`` and ``merkle_root`` in here is what makes a
    fingerprint/key-substitution attack impossible: you cannot keep a victim's
    fingerprint while swapping in attacker commitments — the recomputed
    fingerprint changes.
    """
    return {
        "scheme": public_key["scheme"],
        "version": public_key["version"],
        "hash_alg": public_key["hash_alg"],
        "params_hash": public_key["params_hash"],
        "psi_commitment": public_key["psi_commitment"],
        "one_time_key_id": public_key["one_time_key_id"],
        "pk_commitments": public_key["pk_commitments"],
        "merkle_root": public_key["merkle_root"],
    }


def public_key_fingerprint(public_key: dict) -> str:
    """Recompute the canonical fingerprint of a public key (hex).

    fingerprint = H(WL-OTS-PK-FINGERPRINT-v1 || canonical_json(payload)) where
    ``payload`` is every public-key field except the fingerprint itself, in
    sorted-key canonical JSON. This binds scheme/version/hash_alg/params_hash/
    psi_commitment/one_time_key_id/pk_commitments/merkle_root into one value.
    """
    payload = _canonical_public_key_payload(public_key)
    return _h(_DOM_PK_FINGERPRINT, _canon(payload)).hex()


def signature_transcript(signature: dict) -> str:
    """Deterministic transcript hash over the canonical signature (hex).

    A stable identifier for a signature object that a replay ledger / server can
    dedupe on. Domain-separated so it can never collide with a fingerprint, a
    message digest, or a Merkle node.
    """
    payload = {k: signature.get(k) for k in SIGNATURE_FIELDS}
    return _h(_DOM_SIG_TRANSCRIPT, _canon(payload)).hex()


# ---------------------------------------------------------------------------
# Entropy validation
# ---------------------------------------------------------------------------


def _validate_seed(seed_bytes: bytes) -> None:
    """Reject tiny / low-entropy seeds. No small-integer seeds allowed."""
    if len(seed_bytes) * 8 < MIN_ENTROPY_BITS:
        raise InsufficientEntropyError(
            f"seed has {len(seed_bytes) * 8} bits; WaveLock-OTS requires "
            f">= {MIN_ENTROPY_BITS} bits (prefer {DEFAULT_ENTROPY_BITS})."
        )
    # Guard against an all-zero / trivially-constant seed sneaking through.
    if len(set(seed_bytes)) <= 1:
        raise InsufficientEntropyError(
            "seed is constant (zero effective entropy); refusing."
        )


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------


def generate_ots_keypair(params: Optional[dict] = None,
                         entropy_bits: int = DEFAULT_ENTROPY_BITS,
                         seed: Optional[bytes] = None) -> dict:
    """Generate a one-time WaveLock-OTS keypair.

    Parameters
    ----------
    params : dict, optional
        WaveLock-OTS parameter set (see :func:`default_params`). Defaults to
        ``default_params()``.
    entropy_bits : int
        Seed entropy in bits. Must be >= 128 (default 256). Tiny seeds are
        rejected — there are no small-integer seeds in WaveLock-OTS.
    seed : bytes, optional
        Explicit seed material (for deterministic tests/vectors). Must satisfy
        the entropy floor. If omitted, ``os.urandom(entropy_bits // 8)``.

    Returns
    -------
    dict
        ``{"public_key": {...}, "secret_key": {...}}`` — both JSON-safe.
    """
    if entropy_bits < MIN_ENTROPY_BITS:
        raise InsufficientEntropyError(
            f"entropy_bits={entropy_bits} below minimum {MIN_ENTROPY_BITS}."
        )
    if params is None:
        params = default_params()

    if seed is None:
        seed_bytes = os.urandom(entropy_bits // 8)
    else:
        seed_bytes = bytes(seed)
    _validate_seed(seed_bytes)

    p_hash = params_hash(params)
    psi = evolve_psi_star(seed_bytes, params)
    psi_commit = psi_commitment(psi)

    # Derive the 2 * n_bits public-slice commitments. Secret slices are NOT
    # stored — they are re-derived on demand at signing time from the seed.
    n_bits = int(params["n_bits"])
    pk_commitments: list[list[str]] = []
    leaves: list[bytes] = []
    for i in range(n_bits):
        row = []
        for b in (0, 1):
            sk = _secret_slice(seed_bytes, psi_commit, p_hash, i, b)
            pk = _public_slice(p_hash, i, b, sk)
            row.append(pk.hex())
            leaves.append(pk)
        pk_commitments.append(row)

    merkle_root = _merkle_root(leaves)
    one_time_key_id = str(uuid.uuid4())
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Build the canonical public key (exactly PUBLIC_KEY_FIELDS), then bind every
    # field into the fingerprint. ``params`` is NOT a public-key field — it is
    # bound through ``params_hash`` and kept only in the secret key.
    public_key = {
        "scheme": SCHEME,
        "version": VERSION,
        "hash_alg": HASH_ALG,
        "params_hash": p_hash.hex(),
        "psi_commitment": psi_commit.hex(),
        "one_time_key_id": one_time_key_id,
        "pk_commitments": pk_commitments,
        "merkle_root": merkle_root.hex(),
    }
    public_key["public_key_fingerprint"] = public_key_fingerprint(public_key)

    secret_key = {
        "scheme": SCHEME,
        "hash_alg": HASH_ALG,
        "version": VERSION,
        "params": params,
        "params_hash": p_hash.hex(),
        # High-entropy seed material. Stays local; never written to a *public*
        # artifact. Encrypt at rest with export_secret_key(encrypt=True).
        "seed_hex": seed_bytes.hex(),
        "psi_commitment": psi_commit.hex(),
        "one_time_key_id": one_time_key_id,
        # Bind the secret key to the public identity it signs under, so signing
        # can stamp the fingerprint into the signature without the public key.
        "public_key_fingerprint": public_key["public_key_fingerprint"],
        "created_at": created_at,
        "used": False,
    }

    return {"public_key": public_key, "secret_key": secret_key}


# ---------------------------------------------------------------------------
# Signing
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Local one-time key-state registry (host-local mitigation for Finding D)
# ---------------------------------------------------------------------------
# The `used` bool in the secret-key dict/file is advisory: a copy taken BEFORE
# signing has used=False and would sign again. This registry adds a host-local,
# atomic guard: signing atomically *claims* a one_time_key_id (O_CREAT|O_EXCL),
# so a second signing attempt under the same id on the same host fails closed —
# even from a separate copy of the secret file.
#
# HONEST LIMITS: this is NOT cryptographic reuse prevention. A secret key copied
# to a DIFFERENT host (no shared registry), or a wiped registry, still bypasses
# it. Production MUST additionally reject duplicate one_time_key_id / duplicate
# Merkle-leaf usage at the verifier/server/ledger layer (see OTSReplayLedger and
# docs/WAVELOCK_MERKLE_ROADMAP.md).


def _state_dir() -> str:
    """Directory backing the local key-state registry.

    Override with ``WAVELOCK_OTS_STATE_DIR``; defaults to a per-user temp dir.
    """
    d = os.environ.get("WAVELOCK_OTS_STATE_DIR")
    if not d:
        d = os.path.join(tempfile.gettempdir(), "wavelock-ots-state")
    return d


def _key_id_marker_path(one_time_key_id: str) -> str:
    h = hashlib.sha256(str(one_time_key_id).encode("utf-8")).hexdigest()
    return os.path.join(_state_dir(), h + ".used")


def _claim_one_time_key(one_time_key_id: str) -> bool:
    """Atomically claim ``one_time_key_id``. True if newly claimed, else False.

    Uses ``O_CREAT | O_EXCL`` so two concurrent signers race-safely: exactly one
    wins the create; the loser sees the marker already exists and is refused.
    """
    if not one_time_key_id:
        return True  # nothing to bind on; in-memory `used` guard still applies
    path = _key_id_marker_path(one_time_key_id)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        return False
    except OSError:
        # If we cannot use the registry at all, fall back to the in-memory guard
        # rather than blocking signing. (Documented: registry is best-effort.)
        return True
    try:
        os.write(fd, (str(one_time_key_id) + "\n").encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    return True


def _seed_from_secret(secret_key: dict) -> bytes:
    if "seed_hex" not in secret_key or secret_key["seed_hex"] is None:
        raise WaveLockOTSError(
            "secret key has no seed material (is it an encrypted export? "
            "load it with the passphrase first)."
        )
    return bytes.fromhex(secret_key["seed_hex"])


def sign_ots(secret_key: dict, message, allow_reuse: bool = False) -> dict:
    """Sign ``message`` with a one-time WaveLock-OTS secret key.

    Reveals exactly one secret slice per message-digest bit (the half selected
    by the bit). The unrevealed halves stay secret — that asymmetry is what
    stops the verifier from forging a different message.

    The key is marked one-time-used on success. A second call raises
    :class:`OTSKeyReuseError` unless ``allow_reuse=True`` (tests only).
    """
    if secret_key.get("scheme") != SCHEME:
        raise WaveLockOTSError(f"not a {SCHEME} secret key")

    one_time_key_id = secret_key.get("one_time_key_id")

    if secret_key.get("used") and not allow_reuse:
        raise OTSKeyReuseError(
            "WaveLock-OTS secret key has already signed once. One-time keys "
            "MUST NOT be reused — reuse leaks unrevealed secret halves and "
            "enables forgery. Generate a fresh key. (override: allow_reuse=True)"
        )

    # Host-local atomic guard: claim the one_time_key_id before signing so a
    # copy of the secret file (used=False) cannot sign a second message on this
    # host. Skipped under allow_reuse (tests/demo only).
    if not allow_reuse and not _claim_one_time_key(one_time_key_id):
        raise OTSKeyReuseError(
            "WaveLock-OTS one_time_key_id has already been consumed on this "
            f"host (local key-state registry): {one_time_key_id}. A copied "
            "secret key cannot sign twice here. (override: allow_reuse=True)"
        )

    params = secret_key["params"]
    p_hash = bytes.fromhex(secret_key["params_hash"])
    seed_bytes = _seed_from_secret(secret_key)
    _validate_seed(seed_bytes)

    # Re-run the full WaveLock ψ pipeline; this is the WaveLock binding step.
    psi = evolve_psi_star(seed_bytes, params)
    psi_commit = psi_commitment(psi)
    if secret_key.get("psi_commitment") and \
            psi_commit.hex() != secret_key["psi_commitment"]:
        raise WaveLockOTSError(
            "re-derived psi_commitment does not match secret key; corrupt "
            "seed/params or kernel drift."
        )

    digest = _message_digest(message, p_hash)
    bits = _digest_bits(digest)

    revealed = []
    for i, bit in enumerate(bits):
        sk = _secret_slice(seed_bytes, psi_commit, p_hash, i, bit)
        revealed.append(sk.hex())

    # Mark the in-memory secret key as used (callers persist this).
    secret_key["used"] = True

    fingerprint = secret_key.get("public_key_fingerprint")
    if not fingerprint:
        raise WaveLockOTSError(
            "secret key is missing public_key_fingerprint; regenerate the key "
            "with this version (the signature must bind to a key identity)."
        )

    return {
        "scheme": SCHEME,
        "version": VERSION,
        "hash_alg": HASH_ALG,
        "one_time_key_id": one_time_key_id,
        "public_key_fingerprint": fingerprint,
        "params_hash": secret_key["params_hash"],
        "psi_commitment": psi_commit.hex(),
        "message_digest": digest.hex(),
        "revealed_slices": revealed,
    }


# ---------------------------------------------------------------------------
# Verification (public, fail-closed)
# ---------------------------------------------------------------------------


def _check_public_key_canonical(public_key: dict) -> str:
    """Validate a public key's canonical shape and self-consistency.

    Returns the recomputed canonical fingerprint on success. Raises
    :class:`WaveLockOTSError` on ANY problem: unknown/missing fields, wrong
    constants, a ``merkle_root`` that does not match ``pk_commitments``, or a
    stored ``public_key_fingerprint`` that does not match the recomputed one.

    This is the single place that (a) enforces the exact field set, (b)
    recomputes the Merkle root from ``pk_commitments``, and (c) recomputes and
    checks the fingerprint — closing Finding A (garbage roots / key
    substitution).
    """
    if not isinstance(public_key, dict):
        raise WaveLockOTSError("public key must be a dict")

    keys = set(public_key.keys())
    expected = set(PUBLIC_KEY_FIELDS)
    if keys != expected:
        missing = expected - keys
        extra = keys - expected
        raise WaveLockOTSError(
            f"public key has non-canonical fields (missing={sorted(missing)}, "
            f"unknown={sorted(extra)})"
        )

    if public_key["scheme"] != SCHEME:
        raise WaveLockOTSError("public key has wrong scheme")
    if public_key["version"] != VERSION:
        raise WaveLockOTSError("public key has wrong version")
    if public_key["hash_alg"] != HASH_ALG:
        raise WaveLockOTSError("public key has wrong hash_alg")

    pk_commitments = public_key["pk_commitments"]
    if not isinstance(pk_commitments, list) or len(pk_commitments) != _N_BITS:
        raise WaveLockOTSError(
            f"pk_commitments must be a list of {_N_BITS} [half0, half1] rows"
        )

    # (A) Recompute the Merkle root over pk_commitments and require a match.
    recomputed_root = _merkle_root_from_commitments(pk_commitments).hex()
    if not _ct_eq(recomputed_root, str(public_key["merkle_root"])):
        raise WaveLockOTSError(
            "merkle_root does not match pk_commitments (tampered/garbage root)"
        )

    # (A) Recompute the fingerprint over the canonical payload and require a
    # match. This binds pk_commitments/merkle_root/params_hash/psi_commitment/
    # one_time_key_id to the advertised identity.
    recomputed_fp = public_key_fingerprint(public_key)
    if not _ct_eq(recomputed_fp, str(public_key["public_key_fingerprint"])):
        raise WaveLockOTSError(
            "public_key_fingerprint does not match the canonical public key "
            "(fingerprint/key-substitution attempt)"
        )
    return recomputed_fp


def verify_ots(public_key: dict, message, signature: dict) -> bool:
    """Verify a WaveLock-OTS signature using ONLY public material.

    Strict and fail-closed. A signature is accepted iff ALL of the following
    hold (any failure, or any raised exception, returns ``False``):

    Public key (Finding A):
      * exactly the canonical fields, correct scheme/version/hash_alg;
      * ``merkle_root`` recomputes from ``pk_commitments``;
      * ``public_key_fingerprint`` recomputes from the canonical public key.

    Signature (Finding B):
      * exactly the canonical fields — no missing, no extra;
      * correct scheme/version/hash_alg;
      * ``message_digest`` present and equal to the recomputed digest;
      * ``one_time_key_id`` == public key's ``one_time_key_id``;
      * ``public_key_fingerprint`` == recomputed public-key fingerprint;
      * ``params_hash`` == public key's ``params_hash``;
      * ``psi_commitment`` == public key's ``psi_commitment``;
      * ``revealed_slices`` length == digest bit length;
      * every revealed slice hashes to ``pk[i][selected_bit]``.

    Never touches ψ★, the seed, or any unrevealed slice; the verifier cannot
    forge because it only ever sees the message-selected halves.
    """
    try:
        if not isinstance(public_key, dict) or not isinstance(signature, dict):
            return False

        # --- Public key: canonical shape + self-consistency (Finding A). ------
        fingerprint = _check_public_key_canonical(public_key)
        p_hash_hex = str(public_key["params_hash"])
        try:
            p_hash = bytes.fromhex(p_hash_hex)
        except (ValueError, TypeError):
            return False

        # --- Signature: exact canonical field set, no extras (Finding B). -----
        if set(signature.keys()) != set(SIGNATURE_FIELDS):
            return False
        if signature["scheme"] != SCHEME:
            return False
        if signature["version"] != VERSION:
            return False
        if signature["hash_alg"] != HASH_ALG:
            return False

        # Bind the signature to THIS public identity (Finding B).
        if signature["one_time_key_id"] != public_key["one_time_key_id"]:
            return False
        if not _ct_eq(str(signature["public_key_fingerprint"]), fingerprint):
            return False
        if not _ct_eq(str(signature["params_hash"]), p_hash_hex):
            return False
        if not _ct_eq(str(signature["psi_commitment"]),
                      str(public_key["psi_commitment"])):
            return False

        # Recompute the message digest; require the carried digest to be present
        # AND equal (no missing-field bypass).
        digest = _message_digest(message, p_hash)
        if not _ct_eq(str(signature["message_digest"]), digest.hex()):
            return False
        bits = _digest_bits(digest)

        revealed = signature["revealed_slices"]
        pk_commitments = public_key["pk_commitments"]
        if not isinstance(revealed, list) or len(revealed) != len(bits):
            return False

        for i, bit in enumerate(bits):
            try:
                sk = bytes.fromhex(revealed[i])
            except (ValueError, TypeError):
                return False
            expected_pk = _public_slice(p_hash, i, bit, sk).hex()
            committed_pk = pk_commitments[i][bit]
            if not _ct_eq(expected_pk, committed_pk):
                return False
        return True
    except Exception:
        # Verification must never raise into callers; unknown failure = reject.
        return False


def _ct_eq(a: str, b: str) -> bool:
    """Constant-time-ish string comparison."""
    import hmac

    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# ---------------------------------------------------------------------------
# Server / ledger replay model (Finding D — deployment-layer duplicate rejection)
# ---------------------------------------------------------------------------


class OTSReplayLedger:
    """Minimal model of the duplicate-use rejection a deployment MUST provide.

    WaveLock-OTS is one-time (Finding C: reuse → total forgery is *inherent* to
    Lamport-style OTS and is NOT fixed by the A/B hardening). The local
    key-state registry (:func:`_claim_one_time_key`) only protects a single
    host. A production verifier/server/ledger must additionally refuse a second
    *accepted* signature under the same ``one_time_key_id`` (and, under
    WaveLock-Merkle, the same consumed leaf index). This class is that check,
    consumed by tests and available for server integration. It is intentionally
    fail-closed: a signature that does not verify is never recorded or accepted.

    This is a reference/model object — it is NOT yet wired into block/consensus
    acceptance (see docs/WAVELOCK_MERKLE_ROADMAP.md and the report).
    """

    def __init__(self):
        self._consumed_key_ids: set[str] = set()

    def is_consumed(self, one_time_key_id: str) -> bool:
        return str(one_time_key_id) in self._consumed_key_ids

    def accept(self, public_key: dict, message, signature: dict) -> bool:
        """Verify then consume. Returns True only on first valid use.

        Rejects (returns False) if the signature is invalid OR if its
        ``one_time_key_id`` has already been consumed (replay/reuse).
        """
        if not verify_ots(public_key, message, signature):
            return False
        kid = str(signature.get("one_time_key_id"))
        if kid in self._consumed_key_ids:
            return False
        self._consumed_key_ids.add(kid)
        return True


# ---------------------------------------------------------------------------
# Export / load
# ---------------------------------------------------------------------------

#: Fields that must NEVER appear in a public artifact.
_FORBIDDEN_PUBLIC_FIELDS = ("seed", "seed_hex", "psi_0", "psi_star", "revealed_sk")


def export_public_key(public_key: dict) -> dict:
    """Return a JSON-safe public-key dict, asserting no secret leakage.

    Defensive: raises if any forbidden secret field is present. The public key
    is, by construction, commitments + hashes + Merkle root + params + metadata.
    """
    for f in _FORBIDDEN_PUBLIC_FIELDS:
        if f in public_key:
            raise WaveLockOTSError(
                f"refusing to export public key: forbidden secret field {f!r}"
            )
    return json.loads(json.dumps(public_key))


def export_secret_key(secret_key: dict, encrypt: bool = False,
                      passphrase: Optional[str] = None,
                      unsafe_export_secret_state: bool = False) -> dict:
    """Return a JSON-safe secret-key dict.

    By default this contains the local seed in cleartext (the secret key file
    is expected to stay on the owner's machine and never be published). Raw
    ψ★/ψ₀ are *not* included unless ``unsafe_export_secret_state=True``, which
    prints a loud warning.

    Parameters
    ----------
    encrypt : bool
        If True, encrypt the seed at rest under a passphrase (scrypt KDF +
        SHAKE256 keystream XOR). The exported dict then carries ``seed_enc``
        instead of ``seed_hex``.
    passphrase : str, optional
        Required when ``encrypt=True``. May also be supplied via the
        ``WAVELOCK_OTS_PASSPHRASE`` environment variable.
    unsafe_export_secret_state : bool
        If True, also embed the raw quantized ψ★ bytes. Strongly discouraged.
    """
    out = json.loads(json.dumps(secret_key))

    if encrypt:
        pw = passphrase or os.getenv("WAVELOCK_OTS_PASSPHRASE")
        if not pw:
            raise WaveLockOTSError(
                "encrypt=True requires a passphrase (arg or "
                "WAVELOCK_OTS_PASSPHRASE env var)."
            )
        seed_bytes = bytes.fromhex(out.pop("seed_hex"))
        salt = os.urandom(16)
        out["seed_enc"] = _encrypt_seed(seed_bytes, pw, salt)
        out["kdf"] = {"name": "scrypt", "n": 2**14, "r": 8, "p": 1,
                      "salt": salt.hex()}

    if unsafe_export_secret_state:
        print(
            "\n*** WARNING: --unsafe-export-secret-state ***\n"
            "Embedding raw ψ★ material in the secret key export. This defeats "
            "WaveLock-OTS's secrecy guarantees if the file leaks. Never publish "
            "this file. Never use it as verifier material.\n"
        )
        seed_bytes = _seed_from_secret(secret_key)
        psi = evolve_psi_star(seed_bytes, secret_key["params"])
        out["UNSAFE_psi_star_quantized_hex"] = _quantize_psi(psi).hex()

    return out


def _encrypt_seed(seed_bytes: bytes, passphrase: str, salt: bytes) -> str:
    key = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt,
                         n=2**14, r=8, p=1, dklen=len(seed_bytes) + 32)
    keystream = _h(b"WL-OTS-SEED-ENC-v1", key, salt, out_len=len(seed_bytes))
    enc = bytes(a ^ b for a, b in zip(seed_bytes, keystream))
    return enc.hex()


def _decrypt_seed(seed_enc_hex: str, passphrase: str, kdf: dict) -> bytes:
    salt = bytes.fromhex(kdf["salt"])
    enc = bytes.fromhex(seed_enc_hex)
    key = hashlib.scrypt(passphrase.encode("utf-8"), salt=salt,
                         n=int(kdf["n"]), r=int(kdf["r"]), p=int(kdf["p"]),
                         dklen=len(enc) + 32)
    keystream = _h(b"WL-OTS-SEED-ENC-v1", key, salt, out_len=len(enc))
    return bytes(a ^ b for a, b in zip(enc, keystream))


def load_public_key(source) -> dict:
    """Load and strictly validate a public key (path/file/JSON string/dict).

    Beyond the secret-leak guards, this enforces the canonical public-key shape
    and recomputes the Merkle root + fingerprint, raising on any mismatch. A
    public key that does not validate canonically is rejected at load time, so a
    tampered/garbage root or a spliced fingerprint never reaches verification.
    """
    pk = _load_json(source)
    if pk.get("scheme") != SCHEME:
        raise WaveLockOTSError(f"not a {SCHEME} public key")
    for f in _FORBIDDEN_PUBLIC_FIELDS:
        if f in pk:
            raise WaveLockOTSError(
                f"public key file contains forbidden secret field {f!r}; "
                "this file is compromised — do not trust it."
            )
    # Canonical shape + Merkle-root + fingerprint self-consistency (Finding A).
    _check_public_key_canonical(pk)
    return pk


def load_secret_key(source, passphrase: Optional[str] = None) -> dict:
    """Load a secret key (decrypting the seed if it was exported encrypted)."""
    sk = _load_json(source)
    if sk.get("scheme") != SCHEME:
        raise WaveLockOTSError(f"not a {SCHEME} secret key")
    if "seed_enc" in sk and "seed_hex" not in sk:
        pw = passphrase or os.getenv("WAVELOCK_OTS_PASSPHRASE")
        if not pw:
            raise WaveLockOTSError(
                "secret key seed is encrypted; supply passphrase (arg or "
                "WAVELOCK_OTS_PASSPHRASE)."
            )
        seed = _decrypt_seed(sk["seed_enc"], pw, sk["kdf"])
        sk["seed_hex"] = seed.hex()
    # Validate entropy of the (decrypted) seed. Reject tiny seeds on load too.
    if "seed_hex" in sk:
        _validate_seed(bytes.fromhex(sk["seed_hex"]))
    return sk


def _load_json(source) -> dict:
    if isinstance(source, dict):
        return json.loads(json.dumps(source))
    if hasattr(source, "read"):
        return json.load(source)
    if isinstance(source, (str, os.PathLike)):
        s = os.fspath(source) if isinstance(source, os.PathLike) else source
        if os.path.exists(s):
            with open(s, "r") as fh:
                return json.load(fh)
        # Treat as a JSON string literal.
        return json.loads(s)
    raise TypeError(f"cannot load JSON from {type(source).__name__}")


__all__ = [
    "WaveLockOTSError",
    "OTSKeyReuseError",
    "InsufficientEntropyError",
    "SCHEME",
    "HASH_ALG",
    "VERSION",
    "MIN_ENTROPY_BITS",
    "DEFAULT_ENTROPY_BITS",
    "PUBLIC_KEY_FIELDS",
    "SIGNATURE_FIELDS",
    "default_params",
    "params_hash",
    "evolve_psi_star",
    "psi_commitment",
    "public_key_fingerprint",
    "signature_transcript",
    "generate_ots_keypair",
    "sign_ots",
    "verify_ots",
    "OTSReplayLedger",
    "export_public_key",
    "export_secret_key",
    "load_public_key",
    "load_secret_key",
]
