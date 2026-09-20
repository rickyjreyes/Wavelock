"""Canonical WaveLock-OTS block transcripts and public verification.

Shared by the CLI and node. Importing this module does not create replay state
or start a node. Transcript version 1 remains byte-compatible with prior OTS
blocks; legacy SIGv2 is never accepted by these helpers.
"""
from __future__ import annotations

import hashlib
import hmac
from typing import Optional

from wavelock.chain.Block import Block
from wavelock.chain.WaveLock import _canonical_json
from wavelock.crypto.wavelock_ots import SCHEME as OTS_SCHEME, verify_ots

#: Bumped if the canonical OTS block-signing transcript shape changes.
OTS_BLOCK_TRANSCRIPT_VERSION = 1


def _ots_block_transcript_payload(block_type, previous_hash, messages, meta) -> dict:
    """Build the canonical, signature-free transcript object for an OTS block.

    This is the object an OTS signature actually authorizes (Mythos M1). It binds
    the consensus-stable, attacker-relevant parts of the block:

    * ``messages`` — the actual accepted body payload;
    * ``block_type`` — so an OTS auth can't be retyped onto another block kind;
    * ``previous_hash`` — the parent reference (consensus linkage);
    * ``auth_scheme`` and the **public key** carried in ``ots_auth`` — binds the
      signing identity;
    * any other ``meta`` fields a deployment adds.

    It deliberately EXCLUDES the self-referential ``ots_auth.signature`` and
    ``ots_auth.message`` (those carry / equal the transcript itself) and the
    mining outputs (``nonce``/``hash``) and ``timestamp``/``index`` (not signed:
    a node may legitimately reposition an identical body, and the durable replay
    ledger — not the index — is what prevents reuse).
    """
    m = dict(meta or {})
    auth = m.get("ots_auth")
    if isinstance(auth, dict):
        # Strip the self-referential fields so sign-time and verify-time agree.
        m = {**m, "ots_auth": {k: v for k, v in auth.items()
                               if k not in ("signature", "message")}}
    return {
        "wl_ots_block_transcript": OTS_BLOCK_TRANSCRIPT_VERSION,
        "auth_scheme": m.get("auth_scheme"),
        "block_type": str(block_type or ""),
        "previous_hash": previous_hash,
        "messages": list(messages or []),
        "meta": m,
    }


def canonical_ots_block_message(b: "Block") -> bytes:
    """Canonical signing transcript (preimage bytes) for an OTS block.

    Recomputed from the *received* block at verify time so the OTS signature is
    bound to the actual block body, not to an arbitrary ``meta.ots_auth.message``
    (Mythos M1). Sign-time and verify-time produce identical bytes because the
    self-referential signature/message fields are excluded.
    """
    return _canonical_json(_ots_block_transcript_payload(
        getattr(b, "block_type", ""),
        getattr(b, "previous_hash", None),
        getattr(b, "messages", []) or [],
        getattr(b, "meta", {}) or {},
    ))


def canonical_ots_block_digest(b: "Block") -> str:
    """Domain-separated hex digest of :func:`canonical_ots_block_message`.

    This hex string is the message an OTS signer signs and the verifier
    recomputes; it is what gets stored in ``meta.ots_auth.message``.
    """
    return hashlib.sha256(
        b"WL-OTS-BLOCK-TRANSCRIPT-v1\x00" + canonical_ots_block_message(b)
    ).hexdigest()


def build_ots_block_meta(public_key: dict, message, signature: dict) -> dict:
    """Canonical ``meta`` for an OTS-authenticated block.

    The auth material lives in ``meta`` so it is covered by the block hash
    (``Block.calculate_hash`` hashes a sorted-key JSON of ``meta``), binding the
    OTS signature into the block identity. ``message`` MUST be the canonical
    block-signing digest (:func:`canonical_ots_block_digest`) that ``signature``
    actually signs — a free-text message no longer authorizes a block (M1). Use
    :func:`build_signed_ots_block` to construct blocks correctly.
    """
    return {
        "auth_scheme": OTS_SCHEME,
        "ots_auth": {
            "public_key": public_key,
            "message": message,
            "signature": signature,
        },
    }


def build_signed_ots_block(secret_key: dict, public_key: dict, messages,
                           *, index: int = 1, previous_hash: str = "0" * 64,
                           difficulty: int = 1, block_type: str = "OTS",
                           extra_meta: Optional[dict] = None,
                           allow_reuse: bool = False,
                           mine: bool = True) -> "Block":
    """Build a mined OTS block whose signature is bound to its body (M1).

    The signer signs :func:`canonical_ots_block_digest` of the body (messages,
    block_type, previous_hash, auth scheme, public key, extra meta) — NOT a
    free-text message — so the accepted body is exactly what was authorized.
    """
    from wavelock.crypto.wavelock_ots import sign_ots

    messages = list(messages)
    if not all(isinstance(m, str) for m in messages):
        raise ValueError("block messages must be strings")
    if public_key.get("public_key_fingerprint") != secret_key.get("public_key_fingerprint"):
        raise ValueError("public and secret keys do not match")
    # Validate the public artifact before consuming the one-time key.
    from wavelock.crypto.wavelock_ots import load_public_key
    public_key = load_public_key(public_key)
    base_meta = dict(extra_meta or {})
    base_meta["auth_scheme"] = OTS_SCHEME
    base_meta["ots_auth"] = {"public_key": public_key}
    transcript = hashlib.sha256(
        b"WL-OTS-BLOCK-TRANSCRIPT-v1\x00" + _canonical_json(
            _ots_block_transcript_payload(block_type, previous_hash, messages, base_meta)
        )
    ).hexdigest()
    sig = sign_ots(secret_key, transcript, allow_reuse=allow_reuse)
    meta = build_ots_block_meta(public_key, transcript, sig)
    if extra_meta:
        for k, v in extra_meta.items():
            meta.setdefault(k, v)
    # A signed draft needs no mining and no second signature. Mining changes
    # only the existing transcript's excluded mining fields.
    block = Block(index=index, messages=messages, previous_hash=previous_hash,
                  difficulty=difficulty, block_type=block_type, meta=meta,
                  nonce=0, block_hash="0" * 64)
    if mine:
        block.nonce, block.hash = block.mine_block()
    else:
        block.hash = block.calculate_hash(block.nonce)
    return block


def verify_ots_block(block: Block) -> bool:
    """Pure public-only authentication; does not consume a replay identity."""
    try:
        auth = block.meta["ots_auth"]
        if block.meta.get("auth_scheme") != OTS_SCHEME:
            return False
        message = auth["message"]
        if not isinstance(message, str):
            return False
        expected = canonical_ots_block_digest(block)
        return (hmac.compare_digest(message, expected)
                and verify_ots(auth["public_key"], expected, auth["signature"]))
    except (KeyError, TypeError, ValueError, AttributeError):
        return False


def verify_block_integrity(block: Block, *, require_pow: bool = True) -> bool:
    """Validate stored header, body/Merkle binding, and declared proof of work."""
    try:
        if type(block.index) is not int or block.index < 1:
            return False
        if type(block.nonce) is not int or block.nonce < 0:
            return False
        if type(block.difficulty) is not int or not 0 <= block.difficulty <= 64:
            return False
        if not isinstance(block.messages, list) or not all(
            isinstance(m, str) for m in block.messages
        ):
            return False
        for value in (block.hash, block.previous_hash, block.merkle_root):
            if not isinstance(value, str) or len(value) != 64:
                return False
            if any(c not in "0123456789abcdef" for c in value):
                return False
        if block.merkle_root != block.calculate_merkle_root():
            return False
        if block.hash != block.calculate_hash(block.nonce):
            return False
        return not require_pow or block.hash.startswith("0" * block.difficulty)
    except (TypeError, ValueError, AttributeError):
        return False


def verify_ots_chain(blocks) -> bool:
    """Verify linkage, hashes, public signatures and OTS uniqueness, without I/O."""
    previous = "0" * 64
    keys, leaves = set(), set()
    for index, block in enumerate(blocks, 1):
        if (block.index != index or block.previous_hash != previous
                or not verify_block_integrity(block) or not verify_ots_block(block)):
            return False
        signature = block.meta["ots_auth"]["signature"]
        kid, leaf = signature["one_time_key_id"], signature["public_key_fingerprint"]
        if kid in keys or leaf in leaves:
            return False
        keys.add(kid)
        leaves.add(leaf)
        previous = block.hash
    return True
