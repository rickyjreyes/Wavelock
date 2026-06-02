"""
Tests + red-team harness for WaveLock-Encrypt v1.

WaveLock-Encrypt is NOT a new raw cipher. It is a thin, context-binding
wrapper over standard primitives:

    X25519 (ephemeral-static)  +  HKDF-SHA256  +  ChaCha20-Poly1305

These tests prove the wrapper fails closed under every mutation we could
think of: tampered envelope fields, mismatched context, swapped keys,
malformed encodings, and hostile JSON values.
"""

from __future__ import annotations

import copy
import subprocess
import sys

import pytest

from wavelock.crypto.wavelock_encrypt import (
    SCHEME,
    VERSION,
    KEM,
    KDF,
    AEAD,
    ENVELOPE_FIELDS,
    WaveLockDecryptError,
    WLPrivateKey,
    WLPublicKey,
    make_context,
    encrypt_for_public_key,
    decrypt_with_private_key,
    canonical_json,
    b64e,
    b64d,
)


# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------

PLAINTEXT = b"the curvature is locked; the wave is sealed."


@pytest.fixture
def recipient():
    priv = WLPrivateKey.generate()
    return priv, priv.public_key()


@pytest.fixture
def full_context():
    return make_context(
        purpose="unit-test/transport",
        psi_commitment="psi-commit-abc123",
        block_digest="block-digest-deadbeef",
        ots_public_key_fingerprint="ots-fp-cafe",
        extra={"label": "demo", "n": 7},
    )


@pytest.fixture
def envelope(recipient, full_context):
    _, pub = recipient
    return encrypt_for_public_key(
        recipient_public_key=pub,
        plaintext=PLAINTEXT,
        context=full_context,
    )


def _decrypt(priv, env, ctx):
    return decrypt_with_private_key(
        recipient_private_key=priv,
        envelope=env,
        expected_context=ctx,
    )


# --------------------------------------------------------------------------
# 1. Roundtrip
# --------------------------------------------------------------------------

def test_roundtrip_succeeds(recipient, full_context, envelope):
    priv, _ = recipient
    assert _decrypt(priv, envelope, full_context) == PLAINTEXT


def test_roundtrip_minimal_context(recipient):
    priv, pub = recipient
    ctx = make_context(purpose="minimal")
    env = encrypt_for_public_key(
        recipient_public_key=pub, plaintext=b"x", context=ctx
    )
    assert _decrypt(priv, env, ctx) == b"x"


def test_roundtrip_empty_plaintext(recipient):
    priv, pub = recipient
    ctx = make_context(purpose="empty")
    env = encrypt_for_public_key(
        recipient_public_key=pub, plaintext=b"", context=ctx
    )
    assert _decrypt(priv, env, ctx) == b""


def test_every_encryption_uses_fresh_ephemeral_key(recipient, full_context):
    _, pub = recipient
    e1 = encrypt_for_public_key(
        recipient_public_key=pub, plaintext=PLAINTEXT, context=full_context
    )
    e2 = encrypt_for_public_key(
        recipient_public_key=pub, plaintext=PLAINTEXT, context=full_context
    )
    assert e1["ephemeral_public_key"] != e2["ephemeral_public_key"]
    assert e1["nonce"] != e2["nonce"]
    assert e1["salt"] != e2["salt"]
    assert e1["ciphertext"] != e2["ciphertext"]


# --------------------------------------------------------------------------
# 2-5. Context field mismatches reject
# --------------------------------------------------------------------------

def test_wrong_purpose_rejects(recipient, envelope):
    priv, _ = recipient
    bad = make_context(
        purpose="DIFFERENT",
        psi_commitment="psi-commit-abc123",
        block_digest="block-digest-deadbeef",
        ots_public_key_fingerprint="ots-fp-cafe",
        extra={"label": "demo", "n": 7},
    )
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, envelope, bad)


def test_wrong_psi_commitment_rejects(recipient, envelope):
    priv, _ = recipient
    bad = make_context(
        purpose="unit-test/transport",
        psi_commitment="WRONG",
        block_digest="block-digest-deadbeef",
        ots_public_key_fingerprint="ots-fp-cafe",
        extra={"label": "demo", "n": 7},
    )
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, envelope, bad)


def test_wrong_block_digest_rejects(recipient, envelope):
    priv, _ = recipient
    bad = make_context(
        purpose="unit-test/transport",
        psi_commitment="psi-commit-abc123",
        block_digest="WRONG",
        ots_public_key_fingerprint="ots-fp-cafe",
        extra={"label": "demo", "n": 7},
    )
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, envelope, bad)


def test_wrong_ots_fingerprint_rejects(recipient, envelope):
    priv, _ = recipient
    bad = make_context(
        purpose="unit-test/transport",
        psi_commitment="psi-commit-abc123",
        block_digest="block-digest-deadbeef",
        ots_public_key_fingerprint="WRONG",
        extra={"label": "demo", "n": 7},
    )
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, envelope, bad)


def test_dropped_optional_context_field_rejects(recipient, envelope):
    # Omitting an optional field that was present at encrypt time changes the
    # canonical AAD and must fail closed.
    priv, _ = recipient
    bad = make_context(
        purpose="unit-test/transport",
        psi_commitment="psi-commit-abc123",
        block_digest="block-digest-deadbeef",
        # ots fingerprint omitted
        extra={"label": "demo", "n": 7},
    )
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, envelope, bad)


def test_wrong_extra_rejects(recipient, envelope):
    priv, _ = recipient
    bad = make_context(
        purpose="unit-test/transport",
        psi_commitment="psi-commit-abc123",
        block_digest="block-digest-deadbeef",
        ots_public_key_fingerprint="ots-fp-cafe",
        extra={"label": "demo", "n": 8},  # n changed
    )
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, envelope, bad)


# --------------------------------------------------------------------------
# 6-10. Tampered envelope bytes reject
# --------------------------------------------------------------------------

def _flip_b64(value: str) -> str:
    """Return a valid-base64 string that decodes to mutated bytes."""
    raw = bytearray(b64d(value))
    raw[0] ^= 0x01
    return b64e(bytes(raw))


def test_tampered_ciphertext_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["ciphertext"] = _flip_b64(env["ciphertext"])
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_tampered_nonce_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["nonce"] = _flip_b64(env["nonce"])
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_tampered_salt_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["salt"] = _flip_b64(env["salt"])
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_tampered_ephemeral_public_key_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["ephemeral_public_key"] = _flip_b64(env["ephemeral_public_key"])
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_tampered_aad_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    # Replace AAD with a different (but well-formed) canonical context.
    other = canonical_json(make_context(purpose="attacker"))
    env["aad"] = b64e(other)
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


# --------------------------------------------------------------------------
# 11. Tampered aad_sha256 (field is retained)
# --------------------------------------------------------------------------

def test_aad_sha256_field_is_present():
    assert "aad_sha256" in ENVELOPE_FIELDS


def test_tampered_aad_sha256_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["aad_sha256"] = "0" * 64
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


# --------------------------------------------------------------------------
# 12-16. Wrong algorithm identifiers reject
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("scheme", "WaveLock-Encrypt-v2"),
        ("version", 2),
        ("kem", "RSA-OAEP"),
        ("kdf", "HKDF-SHA512"),
        ("aead", "AES-256-GCM"),
    ],
)
def test_wrong_algorithm_identifier_rejects(
    recipient, full_context, envelope, field, bad_value
):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env[field] = bad_value
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


# --------------------------------------------------------------------------
# 17-19. Envelope shape / encoding
# --------------------------------------------------------------------------

@pytest.mark.parametrize("field", sorted(ENVELOPE_FIELDS))
def test_missing_envelope_field_rejects(recipient, full_context, envelope, field):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    del env[field]
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_extra_envelope_field_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["evil"] = "surprise"
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_invalid_base64_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["ciphertext"] = "!!!not base64!!!"
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_non_string_base64_field_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["nonce"] = 12345
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


# --------------------------------------------------------------------------
# 20. Wrong recipient private key
# --------------------------------------------------------------------------

def test_wrong_recipient_private_key_rejects(full_context, envelope):
    attacker = WLPrivateKey.generate()
    with pytest.raises(WaveLockDecryptError):
        _decrypt(attacker, envelope, full_context)


# --------------------------------------------------------------------------
# 21. CLI smoke test: keygen -> encrypt -> decrypt
# --------------------------------------------------------------------------

def test_cli_keygen_encrypt_decrypt(tmp_path):
    priv = tmp_path / "priv.pem"
    pub = tmp_path / "pub.pem"
    msg = tmp_path / "msg.txt"
    env = tmp_path / "env.json"
    out = tmp_path / "out.txt"

    secret = b"top secret payload \x01\x02\x03"
    msg.write_bytes(secret)

    base = [sys.executable, "-m", "wavelock.crypto.wavelock_encrypt"]

    r = subprocess.run(
        base + ["keygen", "--private", str(priv), "--public", str(pub)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert priv.exists() and pub.exists()

    r = subprocess.run(
        base + [
            "encrypt",
            "--public", str(pub),
            "--input", str(msg),
            "--output", str(env),
            "--purpose", "cli-smoke",
            "--psi-commitment", "psi-xyz",
            "--label", "cli",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert env.exists()

    r = subprocess.run(
        base + [
            "decrypt",
            "--private", str(priv),
            "--input", str(env),
            "--output", str(out),
            "--purpose", "cli-smoke",
            "--psi-commitment", "psi-xyz",
            "--label", "cli",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    assert out.read_bytes() == secret


def test_cli_decrypt_wrong_purpose_fails(tmp_path):
    priv = tmp_path / "priv.pem"
    pub = tmp_path / "pub.pem"
    msg = tmp_path / "msg.txt"
    env = tmp_path / "env.json"
    out = tmp_path / "out.txt"
    msg.write_bytes(b"hello")

    base = [sys.executable, "-m", "wavelock.crypto.wavelock_encrypt"]
    subprocess.run(
        base + ["keygen", "--private", str(priv), "--public", str(pub)],
        capture_output=True, text=True, check=True,
    )
    subprocess.run(
        base + [
            "encrypt", "--public", str(pub), "--input", str(msg),
            "--output", str(env), "--purpose", "real",
        ],
        capture_output=True, text=True, check=True,
    )
    r = subprocess.run(
        base + [
            "decrypt", "--private", str(priv), "--input", str(env),
            "--output", str(out), "--purpose", "FAKE",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert not out.exists()


def test_cli_encrypt_overwrite_protection(tmp_path):
    priv = tmp_path / "priv.pem"
    pub = tmp_path / "pub.pem"
    msg = tmp_path / "msg.txt"
    env = tmp_path / "env.json"
    msg.write_bytes(b"hello")
    env.write_text("PREEXISTING")

    base = [sys.executable, "-m", "wavelock.crypto.wavelock_encrypt"]
    subprocess.run(
        base + ["keygen", "--private", str(priv), "--public", str(pub)],
        capture_output=True, text=True, check=True,
    )
    r = subprocess.run(
        base + [
            "encrypt", "--public", str(pub), "--input", str(msg),
            "--output", str(env), "--purpose", "real",
        ],
        capture_output=True, text=True,
    )
    assert r.returncode != 0
    assert env.read_text() == "PREEXISTING"  # untouched


# --------------------------------------------------------------------------
# Red-team: hostile JSON values and envelope shapes
# --------------------------------------------------------------------------

def test_non_dict_envelope_rejects(recipient, full_context):
    priv, _ = recipient
    for junk in ([], "string", 42, None):
        with pytest.raises(WaveLockDecryptError):
            _decrypt(priv, junk, full_context)


def test_non_bytes_plaintext_rejects(recipient, full_context):
    _, pub = recipient
    for junk in ("string", 42, None, ["bytes"]):
        with pytest.raises(TypeError):
            encrypt_for_public_key(
                recipient_public_key=pub, plaintext=junk, context=full_context
            )


def test_nan_infinity_in_context_rejected_at_build():
    # allow_nan=False must reject non-finite floats during canonicalization.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            make_context(purpose="x", extra={"v": bad})


def test_empty_purpose_rejected():
    with pytest.raises(ValueError):
        make_context(purpose="")


def test_unicode_and_huge_int_context_roundtrip(recipient):
    priv, pub = recipient
    ctx = make_context(
        purpose="unicode-✨-\U0001f680",
        extra={"big": 10 ** 60, "emoji": "\U0001f512", "nul": "a\x00b"},
    )
    env = encrypt_for_public_key(
        recipient_public_key=pub, plaintext=b"payload", context=ctx
    )
    assert _decrypt(priv, env, ctx) == b"payload"
    # A different huge int must fail.
    bad = make_context(
        purpose="unicode-✨-\U0001f680",
        extra={"big": 10 ** 60 + 1, "emoji": "\U0001f512", "nul": "a\x00b"},
    )
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, bad)


def test_canonical_json_is_key_order_independent():
    a = {"b": 1, "a": 2, "z": {"y": 1, "x": 2}}
    b = {"a": 2, "z": {"x": 2, "y": 1}, "b": 1}
    assert canonical_json(a) == canonical_json(b)


def test_replay_under_different_purpose_rejects(recipient):
    # Encrypt for purpose A, attacker re-presents the same envelope expecting B.
    priv, pub = recipient
    ctx_a = make_context(purpose="purpose-A")
    env = encrypt_for_public_key(
        recipient_public_key=pub, plaintext=PLAINTEXT, context=ctx_a
    )
    ctx_b = make_context(purpose="purpose-B")
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, ctx_b)
    # Sanity: still works under the correct purpose.
    assert _decrypt(priv, env, ctx_a) == PLAINTEXT


def test_very_large_field_rejects(recipient, full_context, envelope):
    # A multi-megabyte garbage ciphertext must fail authentication, not crash.
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["ciphertext"] = b64e(b"\x00" * (4 * 1024 * 1024))
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_wrong_length_nonce_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["nonce"] = b64e(b"\x00" * 8)  # too short
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_wrong_length_salt_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["salt"] = b64e(b"\x00" * 16)  # too short
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_wrong_length_ephemeral_key_rejects(recipient, full_context, envelope):
    priv, _ = recipient
    env = copy.deepcopy(envelope)
    env["ephemeral_public_key"] = b64e(b"\x00" * 16)  # not 32
    with pytest.raises(WaveLockDecryptError):
        _decrypt(priv, env, full_context)


def test_error_messages_do_not_leak_secrets(recipient, full_context, envelope):
    # No decrypt error should echo key/secret material.
    priv, _ = recipient
    attacker = WLPrivateKey.generate()
    secret_pem = attacker.to_private_bytes_pem().decode()
    env = copy.deepcopy(envelope)
    env["ciphertext"] = _flip_b64(env["ciphertext"])
    try:
        _decrypt(attacker, env, full_context)
    except WaveLockDecryptError as exc:
        msg = str(exc)
        assert "BEGIN" not in msg
        assert secret_pem not in msg
