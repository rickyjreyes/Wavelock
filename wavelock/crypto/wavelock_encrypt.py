"""
WaveLock Encrypt v1

A WaveLock-branded encryption wrapper using standard cryptographic primitives:

- X25519 ephemeral-static key exchange
- HKDF-SHA256 key derivation
- ChaCha20-Poly1305 authenticated encryption
- Canonical JSON context binding through AAD
- Optional psi_commitment / block_digest / ots_public_key_fingerprint binding

This is NOT a new raw cipher.

The "WaveLock" part is the canonical transcript/context binding:
decryption fails if the authenticated context changes.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305
from cryptography.hazmat.primitives.kdf.hkdf import HKDF


SCHEME = "WaveLock-Encrypt-v1"
VERSION = 1

KEM = "X25519-ephemeral-static"
KDF = "HKDF-SHA256"
AEAD = "ChaCha20-Poly1305"

KDF_INFO = b"WL-ENC-v1|X25519|HKDF-SHA256|ChaCha20-Poly1305"

NONCE_SIZE = 12
SALT_SIZE = 32
KEY_SIZE = 32
X25519_PUBLIC_KEY_SIZE = 32

ENVELOPE_FIELDS: Set[str] = {
    "scheme",
    "version",
    "kem",
    "kdf",
    "aead",
    "ephemeral_public_key",
    "salt",
    "nonce",
    "aad",
    "aad_sha256",
    "ciphertext",
}


class WaveLockEncryptError(Exception):
    """Base WaveLock encryption error."""


class WaveLockDecryptError(WaveLockEncryptError):
    """Raised when decryption or authentication fails."""


def b64e(data: bytes) -> str:
    if not isinstance(data, bytes):
        raise TypeError("b64e expected bytes")
    return base64.urlsafe_b64encode(data).decode("ascii")


def b64d(text: str) -> bytes:
    if not isinstance(text, str):
        raise WaveLockDecryptError("base64 field must be a string")

    try:
        return base64.urlsafe_b64decode(text.encode("ascii"))
    except (binascii.Error, UnicodeEncodeError) as exc:
        raise WaveLockDecryptError("invalid base64 encoding") from exc


def canonical_json(obj: Any) -> bytes:
    """
    Deterministic JSON encoding.

    Rules:
    - sorted keys
    - no whitespace
    - UTF-8
    - no NaN / Infinity
    """
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    digest = hashes.Hash(hashes.SHA256())
    digest.update(data)
    return digest.finalize().hex()


def make_context(
    *,
    purpose: str,
    psi_commitment: Optional[str] = None,
    block_digest: Optional[str] = None,
    ots_public_key_fingerprint: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build the authenticated WaveLock encryption context.

    This context becomes AAD.

    Decryption fails if any included field changes.
    Use this to bind encryption to:
    - protocol purpose
    - psi commitment
    - canonical block digest
    - OTS public key fingerprint
    - chain/session/application metadata
    """
    if not isinstance(purpose, str) or not purpose:
        raise ValueError("purpose must be a non-empty string")

    ctx: Dict[str, Any] = {
        "scheme": SCHEME,
        "purpose": purpose,
    }

    if psi_commitment is not None:
        ctx["psi_commitment"] = psi_commitment

    if block_digest is not None:
        ctx["block_digest"] = block_digest

    if ots_public_key_fingerprint is not None:
        ctx["ots_public_key_fingerprint"] = ots_public_key_fingerprint

    if extra:
        ctx["extra"] = extra

    # Force canonical JSON validation now.
    canonical_json(ctx)

    return ctx


@dataclass(frozen=True)
class WLPrivateKey:
    private_key: x25519.X25519PrivateKey

    @staticmethod
    def generate() -> "WLPrivateKey":
        return WLPrivateKey(x25519.X25519PrivateKey.generate())

    def public_key(self) -> "WLPublicKey":
        return WLPublicKey(self.private_key.public_key())

    def to_private_bytes_pem(self) -> bytes:
        return self.private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @staticmethod
    def from_private_bytes_pem(data: bytes) -> "WLPrivateKey":
        key = serialization.load_pem_private_key(data, password=None)

        if not isinstance(key, x25519.X25519PrivateKey):
            raise WaveLockDecryptError("private key is not an X25519 private key")

        return WLPrivateKey(key)


@dataclass(frozen=True)
class WLPublicKey:
    public_key: x25519.X25519PublicKey

    def to_public_bytes_pem(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def to_raw(self) -> bytes:
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    @staticmethod
    def from_public_bytes_pem(data: bytes) -> "WLPublicKey":
        key = serialization.load_pem_public_key(data)

        if not isinstance(key, x25519.X25519PublicKey):
            raise WaveLockEncryptError("public key is not an X25519 public key")

        return WLPublicKey(key)

    @staticmethod
    def from_raw(data: bytes) -> "WLPublicKey":
        if len(data) != X25519_PUBLIC_KEY_SIZE:
            raise WaveLockDecryptError("bad X25519 public key length")

        return WLPublicKey(x25519.X25519PublicKey.from_public_bytes(data))


def derive_key(shared_secret: bytes, salt: bytes, aad: bytes) -> bytes:
    if len(salt) != SALT_SIZE:
        raise WaveLockEncryptError("bad salt length")

    aad_hash = sha256_hex(aad).encode("ascii")

    return HKDF(
        algorithm=hashes.SHA256(),
        length=KEY_SIZE,
        salt=salt,
        info=KDF_INFO + b"|" + aad_hash,
    ).derive(shared_secret)


def check_envelope_shape(envelope: Dict[str, Any]) -> None:
    if not isinstance(envelope, dict):
        raise WaveLockDecryptError("envelope must be a dict")

    keys = set(envelope.keys())

    if keys != ENVELOPE_FIELDS:
        extra = sorted(keys - ENVELOPE_FIELDS)
        missing = sorted(ENVELOPE_FIELDS - keys)

        raise WaveLockDecryptError(
            f"non-canonical envelope fields; extra={extra}, missing={missing}"
        )


def encrypt_for_public_key(
    *,
    recipient_public_key: WLPublicKey,
    plaintext: bytes,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Encrypt using ephemeral X25519.

    Output is safe to store/send as JSON.

    The recipient decrypts with their private key.
    """
    if not isinstance(recipient_public_key, WLPublicKey):
        raise TypeError("recipient_public_key must be WLPublicKey")

    if not isinstance(plaintext, bytes):
        raise TypeError("plaintext must be bytes")

    aad = canonical_json(context)

    eph_private = x25519.X25519PrivateKey.generate()
    eph_public = eph_private.public_key()

    shared = eph_private.exchange(recipient_public_key.public_key)

    salt = os.urandom(SALT_SIZE)
    nonce = os.urandom(NONCE_SIZE)
    key = derive_key(shared, salt, aad)

    ciphertext = ChaCha20Poly1305(key).encrypt(nonce, plaintext, aad)

    eph_public_raw = eph_public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    envelope = {
        "scheme": SCHEME,
        "version": VERSION,
        "kem": KEM,
        "kdf": KDF,
        "aead": AEAD,
        "ephemeral_public_key": b64e(eph_public_raw),
        "salt": b64e(salt),
        "nonce": b64e(nonce),
        "aad": b64e(aad),
        "aad_sha256": sha256_hex(aad),
        "ciphertext": b64e(ciphertext),
    }

    # Internal sanity check before returning.
    check_envelope_shape(envelope)

    return envelope


def decrypt_with_private_key(
    *,
    recipient_private_key: WLPrivateKey,
    envelope: Dict[str, Any],
    expected_context: Dict[str, Any],
) -> bytes:
    """
    Decrypt and authenticate.

    Decryption fails if:
    - envelope has missing fields
    - envelope has extra fields
    - ciphertext changed
    - nonce/salt changed
    - context/AAD changed
    - wrong private key used
    - wrong ephemeral public key used
    - wrong scheme/version/KEM/KDF/AEAD used
    """
    try:
        if not isinstance(recipient_private_key, WLPrivateKey):
            raise WaveLockDecryptError("recipient_private_key must be WLPrivateKey")

        check_envelope_shape(envelope)

        if envelope["scheme"] != SCHEME:
            raise WaveLockDecryptError("wrong scheme")
        if envelope["version"] != VERSION:
            raise WaveLockDecryptError("wrong version")
        if envelope["kem"] != KEM:
            raise WaveLockDecryptError("wrong KEM")
        if envelope["kdf"] != KDF:
            raise WaveLockDecryptError("wrong KDF")
        if envelope["aead"] != AEAD:
            raise WaveLockDecryptError("wrong AEAD")

        expected_aad = canonical_json(expected_context)
        encoded_aad = b64d(envelope["aad"])

        if encoded_aad != expected_aad:
            raise WaveLockDecryptError("AAD/context mismatch")

        if envelope["aad_sha256"] != sha256_hex(expected_aad):
            raise WaveLockDecryptError("AAD hash mismatch")

        eph_pub_raw = b64d(envelope["ephemeral_public_key"])
        salt = b64d(envelope["salt"])
        nonce = b64d(envelope["nonce"])
        ciphertext = b64d(envelope["ciphertext"])

        if len(eph_pub_raw) != X25519_PUBLIC_KEY_SIZE:
            raise WaveLockDecryptError("bad ephemeral public key length")
        if len(salt) != SALT_SIZE:
            raise WaveLockDecryptError("bad salt length")
        if len(nonce) != NONCE_SIZE:
            raise WaveLockDecryptError("bad nonce length")

        eph_pub = WLPublicKey.from_raw(eph_pub_raw)

        shared = recipient_private_key.private_key.exchange(eph_pub.public_key)
        key = derive_key(shared, salt, expected_aad)

        return ChaCha20Poly1305(key).decrypt(nonce, ciphertext, expected_aad)

    except InvalidTag as exc:
        raise WaveLockDecryptError("authentication failed") from exc
    except KeyError as exc:
        raise WaveLockDecryptError(f"missing envelope field: {exc}") from exc
    except WaveLockDecryptError:
        raise
    except Exception as exc:
        # Never leak the underlying error detail (could carry key material).
        raise WaveLockDecryptError("decrypt failed") from None


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    check_envelope_shape(obj)
    path.write_text(
        json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )


def load_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(obj, dict):
        raise WaveLockDecryptError("JSON envelope must be an object")

    return obj


def cmd_keygen(args: argparse.Namespace) -> None:
    private = WLPrivateKey.generate()
    public = private.public_key()

    private_path = Path(args.private)
    public_path = Path(args.public)

    if private_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite private key: {private_path}")

    if public_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite public key: {public_path}")

    private_path.write_bytes(private.to_private_bytes_pem())
    public_path.write_bytes(public.to_public_bytes_pem())

    print(f"[OK] wrote private key: {private_path}")
    print(f"[OK] wrote public key:  {public_path}")


def cmd_encrypt(args: argparse.Namespace) -> None:
    public = WLPublicKey.from_public_bytes_pem(Path(args.public).read_bytes())
    plaintext = Path(args.input).read_bytes()

    ctx = make_context(
        purpose=args.purpose,
        psi_commitment=args.psi_commitment,
        block_digest=args.block_digest,
        ots_public_key_fingerprint=args.ots_public_key_fingerprint,
        extra={"label": args.label} if args.label else None,
    )

    envelope = encrypt_for_public_key(
        recipient_public_key=public,
        plaintext=plaintext,
        context=ctx,
    )

    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite output file: {output_path}")

    save_json(output_path, envelope)

    print(f"[OK] encrypted {args.input} -> {output_path}")
    print(f"[AAD] {envelope['aad_sha256']}")


def cmd_decrypt(args: argparse.Namespace) -> None:
    private = WLPrivateKey.from_private_bytes_pem(Path(args.private).read_bytes())
    envelope = load_json(Path(args.input))

    ctx = make_context(
        purpose=args.purpose,
        psi_commitment=args.psi_commitment,
        block_digest=args.block_digest,
        ots_public_key_fingerprint=args.ots_public_key_fingerprint,
        extra={"label": args.label} if args.label else None,
    )

    plaintext = decrypt_with_private_key(
        recipient_private_key=private,
        envelope=envelope,
        expected_context=ctx,
    )

    output_path = Path(args.output)

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite output file: {output_path}")

    output_path.write_bytes(plaintext)

    print(f"[OK] decrypted {args.input} -> {output_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("wavelock-encrypt")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("keygen")
    p.add_argument("--private", default="wlenc_private.pem")
    p.add_argument("--public", default="wlenc_public.pem")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_keygen)

    p = sub.add_parser("encrypt")
    p.add_argument("--public", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--purpose", required=True)
    p.add_argument("--psi-commitment")
    p.add_argument("--block-digest")
    p.add_argument("--ots-public-key-fingerprint")
    p.add_argument("--label")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_encrypt)

    p = sub.add_parser("decrypt")
    p.add_argument("--private", required=True)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--purpose", required=True)
    p.add_argument("--psi-commitment")
    p.add_argument("--block-digest")
    p.add_argument("--ots-public-key-fingerprint")
    p.add_argument("--label")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_decrypt)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
