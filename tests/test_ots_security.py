"""
WaveLock-OTS security + correctness tests.

These cover the hard security requirements for the new asymmetric construction:
valid signatures verify, tampered/wrong inputs fail closed, the public key
leaks no ψ★/seed, keys are one-time, tiny seeds are rejected, and the legacy
forge-from-snapshot attack does NOT work against WaveLock-OTS.
"""

import copy
import json

import pytest

from wavelock.crypto.wavelock_ots import (
    SCHEME,
    OTSKeyReuseError,
    InsufficientEntropyError,
    generate_ots_keypair,
    sign_ots,
    verify_ots,
    export_public_key,
    export_secret_key,
    load_public_key,
    load_secret_key,
    psi_commitment,
    evolve_psi_star,
)
from attacks.forge_from_snapshot import attempt_forge_ots_from_public


@pytest.fixture
def keypair():
    return generate_ots_keypair(entropy_bits=256)


def test_ots_valid_signature_verifies(keypair):
    pub, sec = keypair["public_key"], keypair["secret_key"]
    sig = sign_ots(sec, "transfer 5 to alice")
    assert verify_ots(pub, "transfer 5 to alice", sig) is True


def test_ots_wrong_message_fails(keypair):
    pub, sec = keypair["public_key"], keypair["secret_key"]
    sig = sign_ots(sec, "transfer 5 to alice")
    assert verify_ots(pub, "transfer 5000000 to mallory", sig) is False


def test_ots_modified_signature_fails(keypair):
    pub, sec = keypair["public_key"], keypair["secret_key"]
    sig = sign_ots(sec, "hello")
    # Flip one nibble in a revealed slice.
    tampered = copy.deepcopy(sig)
    r0 = tampered["revealed_slices"][0]
    flipped = ("0" if r0[0] != "0" else "1") + r0[1:]
    tampered["revealed_slices"][0] = flipped
    assert verify_ots(pub, "hello", tampered) is False


def test_ots_truncated_signature_fails(keypair):
    pub, sec = keypair["public_key"], keypair["secret_key"]
    sig = sign_ots(sec, "hello")
    sig2 = copy.deepcopy(sig)
    sig2["revealed_slices"] = sig2["revealed_slices"][:-1]  # wrong count
    assert verify_ots(pub, "hello", sig2) is False


def test_ots_public_key_cannot_forge(keypair):
    """Holding only the public key must not let you forge any message."""
    pub = keypair["public_key"]
    for target in ["x", "give me money", "transfer 1 to attacker"]:
        forged = attempt_forge_ots_from_public(pub, target)
        assert verify_ots(pub, target, forged) is False


def test_ots_key_reuse_rejected(keypair):
    sec = keypair["secret_key"]
    sign_ots(sec, "first")
    with pytest.raises(OTSKeyReuseError):
        sign_ots(sec, "second")


def test_ots_key_reuse_override_allows(keypair):
    """allow_reuse exists strictly for tests and is honored."""
    sec = keypair["secret_key"]
    sign_ots(sec, "first")
    sig2 = sign_ots(sec, "second", allow_reuse=True)
    assert verify_ots(keypair["public_key"], "second", sig2) is True


def test_no_full_psi_star_in_public_key(keypair):
    pub = keypair["public_key"]
    blob = json.dumps(pub)
    assert "psi_star" not in pub
    assert "psi_0" not in pub
    assert "psi_star" not in blob.lower().replace("psi_commitment", "")
    # Only the *commitment* (a hash) may reference psi.
    assert "psi_commitment" in pub
    # The ψ-commitment is 32 bytes hex; not a serialized array.
    assert len(pub["psi_commitment"]) == 64


def test_no_seed_in_public_key(keypair):
    pub = keypair["public_key"]
    blob = json.dumps(pub).lower()
    assert "seed" not in pub
    assert "seed_hex" not in pub
    assert "seed" not in blob


def test_export_public_key_rejects_secret_leak(keypair):
    pub = dict(keypair["public_key"])
    pub["seed_hex"] = keypair["secret_key"]["seed_hex"]
    from wavelock.crypto.wavelock_ots import WaveLockOTSError
    with pytest.raises(WaveLockOTSError):
        export_public_key(pub)


def test_load_public_key_rejects_secret_leak(tmp_path, keypair):
    pub = dict(keypair["public_key"])
    pub["psi_star"] = [[1.0, 2.0], [3.0, 4.0]]
    p = tmp_path / "bad_public.json"
    p.write_text(json.dumps(pub))
    from wavelock.crypto.wavelock_ots import WaveLockOTSError
    with pytest.raises(WaveLockOTSError):
        load_public_key(str(p))


def test_small_seed_rejected():
    with pytest.raises(InsufficientEntropyError):
        generate_ots_keypair(entropy_bits=64)
    with pytest.raises(InsufficientEntropyError):
        generate_ots_keypair(seed=b"\x00" * 4)  # 32-bit
    with pytest.raises(InsufficientEntropyError):
        generate_ots_keypair(seed=b"\x00" * 16)  # constant seed, zero entropy


def test_old_forge_from_snapshot_attack_fails_against_ots(keypair):
    """The exact attack that breaks legacy SIGv2 must fail against OTS."""
    pub = keypair["public_key"]
    forged = attempt_forge_ots_from_public(pub, "TRANSFER 1000000 TO ATTACKER")
    assert verify_ots(pub, "TRANSFER 1000000 TO ATTACKER", forged) is False


def test_verify_is_deterministic_and_pure(keypair):
    pub, sec = keypair["public_key"], keypair["secret_key"]
    sig = sign_ots(sec, "msg")
    assert verify_ots(pub, "msg", sig)
    assert verify_ots(pub, "msg", sig)  # repeatable, no state mutation


def test_signature_reveals_only_the_selected_half(keypair):
    """A signature reveals exactly the message-selected half per bit position.

    For each bit i the revealed slice must hash to pk[i][bit_i] and must NOT
    hash to pk[i][1-bit_i]. The verifier therefore never learns the unrevealed
    half, which is precisely why it cannot forge a different message.
    """
    pub, sec = keypair["public_key"], keypair["secret_key"]
    p_hash = bytes.fromhex(pub["params_hash"])
    sig = sign_ots(sec, "leak check")

    from wavelock.crypto.wavelock_ots import (
        _public_slice, _message_digest, _digest_bits,
    )

    digest = _message_digest("leak check", p_hash)
    bits = _digest_bits(digest)
    pk = pub["pk_commitments"]

    for i, bit in enumerate(bits):
        sk = bytes.fromhex(sig["revealed_slices"][i])
        # Revealed half matches the committed selected half.
        assert _public_slice(p_hash, i, bit, sk).hex() == pk[i][bit]
        # Revealed half does NOT match the unrevealed (other) half.
        other = 1 - bit
        assert _public_slice(p_hash, i, other, sk).hex() != pk[i][other]
