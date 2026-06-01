"""
WaveLock-OTS export/load round-trips, determinism, and ψ-binding tests.
"""

import json

import pytest

from wavelock.crypto.wavelock_ots import (
    SCHEME,
    generate_ots_keypair,
    sign_ots,
    verify_ots,
    export_public_key,
    export_secret_key,
    load_public_key,
    load_secret_key,
    psi_commitment,
    evolve_psi_star,
    params_hash,
)


def test_export_load_public_roundtrip(tmp_path):
    kp = generate_ots_keypair()
    pub = export_public_key(kp["public_key"])
    p = tmp_path / "pub.json"
    p.write_text(json.dumps(pub))
    loaded = load_public_key(str(p))
    assert loaded["scheme"] == SCHEME
    sec = kp["secret_key"]
    sig = sign_ots(sec, "hi")
    assert verify_ots(loaded, "hi", sig)


def test_export_load_secret_roundtrip(tmp_path):
    kp = generate_ots_keypair()
    sec = export_secret_key(kp["secret_key"])
    p = tmp_path / "sec.json"
    p.write_text(json.dumps(sec))
    loaded = load_secret_key(str(p))
    sig = sign_ots(loaded, "hi")
    assert verify_ots(kp["public_key"], "hi", sig)


def test_encrypted_secret_roundtrip(tmp_path):
    kp = generate_ots_keypair()
    enc = export_secret_key(kp["secret_key"], encrypt=True, passphrase="hunter2")
    assert "seed_hex" not in enc
    assert "seed_enc" in enc
    p = tmp_path / "sec_enc.json"
    p.write_text(json.dumps(enc))
    # Wrong/no passphrase fails.
    from wavelock.crypto.wavelock_ots import WaveLockOTSError
    with pytest.raises(WaveLockOTSError):
        load_secret_key(str(p))
    loaded = load_secret_key(str(p), passphrase="hunter2")
    sig = sign_ots(loaded, "hi")
    assert verify_ots(kp["public_key"], "hi", sig)


def test_encrypted_export_has_no_cleartext_seed():
    kp = generate_ots_keypair()
    seed_hex = kp["secret_key"]["seed_hex"]
    enc = export_secret_key(kp["secret_key"], encrypt=True, passphrase="pw")
    assert seed_hex not in json.dumps(enc)


def test_deterministic_from_seed():
    seed = bytes(range(32))
    a = generate_ots_keypair(seed=seed)
    b = generate_ots_keypair(seed=seed)
    # Same ψ-commitment, params_hash, merkle_root and pk commitments.
    assert a["public_key"]["psi_commitment"] == b["public_key"]["psi_commitment"]
    assert a["public_key"]["merkle_root"] == b["public_key"]["merkle_root"]
    assert a["public_key"]["pk_commitments"] == b["public_key"]["pk_commitments"]


def test_psi_commitment_binds_to_evolution():
    """psi_commitment is H of the evolved ψ★, not of ψ₀ or the seed."""
    seed = bytes(range(1, 33))
    params = generate_ots_keypair(seed=seed)["public_key"]["params"]
    psi = evolve_psi_star(seed, params)
    assert psi_commitment(psi).hex() == \
        generate_ots_keypair(seed=seed)["public_key"]["psi_commitment"]


def test_signature_bound_to_psi_commitment():
    """A signature with a mismatched psi_commitment must not verify."""
    kp = generate_ots_keypair()
    sig = sign_ots(kp["secret_key"], "m")
    sig["psi_commitment"] = "00" * 32
    assert verify_ots(kp["public_key"], "m", sig) is False


def test_params_tamper_fails():
    """Tampering the published params (so params_hash drifts) fails closed."""
    kp = generate_ots_keypair()
    sig = sign_ots(kp["secret_key"], "m")
    pub = dict(kp["public_key"])
    pub["params"] = dict(pub["params"])
    pub["params"]["n_bits"] = 128  # mismatched
    assert verify_ots(pub, "m", sig) is False


def test_unsafe_export_secret_state_includes_psi(capsys):
    kp = generate_ots_keypair()
    out = export_secret_key(kp["secret_key"], unsafe_export_secret_state=True)
    assert "UNSAFE_psi_star_quantized_hex" in out
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
