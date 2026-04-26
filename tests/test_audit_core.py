"""
Core audit tests for WaveLock invariants.

Tests:
  - Commitment generation / verification consistency
  - Signature generation / verification consistency (sign/verify duality)
  - Block save/load roundtrip
  - Ledger verification
  - PDE determinism for fixed seed
  - Attack battery reproducibility (soliton seed=12 data match)
  - Cross-implementation consistency (WaveLock.py vs Wavelock_numpy.py)
"""

import os
import tempfile
import numpy as np
import pytest

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    _serialize_commitment_v2,
    _curvature_functional,
    _canonical_json,
    _kernel_hash,
    _to_numpy,
    alpha, beta, theta, epsilon, delta,
    _dt, _steps, _damping,
    laplacian,
)
from wavelock.chain.Wavelock_numpy import CurvatureKeyPairV3
from wavelock.chain.hash_families import (
    HashFamily, DualHash, hash_hex, parse_commitment, format_commitment_v3,
)
from wavelock.chain.Block import Block
from wavelock.chain.CurvaChain import CurvaChain
from wavelock.chain.chain_utils import save_block_to_disk, load_all_blocks, LEDGER_DIR


# ============================================================
# 1. Commitment generation / verification consistency
# ============================================================

class TestCommitmentConsistency:
    def test_commitment_roundtrip(self):
        """Commitment verification must pass immediately after generation."""
        kp = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        p_ok, s_ok = kp.verify_commitment()
        assert p_ok, "Primary commitment hash mismatch"
        assert s_ok, "Secondary commitment hash mismatch"

    def test_commitment_format_v3(self):
        """Commitment string must be schema:primary:secondary."""
        kp = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        schema, ph, sh = parse_commitment(kp.commitment)
        assert schema == "WLv2"
        assert len(ph) == 64, "Primary hash should be 64 hex chars"
        assert sh is not None, "Secondary hash should be present"
        assert len(sh) == 64, "Secondary hash should be 64 hex chars"

    def test_commitment_deterministic(self):
        """Same seed must produce identical commitments."""
        kp1 = CurvatureKeyPair(n=4, seed=99, test_mode=True)
        kp2 = CurvatureKeyPair(n=4, seed=99, test_mode=True)
        assert kp1.commitment == kp2.commitment

    def test_commitment_different_seeds(self):
        """Different seeds must produce different commitments."""
        kp1 = CurvatureKeyPair(n=4, seed=1, test_mode=True)
        kp2 = CurvatureKeyPair(n=4, seed=2, test_mode=True)
        assert kp1.commitment != kp2.commitment

    def test_commitment_v2_backward_compat(self):
        """commitment_v2 (single-hash) must be a valid prefix of the full commitment."""
        kp = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        assert kp.commitment.startswith(kp.commitment_v2)

    def test_dual_hash_from_raw_bytes(self):
        """DualHash.from_data and verify must be inverses."""
        raw = b"test data for dual hash"
        dh = DualHash.from_data(raw)
        p_ok, s_ok = dh.verify(raw)
        assert p_ok and s_ok
        # Tampered data must fail
        p_ok2, s_ok2 = dh.verify(raw + b"X")
        assert not p_ok2 and not s_ok2


# ============================================================
# 2. Signature generation / verification consistency
# ============================================================

class TestSignatureConsistency:
    def test_sign_verify_roundtrip(self):
        """sign() output must be accepted by verify()."""
        kp = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        msg = "test message"
        sig = kp.sign(msg)
        assert kp.verify(msg, sig), "Signature verification failed"

    def test_sign_dual_verify_strict(self):
        """sign_dual() output must pass verify_strict()."""
        kp = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        msg = "dual test"
        sig_p, sig_s = kp.sign_dual(msg)
        assert kp.verify_strict(msg, sig_p, sig_s)

    def test_verify_rejects_tampered_signature(self):
        """Tampered signature must be rejected."""
        kp = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        msg = "test"
        sig = kp.sign(msg)
        tampered = sig[:-4] + "XXXX"
        assert not kp.verify(msg, tampered)

    def test_verify_rejects_wrong_message(self):
        """Signature for one message must not verify a different message."""
        kp = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        sig = kp.sign("message A")
        assert not kp.verify("message B", sig)

    def test_verify_rejects_different_keypair(self):
        """Signature from one keypair must not verify with another."""
        kp1 = CurvatureKeyPair(n=4, seed=1, test_mode=True)
        kp2 = CurvatureKeyPair(n=4, seed=2, test_mode=True)
        msg = "cross-key test"
        sig = kp1.sign(msg)
        assert not kp2.verify(msg, sig)

    def test_secondary_signature_accepted(self):
        """verify() must accept secondary-family signatures (survivability)."""
        kp = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        msg = "survivability test"
        _, sig_s = kp.sign_dual(msg)
        assert kp.verify(msg, sig_s), "Secondary signature should be accepted"

    def test_sign_deterministic(self):
        """Same keypair + message must produce same signature."""
        kp = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        s1 = kp.sign("hello")
        s2 = kp.sign("hello")
        assert s1 == s2


# ============================================================
# 3. Block save/load roundtrip
# ============================================================

class TestBlockRoundtrip:
    def test_to_dict_from_dict(self):
        """Block.to_dict() -> Block.from_dict() must preserve all fields."""
        b = Block(
            index=5,
            messages=["message: hello", "signature: abc123", "commitment: WLv2:xyz"],
            previous_hash="0" * 64,
            difficulty=3,
            block_type="GENERIC",
            meta={"foo": "bar"},
        )
        d = b.to_dict()
        b2 = Block.from_dict(d)

        assert b2.index == b.index
        assert b2.messages == b.messages
        assert b2.previous_hash == b.previous_hash
        assert b2.difficulty == b.difficulty
        assert b2.nonce == b.nonce
        assert b2.hash == b.hash
        assert b2.merkle_root == b.merkle_root
        assert b2.block_type == b.block_type
        assert b2.meta == b.meta
        assert b2.timestamp == b.timestamp

    def test_hash_stability(self):
        """Rehydrated block must recalculate to same hash."""
        b = Block(
            index=1,
            messages=["test"],
            previous_hash="0" * 64,
            difficulty=2,
        )
        d = b.to_dict()
        b2 = Block.from_dict(d)
        assert b2.calculate_hash(b2.nonce) == b2.hash

    def test_merkle_root_stability(self):
        """Merkle root must be same for same messages."""
        msgs = ["a", "b", "c"]
        b1 = Block(index=0, messages=msgs, previous_hash="0" * 64, difficulty=1)
        b2 = Block(index=0, messages=msgs, previous_hash="0" * 64, difficulty=1)
        assert b1.merkle_root == b2.merkle_root

    def test_ledger_roundtrip(self):
        """Block saved to ledger and loaded back must be identical."""
        with tempfile.TemporaryDirectory() as tmpdir:
            import wavelock.chain.chain_utils as cu
            old_dir = cu.LEDGER_DIR
            old_file = cu.LEDGER_FILE
            cu.LEDGER_DIR = type(old_dir)(tmpdir) / "ledger"
            cu.LEDGER_DIR.mkdir()
            cu.LEDGER_FILE = cu.LEDGER_DIR / "blk00000.jsonl"
            try:
                b = Block(
                    index=1,
                    messages=["message: test", "signature: sig", "commitment: com"],
                    previous_hash="0" * 64,
                    difficulty=2,
                )
                save_block_to_disk(b)
                loaded = load_all_blocks()
                assert len(loaded) == 1
                lb = loaded[0]
                assert lb.index == b.index
                assert lb.hash == b.hash
                assert lb.messages == b.messages
                assert lb.nonce == b.nonce
            finally:
                cu.LEDGER_DIR = old_dir
                cu.LEDGER_FILE = old_file


# ============================================================
# 4. Ledger verification
# ============================================================

class TestLedgerVerification:
    def test_curvachain_valid(self):
        """A freshly built CurvaChain must validate."""
        chain = CurvaChain(difficulty=2)
        chain.add_block(["message: hello", "signature: abc", "commitment: WLv2:xyz"])
        chain.add_block(["message: world", "signature: def", "commitment: WLv2:xyz"])
        assert chain.is_chain_valid()

    def test_curvachain_tamper_hash_detection(self):
        """Tampering a block's hash must invalidate the chain."""
        chain = CurvaChain(difficulty=2)
        chain.add_block(["test"])
        chain.chain[1].hash = "0" * 64
        assert not chain.is_chain_valid()

    def test_curvachain_tamper_linkage_detection(self):
        """Breaking hash linkage must invalidate the chain."""
        chain = CurvaChain(difficulty=2)
        chain.add_block(["block 1"])
        chain.add_block(["block 2"])
        chain.chain[2].previous_hash = "f" * 64
        assert not chain.is_chain_valid()

    def test_merkle_root_detects_message_tamper(self):
        """Merkle root recalculation must detect message tampering."""
        from wavelock.chain.chain_utils import verify_merkle_root
        b = Block(index=1, messages=["original"], previous_hash="0" * 64, difficulty=2)
        assert verify_merkle_root(b)
        b.messages = ["tampered"]
        assert not verify_merkle_root(b)


# ============================================================
# 5. PDE determinism for fixed seed
# ============================================================

class TestPDEDeterminism:
    def test_curvature_keypair_deterministic(self):
        """CurvatureKeyPair with same seed must produce identical psi*."""
        kp1 = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        kp2 = CurvatureKeyPair(n=4, seed=42, test_mode=True)
        np.testing.assert_array_equal(
            _to_numpy(kp1.psi_star),
            _to_numpy(kp2.psi_star),
        )

    def test_curvature_keypair_v3_deterministic(self):
        """CurvatureKeyPairV3 (numpy) with same seed must be deterministic."""
        kp1 = CurvatureKeyPairV3(n=4, seed=42)
        kp2 = CurvatureKeyPairV3(n=4, seed=42)
        np.testing.assert_array_equal(kp1.psi_star, kp2.psi_star)
        assert kp1.commitment == kp2.commitment

    def test_curvature_functional_deterministic(self):
        """Curvature functional on same input must produce same output."""
        np.random.seed(7)
        psi = np.random.rand(4, 4).astype(np.float64)
        r1 = _curvature_functional(psi)
        r2 = _curvature_functional(psi)
        assert r1 == r2

    def test_different_seeds_produce_different_psi(self):
        """Different seeds must produce different psi*."""
        kp1 = CurvatureKeyPair(n=4, seed=1, test_mode=True)
        kp2 = CurvatureKeyPair(n=4, seed=2, test_mode=True)
        assert not np.array_equal(
            _to_numpy(kp1.psi_star),
            _to_numpy(kp2.psi_star),
        )


# ============================================================
# 6. Soliton / attack battery data reproducibility
# ============================================================

class TestSolitonReproducibility:
    """Verify that the soliton.py PDE generator reproduces the provided data files."""

    def _run_soliton_pde(self, seed=12, N=32, T=50):
        """Reproduce the exact PDE from soliton.py."""
        rng = np.random.default_rng(seed)
        psi = rng.standard_normal((N, N), dtype=np.float64)
        _alpha = 1.50
        _beta = 2.6e-3
        _theta = 1.0e-5
        _eps = 1.0e-12
        _delta = 1.0e-12
        _mu = 2.0e-5
        _dt_local = 0.1

        for _ in range(T):
            L = (np.roll(psi, 1, 0) + np.roll(psi, -1, 0) +
                 np.roll(psi, 1, 1) + np.roll(psi, -1, 1) - 4.0 * psi)
            fb = _alpha * L / (psi + _eps * np.exp(-_beta * psi**2))
            ent = _theta * psi * (
                np.roll(np.log(psi**2 + _delta), 1, 0) +
                np.roll(np.log(psi**2 + _delta), -1, 0) +
                np.roll(np.log(psi**2 + _delta), 1, 1) +
                np.roll(np.log(psi**2 + _delta), -1, 1) -
                4.0 * np.log(psi**2 + _delta)
            )
            psi = psi + _dt_local * (fb - ent) - _mu * psi
        return psi

    def test_soliton_matrix_matches(self):
        """Regenerated PDE output must match soliton_n12_matrix_32x32.csv."""
        data_path = os.path.join(
            os.path.dirname(__file__), "..",
            "data", "wavelock_data", "soliton_n12_matrix_32x32.csv"
        )
        if not os.path.exists(data_path):
            pytest.skip("soliton_n12_matrix_32x32.csv not found")

        expected = np.loadtxt(data_path, delimiter=",")
        actual = self._run_soliton_pde()
        np.testing.assert_allclose(actual, expected, rtol=1e-10,
                                   err_msg="PDE output does not match stored data")

    def test_soliton_flat_matches(self):
        """Regenerated PDE output must match soliton_n12.csv (flattened)."""
        data_path = os.path.join(
            os.path.dirname(__file__), "..",
            "data", "wavelock_data", "soliton_n12.csv"
        )
        if not os.path.exists(data_path):
            pytest.skip("soliton_n12.csv not found")

        expected_flat = np.loadtxt(data_path, delimiter=",")
        actual = self._run_soliton_pde()
        np.testing.assert_allclose(actual.ravel(), expected_flat, rtol=1e-10,
                                   err_msg="PDE flat output does not match stored data")

    def test_pde_deterministic(self):
        """Same seed must produce bitwise-identical PDE output."""
        r1 = self._run_soliton_pde(seed=42)
        r2 = self._run_soliton_pde(seed=42)
        np.testing.assert_array_equal(r1, r2)


# ============================================================
# 7. Cross-implementation consistency
# ============================================================
#
# CurvatureKeyPairV3 defaults to use_xof_init=True, which produces
# WLv3.1 commitments derived from SHAKE-256. CurvatureKeyPair (the
# GPU/cupy reference) emits WLv2 commitments by default and uses
# numpy MT for ψ₀ generation. To test cross-implementation byte
# parity we pin V3 to its legacy WLv2 + numpy-MT path so both
# implementations evaluate the same canonical serialization.

class TestCrossImplementation:
    """Verify WaveLock.py (cupy shim) and Wavelock_numpy.py produce consistent results."""

    def test_same_commitment_for_same_seed(self):
        """Both implementations with same seed must produce same commitment."""
        kp_main = CurvatureKeyPair(n=4, seed=77, test_mode=True)
        kp_np = CurvatureKeyPairV3(n=4, seed=77, use_xof_init=False)
        assert kp_main.commitment == kp_np.commitment, (
            "Main and numpy implementations produce different commitments for same seed"
        )

    def test_same_signature(self):
        """Both implementations must produce same signature for same message."""
        kp_main = CurvatureKeyPair(n=4, seed=77, test_mode=True)
        kp_np = CurvatureKeyPairV3(n=4, seed=77, use_xof_init=False)
        msg = "cross-impl test"
        assert kp_main.sign(msg) == kp_np.sign(msg)

    def test_cross_verify(self):
        """Signature from one implementation must verify with the other."""
        kp_main = CurvatureKeyPair(n=4, seed=77, test_mode=True)
        kp_np = CurvatureKeyPairV3(n=4, seed=77, use_xof_init=False)
        msg = "cross verify"

        sig_main = kp_main.sign(msg)
        assert kp_np.verify(msg, sig_main)

        sig_np = kp_np.sign(msg)
        assert kp_main.verify(msg, sig_np)
