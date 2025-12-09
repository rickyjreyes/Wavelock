"""
WaveLock v3 - NumPy-only Test Version

This version uses NumPy instead of CuPy for testing in CPU-only environments.
The physics is identical - only the array backend differs.
"""

from __future__ import annotations
from typing import Optional, Tuple
import hashlib
import json
import struct
import numpy as np
from dataclasses import dataclass

# Import hash families module
from hash_families import (
    HashFamily,
    DualHash,
    hash_data,
    hash_hex,
    format_commitment_v3,
    parse_commitment,
    DEFAULT_PRIMARY_FAMILY,
    DEFAULT_SECONDARY_FAMILY,
)


# ===========================================================================
#  Physics Parameters
# ===========================================================================

alpha   = 1.50
beta    = 0.0026
theta   = 1.0e-5
epsilon = 1.0e-12
delta   = 1.0e-12

_dt      = 0.1
_steps   = 50
_damping = 0.00002

KERNEL_VERSION = "WL-psi-001"
SCHEMA_V2 = "WLv2"


# ===========================================================================
#  Utility Functions
# ===========================================================================

def _canonical_json(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _float64_bytes(x) -> bytes:
    arr = np.asarray(x, dtype=np.float64)
    return arr.ravel(order="C").tobytes()


def laplacian(x):
    return (
        -4.0 * x
        + np.roll(x, +1, 0) + np.roll(x, -1, 0)
        + np.roll(x, +1, 1) + np.roll(x, -1, 1)
    )


def _kernel_descriptor() -> dict:
    return {
        "kernel_version": KERNEL_VERSION,
        "alpha": float(alpha),
        "beta":  float(beta),
        "theta": float(theta),
        "epsilon": float(epsilon),
        "delta": float(delta),
    }


def _kernel_hash() -> str:
    desc_bytes = _canonical_json(_kernel_descriptor())
    return hashlib.sha256(desc_bytes).hexdigest()


def _curvature_functional(psi) -> Tuple[float, float, float, float]:
    gx, gy = np.gradient(psi)
    E_grad = float(np.sum(gx * gx) + np.sum(gy * gy))

    lap = laplacian(psi)
    feedback = alpha * lap / (psi + epsilon * np.exp(-beta * psi ** 2))
    entropy_term = theta * (psi * laplacian(np.log(psi ** 2 + delta)))

    E_fb  = float(np.sum(feedback * feedback))
    E_ent = float(np.sum(entropy_term * entropy_term))
    E_tot = float(E_grad + E_fb + E_ent)
    
    return E_grad, E_fb, E_ent, E_tot


def _serialize_commitment_v2(psi) -> bytes:
    header_bytes = _canonical_json({
        "schema": SCHEMA_V2,
        "dtype": "float64",
        "ord": "C",
        "shape": [int(x) for x in psi.shape],
        "alpha": float(alpha),
        "beta":  float(beta),
        "theta": float(theta),
        "epsilon": float(epsilon),
        "delta": float(delta),
        "kernel_version": KERNEL_VERSION,
        "kernel_hash": _kernel_hash(),
    })

    E_grad, E_fb, E_ent, E_tot = _curvature_functional(psi)
    packed_E = struct.pack("<4d", E_grad, E_fb, E_ent, E_tot)

    return b"WLv2\0" + header_bytes + _float64_bytes(psi) + packed_E


# ===========================================================================
#  CurvatureKeyPairV3 - NumPy Version
# ===========================================================================

class CurvatureKeyPairV3:
    """Curvature keypair with dual-hash binding (NumPy version)."""
    
    def __init__(
        self,
        n: int,
        seed: Optional[int] = None,
        primary_family: HashFamily = DEFAULT_PRIMARY_FAMILY,
        secondary_family: HashFamily = DEFAULT_SECONDARY_FAMILY,
    ):
        self.n = n
        self.primary_family = primary_family
        self.secondary_family = secondary_family
        self.schema = SCHEMA_V2
        
        # RNG seed
        np.random.seed(seed)
        
        # Side length is always power-of-two
        side = 2 ** max(1, n // 2)
        
        # Initial field
        self.psi_0 = np.random.rand(side, side)
        
        # Evolve to fixed point
        self.psi_star = self._evolve(self.psi_0, n)
        
        # Serialize for commitment
        self._serialized = _serialize_commitment_v2(self.psi_star)
        
        # Compute dual-hash commitment
        self.dual_hash = DualHash.from_data(
            self._serialized,
            primary_family=primary_family,
            secondary_family=secondary_family,
        )
        
        primary_hex, secondary_hex = self.dual_hash.to_hex()
        self.commitment = format_commitment_v3(self.schema, primary_hex, secondary_hex)
        self.commitment_v2 = f"{self.schema}:{primary_hex}"
    
    def _evolve(self, psi0, n: int):
        """PDE evolution ψ₀ → ψ★"""
        psi = np.asarray(psi0, dtype=np.float64).copy()
        
        for _ in range(_steps):
            lap = laplacian(psi)
            fb  = alpha * lap / (psi + epsilon * np.exp(-beta * psi ** 2))
            ent = theta * (psi * laplacian(np.log(psi ** 2 + delta)))
            dpsi = _dt * (fb - ent) - _damping * psi
            psi = psi + dpsi
        
        return psi
    
    def _sig_payload(self, message: str) -> bytes:
        header = _canonical_json({
            "schema": self.schema,
            "dtype": "float64",
            "ord": "C",
            "shape": [int(x) for x in self.psi_star.shape],
            "alpha": float(alpha),
            "beta":  float(beta),
            "theta": float(theta),
            "epsilon": float(epsilon),
            "delta": float(delta),
            "kernel_version": KERNEL_VERSION,
            "kernel_hash": _kernel_hash(),
        })
        return (
            b"SIGv2\0"
            + message.encode() + b"\0"
            + header + b"\0"
            + _float64_bytes(self.psi_star)
        )
    
    def sign(self, message: str, family: Optional[HashFamily] = None) -> str:
        family = family or self.primary_family
        payload = self._sig_payload(message)
        return hash_hex(payload, family)
    
    def sign_dual(self, message: str) -> Tuple[str, str]:
        payload = self._sig_payload(message)
        return (
            hash_hex(payload, self.primary_family),
            hash_hex(payload, self.secondary_family),
        )
    
    def verify(self, message: str, signature: str) -> bool:
        expected_primary = self.sign(message, self.primary_family)
        if expected_primary == signature:
            return True
        expected_secondary = self.sign(message, self.secondary_family)
        return expected_secondary == signature
    
    def verify_strict(self, message: str, primary_sig: str, secondary_sig: str) -> bool:
        expected_p, expected_s = self.sign_dual(message)
        return (expected_p == primary_sig) and (expected_s == secondary_sig)
    
    def verify_commitment(self) -> Tuple[bool, bool]:
        return self.dual_hash.verify(
            self._serialized,
            self.primary_family,
            self.secondary_family,
        )
    
    def get_curvature_metrics(self) -> dict:
        E_grad, E_fb, E_ent, E_tot = _curvature_functional(self.psi_star)
        return {
            "E_grad": E_grad,
            "E_fb": E_fb,
            "E_ent": E_ent,
            "E_tot": E_tot,
        }


# ===========================================================================
#  Self-Test
# ===========================================================================

if __name__ == "__main__":
    print("=== CurvatureKeyPairV3 Self-Test (NumPy) ===\n")
    
    # Create keypair
    print("Creating V3 keypair with seed=42...")
    kp = CurvatureKeyPairV3(n=4, seed=42)
    
    print(f"ψ★ shape: {kp.psi_star.shape}")
    print(f"Commitment: {kp.commitment[:70]}...")
    print(f"Backward-compatible: {kp.commitment_v2[:50]}...")
    
    # Verify commitment
    p_ok, s_ok = kp.verify_commitment()
    print(f"\n✓ Commitment verification: primary={p_ok}, secondary={s_ok}")
    
    # Sign and verify
    message = "Hello, WaveLock V3!"
    sig_p, sig_s = kp.sign_dual(message)
    print(f"\nSignatures for '{message}':")
    print(f"  Primary (SHA-256):  {sig_p[:32]}...")
    print(f"  Secondary (SHA3):   {sig_s[:32]}...")
    
    # Verify with primary
    assert kp.verify(message, sig_p), "Primary signature verification failed"
    print("✓ Primary signature verified")
    
    # Verify with secondary (survivability mode)
    assert kp.verify(message, sig_s), "Secondary signature verification failed"
    print("✓ Secondary signature verified (survivability mode)")
    
    # Verify strict (both required)
    assert kp.verify_strict(message, sig_p, sig_s), "Strict verification failed"
    print("✓ Strict dual-signature verified")
    
    # Tamper detection
    tampered_sig = sig_p[:-4] + "XXXX"
    assert not kp.verify(message, tampered_sig), "Tampered signature should fail"
    print("✓ Tampered signature rejected")
    
    # Curvature metrics
    metrics = kp.get_curvature_metrics()
    print(f"\nCurvature metrics:")
    print(f"  E_grad: {metrics['E_grad']:.6f}")
    print(f"  E_fb:   {metrics['E_fb']:.6f}")
    print(f"  E_ent:  {metrics['E_ent']:.6f}")
    print(f"  E_tot:  {metrics['E_tot']:.6f}")
    
    # Parse and verify commitment format
    schema, ph, sh = parse_commitment(kp.commitment)
    assert schema == "WLv2", "Schema mismatch"
    assert len(ph) == 64, "Primary hash should be 64 hex chars"
    assert len(sh) == 64, "Secondary hash should be 64 hex chars"
    print(f"\n✓ Commitment format valid: {schema}:<sha256>:<sha3>")
    
    # Test determinism
    print("\nTesting determinism...")
    kp2 = CurvatureKeyPairV3(n=4, seed=42)
    assert kp.commitment == kp2.commitment, "Same seed should produce same commitment"
    print("✓ Deterministic: same seed → same commitment")
    
    kp3 = CurvatureKeyPairV3(n=4, seed=99)
    assert kp.commitment != kp3.commitment, "Different seed should produce different commitment"
    print("✓ Different seeds → different commitments")
    
    print("\n" + "="*50)
    print("ALL TESTS PASSED")
    print("="*50)
    print("\nWaveLock V3 dual-hash binding is operational.")
    print("The physics layer is unchanged. Only the binding is upgraded.")
    print("If SHA-256 breaks, SHA3 signatures remain valid.")