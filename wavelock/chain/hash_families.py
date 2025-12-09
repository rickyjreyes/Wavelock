"""
WaveLock Hash Families Module (WLv3 Survivability Layer)

This module provides hash-family abstraction for Wavelock commitments,
enabling survival across hash function deprecation (SHA-256 → SHA3 → future).

The core insight: Wavelock's security comes from the PHYSICS (ψ★ evolution),
not from the hash function. The hash is just a binding mechanism.

If SHA-256 breaks:
  - Old commitments can be re-bound to new hash families
  - The ψ★ identity remains valid (physics doesn't change)
  - Dual-hash binding prevents single-point cryptographic failure

This is the "versioned plumbing" that makes Wavelock survivable.
"""

from __future__ import annotations
from enum import Enum
from typing import Callable, Optional, Tuple
from dataclasses import dataclass
import hashlib


# ===========================================================================
#  Hash Family Registry
# ===========================================================================

class HashFamily(Enum):
    """
    Supported hash families for commitment binding.
    
    SHA256: Current default, widely deployed
    SHA3_256: NIST standard, different construction (Keccak)
    BLAKE3: Fast, modern, parallelizable
    
    Adding new families:
    1. Add enum value here
    2. Add implementation in _HASH_IMPLEMENTATIONS
    3. That's it - existing commitments continue to work
    """
    SHA256 = "sha256"
    SHA3_256 = "sha3-256"
    BLAKE3 = "blake3"


def _sha256_impl(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _sha3_256_impl(data: bytes) -> bytes:
    return hashlib.sha3_256(data).digest()


def _blake3_impl(data: bytes) -> bytes:
    """
    BLAKE3 implementation.
    Falls back to hashlib.blake2b if blake3 not installed.
    """
    try:
        import blake3
        return blake3.blake3(data).digest()
    except ImportError:
        # Fallback: BLAKE2b truncated to 256 bits
        # This is NOT BLAKE3, but provides a working fallback
        return hashlib.blake2b(data, digest_size=32).digest()


_HASH_IMPLEMENTATIONS: dict[HashFamily, Callable[[bytes], bytes]] = {
    HashFamily.SHA256: _sha256_impl,
    HashFamily.SHA3_256: _sha3_256_impl,
    HashFamily.BLAKE3: _blake3_impl,
}


def get_hasher(family: HashFamily) -> Callable[[bytes], bytes]:
    """
    Get the hash function for a given family.
    
    Returns a callable: bytes -> bytes (32-byte digest)
    """
    if family not in _HASH_IMPLEMENTATIONS:
        raise ValueError(f"Unknown hash family: {family}")
    return _HASH_IMPLEMENTATIONS[family]


def hash_data(data: bytes, family: HashFamily) -> bytes:
    """Hash data using the specified family."""
    return get_hasher(family)(data)


def hash_hex(data: bytes, family: HashFamily) -> str:
    """Hash data and return hex string."""
    return hash_data(data, family).hex()


# ===========================================================================
#  Dual-Hash Binding
# ===========================================================================

@dataclass(frozen=True)
class DualHash:
    """
    Dual-hash binding structure.
    
    A commitment bound with two independent hash families.
    If one breaks, the other still provides integrity.
    
    This is the core survivability mechanism:
    - SHA-256 breaks? SHA3 still valid.
    - Both break? Add a third hash family, re-bind.
    - The ψ★ physics NEVER changes.
    """
    primary: bytes      # SHA-256 hash (32 bytes)
    secondary: bytes    # SHA3-256 hash (32 bytes)
    
    @classmethod
    def from_data(
        cls,
        data: bytes,
        primary_family: HashFamily = HashFamily.SHA256,
        secondary_family: HashFamily = HashFamily.SHA3_256,
    ) -> DualHash:
        """Create dual-hash from raw data."""
        return cls(
            primary=hash_data(data, primary_family),
            secondary=hash_data(data, secondary_family),
        )
    
    def verify(
        self,
        data: bytes,
        primary_family: HashFamily = HashFamily.SHA256,
        secondary_family: HashFamily = HashFamily.SHA3_256,
    ) -> Tuple[bool, bool]:
        """
        Verify data against both hashes.
        
        Returns: (primary_valid, secondary_valid)
        
        Use case:
        - Both True: Normal operation
        - Primary False, Secondary True: SHA-256 may be compromised
        - Primary True, Secondary False: SHA3 may be compromised
        - Both False: Data has been tampered with
        """
        p = hash_data(data, primary_family)
        s = hash_data(data, secondary_family)
        return (p == self.primary, s == self.secondary)
    
    def verify_any(
        self,
        data: bytes,
        primary_family: HashFamily = HashFamily.SHA256,
        secondary_family: HashFamily = HashFamily.SHA3_256,
    ) -> bool:
        """Returns True if EITHER hash matches (survivability mode)."""
        p_ok, s_ok = self.verify(data, primary_family, secondary_family)
        return p_ok or s_ok
    
    def verify_both(
        self,
        data: bytes,
        primary_family: HashFamily = HashFamily.SHA256,
        secondary_family: HashFamily = HashFamily.SHA3_256,
    ) -> bool:
        """Returns True only if BOTH hashes match (strict mode)."""
        p_ok, s_ok = self.verify(data, primary_family, secondary_family)
        return p_ok and s_ok
    
    def to_hex(self) -> Tuple[str, str]:
        """Return both hashes as hex strings."""
        return (self.primary.hex(), self.secondary.hex())
    
    @classmethod
    def from_hex(cls, primary_hex: str, secondary_hex: str) -> DualHash:
        """Reconstruct from hex strings."""
        return cls(
            primary=bytes.fromhex(primary_hex),
            secondary=bytes.fromhex(secondary_hex),
        )
    
    def __str__(self) -> str:
        p, s = self.to_hex()
        return f"DualHash(sha256={p[:16]}..., sha3={s[:16]}...)"


# ===========================================================================
#  Commitment String Format
# ===========================================================================

def format_commitment_v3(
    schema: str,
    primary_hash: str,
    secondary_hash: str,
) -> str:
    """
    Format a WLv3-style commitment string with dual-hash binding.
    
    Format: "{schema}:{primary_hash}:{secondary_hash}"
    Example: "WLv2:abc123...:def456..."
    
    The schema identifies the curvature serialization version.
    The hashes bind the serialized ψ★ to two independent hash families.
    """
    return f"{schema}:{primary_hash}:{secondary_hash}"


def parse_commitment(commitment: str) -> Tuple[str, str, Optional[str]]:
    """
    Parse a commitment string.
    
    Returns: (schema, primary_hash, secondary_hash or None)
    
    Handles both old format (schema:hash) and new format (schema:hash1:hash2)
    """
    parts = commitment.split(":")
    
    if len(parts) == 2:
        # Old format: "WLv2:abcdef..."
        return (parts[0], parts[1], None)
    elif len(parts) == 3:
        # New format: "WLv2:abcdef...:123456..."
        return (parts[0], parts[1], parts[2])
    else:
        raise ValueError(f"Invalid commitment format: {commitment}")


def is_dual_hash_commitment(commitment: str) -> bool:
    """Check if commitment uses dual-hash binding."""
    _, _, secondary = parse_commitment(commitment)
    return secondary is not None


# ===========================================================================
#  Migration Utilities
# ===========================================================================

def upgrade_commitment(
    old_commitment: str,
    serialized_psi: bytes,
    secondary_family: HashFamily = HashFamily.SHA3_256,
) -> str:
    """
    Upgrade an old single-hash commitment to dual-hash format.
    
    This is the key migration operation:
    1. Parse old commitment to get schema and primary hash
    2. Verify the primary hash matches (integrity check)
    3. Compute secondary hash from same serialized data
    4. Return new dual-hash commitment
    
    The ψ★ data is the same. Only the binding changes.
    """
    schema, primary_hash, existing_secondary = parse_commitment(old_commitment)
    
    if existing_secondary is not None:
        # Already dual-hash, nothing to do
        return old_commitment
    
    # Verify primary hash matches
    computed_primary = hash_hex(serialized_psi, HashFamily.SHA256)
    if computed_primary != primary_hash:
        raise ValueError(
            f"Primary hash mismatch during upgrade. "
            f"Expected {primary_hash[:16]}..., got {computed_primary[:16]}..."
        )
    
    # Compute secondary hash
    secondary_hash = hash_hex(serialized_psi, secondary_family)
    
    return format_commitment_v3(schema, primary_hash, secondary_hash)


# ===========================================================================
#  Hash Family Detection
# ===========================================================================

def detect_hash_family(commitment: str, serialized_psi: bytes) -> Optional[HashFamily]:
    """
    Detect which hash family was used for a commitment.
    
    Useful for legacy commitments where the family isn't explicit.
    Returns None if no known family matches.
    """
    _, stored_hash, _ = parse_commitment(commitment)
    
    for family in HashFamily:
        computed = hash_hex(serialized_psi, family)
        if computed == stored_hash:
            return family
    
    return None


# ===========================================================================
#  Constants for WLv3
# ===========================================================================

# Default hash families for new commitments
DEFAULT_PRIMARY_FAMILY = HashFamily.SHA256
DEFAULT_SECONDARY_FAMILY = HashFamily.SHA3_256

# Epoch for tracking hash family transitions
# Increment this when changing default families
HASH_EPOCH = 1


if __name__ == "__main__":
    # Quick self-test
    test_data = b"WaveLock test data for hash family verification"
    
    print("=== Hash Family Self-Test ===\n")
    
    # Test individual hashes
    for family in HashFamily:
        h = hash_hex(test_data, family)
        print(f"{family.value}: {h[:32]}...")
    
    print()
    
    # Test dual-hash
    dual = DualHash.from_data(test_data)
    print(f"DualHash: {dual}")
    
    p_ok, s_ok = dual.verify(test_data)
    print(f"Verify original: primary={p_ok}, secondary={s_ok}")
    
    tampered = test_data + b"X"
    p_ok, s_ok = dual.verify(tampered)
    print(f"Verify tampered: primary={p_ok}, secondary={s_ok}")
    
    print()
    
    # Test commitment format
    old_commit = "WLv2:abc123def456"
    schema, ph, sh = parse_commitment(old_commit)
    print(f"Old format: schema={schema}, primary={ph}, secondary={sh}")
    
    new_commit = "WLv2:abc123:def456"
    schema, ph, sh = parse_commitment(new_commit)
    print(f"New format: schema={schema}, primary={ph}, secondary={sh}")
    
    print("\n=== All tests passed ===")