"""
WaveLock V3 Migration Utility

Upgrades existing WLv2 commitments to dual-hash format.

Usage:
    # Upgrade a single keypair
    python migrate.py upgrade keypair.json
    
    # Upgrade trusted commitments file
    python migrate.py upgrade-trust trusted_commitments.json
    
    # Batch upgrade user registry
    python migrate.py upgrade-users users.json

The migration is NON-DESTRUCTIVE:
- Original files are backed up to *.backup
- New files contain dual-hash format
- Old signatures remain valid (backward compatibility)
"""

import json
import sys
import os
import shutil
from typing import Optional
import numpy as np

from .hash_families import (
    HashFamily,
    hash_hex,
    format_commitment_v3,
    parse_commitment,
    is_dual_hash_commitment,
    DEFAULT_SECONDARY_FAMILY,
)


# ===========================================================================
#  Serialization (must match WaveLock.py exactly)
# ===========================================================================

import struct
import hashlib

alpha   = 1.50
beta    = 0.0026
theta   = 1.0e-5
epsilon = 1.0e-12
delta   = 1.0e-12
KERNEL_VERSION = "WL-psi-001"
SCHEMA_V2 = "WLv2"


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


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
    return hashlib.sha256(_canonical_json(_kernel_descriptor())).hexdigest()


def laplacian(x):
    return (
        -4.0 * x
        + np.roll(x, +1, 0) + np.roll(x, -1, 0)
        + np.roll(x, +1, 1) + np.roll(x, -1, 1)
    )


def _curvature_functional(psi):
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
    """Serialize ψ★ for commitment (must match WaveLock.py exactly)."""
    psi = np.asarray(psi, dtype=np.float64)
    
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
    
    psi_bytes = psi.ravel(order="C").tobytes()

    return b"WLv2\0" + header_bytes + psi_bytes + packed_E


# ===========================================================================
#  Upgrade Functions
# ===========================================================================
def upgrade_commitment_with_psi(old_commit, raw_bytes: bytes) -> str:
    parts = old_commit.split(":")

    # Case 1 — Already WLv3
    if len(parts) == 3:
        return old_commit

    # Case 2 — Must be exactly (schema:primary)
    if len(parts) != 2:
        raise ValueError("Invalid legacy commitment format")

    schema, primary_hex = parts

    # ❗ REQUIRED BY TEST: Verify old_raw matches primary hash
    computed_primary = hash_hex(raw_bytes, HashFamily.SHA256)
    if computed_primary != primary_hex:
        raise ValueError("Primary hash mismatch during upgrade")

    # Compute secondary hash (WLv3)
    secondary_hex = hash_hex(raw_bytes, HashFamily.SHA3_256)

    return f"{schema}:{primary_hex}:{secondary_hex}"




def backup_file(path: str) -> str:
    """Create backup of file, return backup path."""
    backup_path = path + ".backup"
    counter = 1
    while os.path.exists(backup_path):
        backup_path = f"{path}.backup.{counter}"
        counter += 1
    shutil.copy2(path, backup_path)
    return backup_path


# ===========================================================================
#  Command: Upgrade Single Keypair
# ===========================================================================

def cmd_upgrade_keypair(path: str):
    """Upgrade a single keypair.json file."""
    print(f"\n=== Upgrading keypair: {path} ===\n")
    
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return False
    
    with open(path, "r") as f:
        data = json.load(f)
    
    # Extract fields
    old_commitment = data.get("commitment", "")
    psi_star = np.array(data.get("psi_star", []), dtype=np.float64)
    
    if not old_commitment:
        print("ERROR: No commitment found in file")
        return False
    
    if psi_star.size == 0:
        print("ERROR: No psi_star found in file")
        return False
    
    # Upgrade
    try:
        new_commitment = upgrade_commitment_with_psi(old_commitment, psi_star)
    except ValueError as e:
        print(f"ERROR: {e}")
        return False
    
    if new_commitment == old_commitment:
        print("No upgrade needed.")
        return True
    
    # Backup and save
    backup_path = backup_file(path)
    print(f"  Backup: {backup_path}")
    
    data["commitment"] = new_commitment
    data["commitment_v2"] = old_commitment  # Keep old for backward compat
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"✓ Upgraded: {path}")
    return True


# ===========================================================================
#  Command: Upgrade Users Registry
# ===========================================================================

def cmd_upgrade_users(path: str):
    """Upgrade users.json registry file."""
    print(f"\n=== Upgrading user registry: {path} ===\n")
    
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return False
    
    with open(path, "r") as f:
        users = json.load(f)
    
    upgraded_count = 0
    skipped_count = 0
    
    for user_id, user_data in users.items():
        print(f"\nUser: {user_id}")
        
        old_commitment = user_data.get("commitment", "")
        psi_star_list = user_data.get("psi_star", [])
        
        if not old_commitment or not psi_star_list:
            print("  Skipped: missing commitment or psi_star")
            skipped_count += 1
            continue
        
        psi_star = np.array(psi_star_list, dtype=np.float64)
        
        try:
            new_commitment = upgrade_commitment_with_psi(old_commitment, psi_star)
        except ValueError as e:
            print(f"  ERROR: {e}")
            skipped_count += 1
            continue
        
        if new_commitment != old_commitment:
            user_data["commitment"] = new_commitment
            user_data["commitment_v2"] = old_commitment
            upgraded_count += 1
        else:
            skipped_count += 1
    
    # Backup and save
    backup_path = backup_file(path)
    print(f"\nBackup: {backup_path}")
    
    with open(path, "w") as f:
        json.dump(users, f, indent=2)
    
    print(f"\n✓ Upgraded {upgraded_count} users, skipped {skipped_count}")
    return True


# ===========================================================================
#  Command: Upgrade Trusted Commitments
# ===========================================================================

def cmd_upgrade_trust(path: str, users_path: Optional[str] = None):
    """
    Upgrade trusted_commitments.json file.
    
    Note: This requires access to the corresponding ψ★ data.
    If users_path is provided, looks up ψ★ from users.json.
    """
    print(f"\n=== Upgrading trusted commitments: {path} ===\n")
    
    if not os.path.exists(path):
        print(f"ERROR: File not found: {path}")
        return False
    
    with open(path, "r") as f:
        trusted = json.load(f)
    
    # Load users if available
    users = {}
    if users_path and os.path.exists(users_path):
        with open(users_path, "r") as f:
            users = json.load(f)
        print(f"Loaded {len(users)} users from {users_path}")
    
    # Build commitment -> psi_star mapping
    commitment_to_psi = {}
    for user_id, user_data in users.items():
        commit = user_data.get("commitment", "")
        psi = user_data.get("psi_star", [])
        if commit and psi:
            # Store by primary hash for lookup
            schema, ph, _ = parse_commitment(commit)
            commitment_to_psi[ph] = np.array(psi, dtype=np.float64)
    
    # Upgrade trusted list
    new_trusted = []
    upgraded_count = 0
    
    for commit in trusted:
        if is_dual_hash_commitment(commit):
            new_trusted.append(commit)
            continue
        
        schema, ph, _ = parse_commitment(commit)
        
        if ph in commitment_to_psi:
            psi_star = commitment_to_psi[ph]
            try:
                new_commit = upgrade_commitment_with_psi(commit, psi_star)
                new_trusted.append(new_commit)
                upgraded_count += 1
            except ValueError as e:
                print(f"  ERROR upgrading {commit[:30]}...: {e}")
                new_trusted.append(commit)  # Keep original
        else:
            print(f"  No ψ★ found for {commit[:30]}... (keeping original)")
            new_trusted.append(commit)
    
    # Backup and save
    backup_path = backup_file(path)
    print(f"\nBackup: {backup_path}")
    
    with open(path, "w") as f:
        json.dump(new_trusted, f, indent=2)
    
    print(f"\n✓ Upgraded {upgraded_count} of {len(trusted)} commitments")
    return True


# ===========================================================================
#  Main
# ===========================================================================

def print_usage():
    print("""
WaveLock V3 Migration Utility

Usage:
    python migrate.py upgrade <keypair.json>
        Upgrade a single keypair file to dual-hash format
    
    python migrate.py upgrade-users <users.json>
        Upgrade all users in a registry file
    
    python migrate.py upgrade-trust <trusted_commitments.json> [users.json]
        Upgrade trusted commitments (requires users.json for ψ★ lookup)
    
    python migrate.py verify <file.json>
        Verify a file contains valid dual-hash commitments

All operations create .backup files before modifying.
""")


def main():
    if len(sys.argv) < 3:
        print_usage()
        sys.exit(1)
    
    cmd = sys.argv[1]
    path = sys.argv[2]
    
    if cmd == "upgrade":
        success = cmd_upgrade_keypair(path)
    elif cmd == "upgrade-users":
        success = cmd_upgrade_users(path)
    elif cmd == "upgrade-trust":
        users_path = sys.argv[3] if len(sys.argv) > 3 else None
        success = cmd_upgrade_trust(path, users_path)
    elif cmd == "verify":
        print("Verify not yet implemented")
        success = False
    else:
        print(f"Unknown command: {cmd}")
        print_usage()
        sys.exit(1)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()