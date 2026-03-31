#!/usr/bin/env python3
"""
hello_wavelock.py — WaveLock Hello World Demo

Demonstrates the core primitives of WaveLock:
  1. Generate a curvature keypair (psi*) and commitment
  2. Sign a message with SIGv2 curvature signature
  3. Mine a curvature-locked block into the local ledger
  4. Verify the entire ledger (hash -> linkage -> Merkle -> curvature)
  5. Run the Runaway-Drift Test (WaveLock's safety guarantee)

Safe to run. Writes only to ./ledger.
Requires: numpy (cupy optional for GPU acceleration).
"""

import sys, os
import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wavelock.chain.WaveLock import CurvatureKeyPair, symbolic_verifier, _to_numpy
from wavelock.chain.chain_utils import save_block_to_disk, verify_chain, reset_ledger
from wavelock.chain.Block import Block

print("\n====================================")
print("  WaveLock — Hello World Demo")
print("====================================\n")

# ---------------------------------------------------
# 1. Generate psi*, commitment, and signature
# ---------------------------------------------------

print("Generating curvature keypair...")
kp = CurvatureKeyPair(n=4, seed=123, test_mode=True)

print(f"    psi* shape       : {kp.psi_star.shape}")
print(f"    commitment       : {kp.commitment[:70]}...\n")

message = "hello wavelock"
signature = kp.sign(message)

print("Signing message:")
print(f"    message          : {message}")
print(f"    signature        : {signature}\n")

# ---------------------------------------------------
# 2. Build a curvature-locked block
# ---------------------------------------------------

print("Mining curvature-locked block...")

# Reset ledger for clean demo
reset_ledger(force=True)

messages = [
    f"message: {message}",
    f"signature: {signature}",
    f"commitment: {kp.commitment}",
]

block = Block(
    index=1,
    messages=messages,
    previous_hash="0" * 64,
    block_type="GENERIC",
    meta={}
)

save_block_to_disk(block)
print(f"    mined hash       : {block.hash[:16]}...\n")

# ---------------------------------------------------
# 3. Verify chain integrity
# ---------------------------------------------------

print("Verifying chain integrity...\n")
verify_chain(keypair=kp)
print()

# ---------------------------------------------------
# 4. Runaway Drift Test — WaveLock's safety guarantee
# ---------------------------------------------------

print("====================================")
print("  RUNAWAY DRIFT TEST")
print("====================================")

print("\nIntroducing curvature drift (tampering psi*)...")
tampered = _to_numpy(kp.psi_star).copy()
tampered[0, 0] += 0.5   # inject controlled drift

# 4A — Quantify drift
drift_mag = float(np.abs(tampered - _to_numpy(kp.psi_star)).sum())
print(f"    Drift magnitude (L1 norm): {drift_mag:.6f}")

# 4B — Curvature rail (geometric verification)
print("\n    symbolic_verifier(tampered, psi*) = ", end="")
curv_ok = symbolic_verifier(cp.asarray(tampered), kp.psi_star)
print(curv_ok)

if not curv_ok:
    print("    [OK] Curvature drift detected — WaveLock halts unsafe evolution")
else:
    print("    [FAIL] drift should never pass curvature verification")

# 4C — Signature rail (cryptographic verification)
print("\nVerifying signature under drift...")

kp_tampered = CurvatureKeyPair(n=4, test_mode=True)
kp_tampered.psi_star = cp.asarray(tampered, dtype=cp.float64)
kp_tampered.psi_0 = cp.zeros_like(cp.asarray(tampered))
kp_tampered.commitment = kp.commitment

sig_ok = kp_tampered.verify(message, signature)
print(f"    signature valid? = {sig_ok}")

if not sig_ok:
    print("    [OK] Drifted psi* cannot reproduce SIGv2 — recursion safely terminated")
else:
    print("    [FAIL] signature should have failed under psi* drift")

print("\n====================================")
print("  WaveLock Demo Complete")
print("====================================\n")
