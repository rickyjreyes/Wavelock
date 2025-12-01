# hello_wavelock.py
#
# WaveLock — Hello World Demo
#
# This script demonstrates the core primitives of WaveLock:
#   1. Generate a curvature keypair (ψ*) and commitment
#   2. Sign a message with SIGv2 curvature signature
#   3. Mine a curvature-locked block into the local ledger
#   4. Verify the entire ledger (hash → linkage → Merkle → curvature)
#   5. Run the Runaway-Drift Test (WaveLock's safety guarantee)
#
# Safe to run. Writes only to ./ledger.

import sys, os
import cupy as cp

# Ensure project root on path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wavelock.chain.WaveLock import CurvatureKeyPair, symbolic_verifier
from wavelock.chain.chain_utils import save_block_to_disk, verify_chain
from wavelock.chain.Block import Block

print("\n====================================")
print(" 🌊  WaveLock — Hello World Demo")
print("====================================\n")

# ---------------------------------------------------
# 1. Generate ψ*, commitment, and signature
# ---------------------------------------------------

print("🔐  Generating curvature keypair...")
kp = CurvatureKeyPair(n=4, seed=123)

print(f"    ψ* shape       : {kp.psi_star.shape}")
print(f"    commitment     : {kp.commitment}\n")

message = "hello wavelock"
signature = kp.sign(message)

print("✍️  Signing message:")
print(f"    message        : {message}")
print(f"    signature      : {signature}\n")

# ---------------------------------------------------
# 2. Build a curvature-locked block
# ---------------------------------------------------

print("⛏️  Mining curvature-locked block...")

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
print(f"    mined hash     : {block.hash[:16]}…\n")

# ---------------------------------------------------
# 3. Verify chain integrity
# ---------------------------------------------------

print("🔎  Verifying chain integrity...\n")
verify_chain(keypair=kp)
print()

# ---------------------------------------------------
# 4. Runaway Drift Test — WaveLock’s critical safety guarantee
# ---------------------------------------------------

print("====================================")
print(" 🚨  RUNAWAY DRIFT TEST")
print("====================================")

print("\n🌀 Introducing curvature drift (tampering ψ*)...")
tampered = kp.psi_star.copy()
tampered[0, 0] += 0.5   # inject controlled drift

# 4A — Quantify drift
drift_mag = float(cp.abs(tampered - kp.psi_star).sum())
print(f"    Drift magnitude (L1 norm): {drift_mag:.6f}")

# 4B — Curvature rail (geometric verification)
print("\n🔧 symbolic_verifier(tampered, ψ*) → ", end="")
curv_ok = symbolic_verifier(tampered, kp.psi_star)
print(curv_ok)

if not curv_ok:
    print("    ✅ Curvature drift detected — WaveLock halts unsafe evolution")
else:
    print("    ❌ ERROR — drift should never pass curvature verification")

# 4C — Signature rail (cryptographic verification)
print("\n🔐 Verifying signature under drift...")

kp_tampered = CurvatureKeyPair(n=4)
kp_tampered.psi_star = tampered
kp_tampered.psi_0 = cp.zeros_like(tampered)
kp_tampered.commitment = kp.commitment

sig_ok = kp_tampered.verify(message, signature)
print(f"    signature valid? → {sig_ok}")

if not sig_ok:
    print("    ✅ Drifted ψ* cannot reproduce SIGv2 — recursion safely terminated")
else:
    print("    ❌ ERROR — signature should have failed under ψ* drift")

print("\n====================================")
print(" 🎉  WaveLock Demo Complete")
print("====================================\n")
