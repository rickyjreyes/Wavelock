#!/usr/bin/env python3
import os
import numpy as np
import cupy as cp

print("[WaveLock] Secure RNG + GPU reset")

# -------------------------------
# Generate valid 32-bit NumPy seed
# -------------------------------
seed32 = int.from_bytes(os.urandom(4), "little")  # 4 bytes → 32-bit
np.random.seed(seed32)
print(f"[OK] NumPy reseeded with 0x{seed32:08X}")

# -------------------------------
# Generate valid 64-bit CuPy seed
# -------------------------------
seed64 = int.from_bytes(os.urandom(8), "little")  # CuPy allows 64-bit
cp.random.seed(seed64)
print(f"[OK] CuPy reseeded with 0x{seed64:016X}")

# -------------------------------
# GPU memory wipe
# -------------------------------
cp.cuda.runtime.deviceSynchronize()
cp.get_default_memory_pool().free_all_blocks()
cp.get_default_pinned_memory_pool().free_all_blocks()
print("[OK] GPU memory pools cleared")

print("[WaveLock] Secure reset complete.")
