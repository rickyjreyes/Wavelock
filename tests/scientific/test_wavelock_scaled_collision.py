#!/usr/bin/env python3
import os
import sys
import numpy as np

# Ensure WaveLock root path is visible
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

print("[INFO] Importing WaveLock CurvatureKeyPair...")
try:
    from wavelock.chain.WaveLock import CurvatureKeyPair
    print("[INFO] Import successful.")
except Exception as e:
    print("[ERROR] Failed to import WaveLock:", e)
    sys.exit(1)


# ---------------------------------------------------------
# Collision helper
# ---------------------------------------------------------
def commit_from_seed(seed, n=4):
    """
    Deterministic mapping: seed -> Wavelock commitment.
    """
    kp = CurvatureKeyPair(n=n, seed=int(seed))
    return kp.commitment


# ---------------------------------------------------------
# Main collision search
# ---------------------------------------------------------
def run_scaled_collision_search():
    print("\n====================================================")
    print("        WAVELOCK — SCALED COLLISION SEARCH")
    print("====================================================")

    NUM = 1_000_000  # 1 million
    seen = set()

    print(f"[INFO] Searching {NUM:,} commitments for collisions...")

    for i in range(NUM):
        if i % 50_000 == 0:
            print(f"  - Progress: {i:,}/{NUM:,}")

        c = commit_from_seed(i)

        if c in seen:
            print("\n==================== COLLISION FOUND ====================")
            print(f"Seed i = {i}")
            print("Commitment:", c)
            print("==========================================================")
            return False  # suspicious

        seen.add(c)

    print("\n==================== RESULTS ====================")
    print(f"Completed {NUM:,} commitments — NO collisions found.")
    print("Expected: collision_probability ≈ 0 for 256-bit hash space.")

    print("\nPASS: No collisions detected.")
    print("====================================================\n")

    return True


# ---------------------------------------------------------
# Entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    result = run_scaled_collision_search()
    print("FINAL RESULT (True=Secure / False=Collision):", result)
    sys.exit(0 if result else 1)
