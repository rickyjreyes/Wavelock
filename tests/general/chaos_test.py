# tests/chaos_test.py
# ============================================================
# Phase Space Determinism Detector (k-Nearest Neighbor)
# Replaces the fragile Lyapunov Sync.
# Logic: Checks if similar inputs produce similar outputs.
# ============================================================

import numpy as np

def run_attack(stream_data, embed_dim=1):
    print(f"   [CHAOS] Analyzing Phase Space Determinism ({len(stream_data)} samples)...")
    
    # 1. Normalize Data
    data = np.array(stream_data)
    d_min, d_max = data.min(), data.max()
    if d_max - d_min < 1e-9: return "FAIL (Constant Data)"
    norm_data = (data - d_min) / (d_max - d_min)
    
    # 2. Phase Space Reconstruction (Find False Nearest Neighbors)
    # We look for points x_i that are close to x_j,
    # and check if their futures x_{i+1} and x_{j+1} are also close.
    
    errors = []
    # Sample 100 random points to test
    indices = np.random.choice(len(norm_data)-2, size=min(100, len(norm_data)-2), replace=False)
    
    for idx in indices:
        current_val = norm_data[idx]
        
        # Find the nearest neighbor in the rest of the dataset
        # (Excluding the point itself and immediate neighbors to avoid auto-correlation)
        # Create a mask to exclude self
        candidates = np.delete(np.arange(len(norm_data)-1), [idx, idx+1, idx-1])
        candidate_vals = norm_data[candidates]
        
        # Find index of closest value
        closest_arg = np.abs(candidate_vals - current_val).argmin()
        neighbor_idx = candidates[closest_arg]
        
        # 3. Predict Future
        # If deterministic, the neighbor's future should match the point's future
        real_future = norm_data[idx + 1]
        neighbor_future = norm_data[neighbor_idx + 1]
        
        prediction_error = abs(real_future - neighbor_future)
        errors.append(prediction_error)

    avg_determinism_error = np.mean(errors)
    print(f"   [CHAOS] Phase Space Divergence: {avg_determinism_error:.5f}")
    
    # Interpretation:
    # Low Error (< 0.05) = Points that look alike behave alike (Math/Deterministic) -> FAIL
    # High Error (> 0.20) = Points that look alike behave differently (Vacuum/Noise) -> PASS
    
    if avg_determinism_error < 0.10: # Threshold for determinism
        return "FAIL (Deterministic Patterns Detected)"
    else:
        return "PASS (High Entropy / Non-Deterministic)"

if __name__ == "__main__":
    print("=== SELF-TEST: Chaos Determinism Detector ===")
    
    # Test 1: Logistic Map (Math Chaos) -> Should FAIL (Low Divergence)
    print("\nTest 1: Logistic Map (Math Chaos)")
    chaos_data = []
    x = 0.1
    for _ in range(500):
        x = 4.0 * x * (1.0 - x)
        chaos_data.append(x)
    print("Verdict:", run_attack(chaos_data))
    
    # Test 2: Random Noise -> Should PASS (High Divergence)
    print("\nTest 2: Random Noise (Entropy)")
    rand_data = [np.random.random() for _ in range(500)]
    print("Verdict:", run_attack(rand_data))