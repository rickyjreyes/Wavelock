#!/usr/bin/env python3
import os
import sys
import numpy as np
import cupy as cp
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


# ---------------------------------------------------------
# Try importing WaveLock
# ---------------------------------------------------------
print("[INFO] Importing WaveLock CurvatureKeyPair...")

try:
    from wavelock.chain.WaveLock import CurvatureKeyPair
    print("[INFO] Import successful.")
except Exception as e:
    print("[ERROR] Could not import WaveLock!")
    print("Exception:", e)
    sys.exit(1)

# ---------------------------------------------------------
# Optional: check GPU availability
# ---------------------------------------------------------
print("[INFO] Checking GPU availability...")
try:
    _ = cp.zeros((1,))
    print("[INFO] CuPy GPU backend is working.")
except Exception as e:
    print("[WARNING] CuPy failed:", e)
    print("[WARNING] Running without GPU acceleration.")

# ---------------------------------------------------------
# Neural attack requires sklearn
# ---------------------------------------------------------
print("[INFO] Importing sklearn MLPRegressor...")
try:
    from sklearn.neural_network import MLPRegressor
    print("[INFO] sklearn import OK.")
except Exception as e:
    print("[ERROR] sklearn not installed:", e)
    sys.exit(1)


# ---------------------------------------------------------
# Neural inversion attack
# ---------------------------------------------------------
def run_scaled_neural_inversion():
    print("\n====================================================")
    print("      WAVELOCK — SCALED NEURAL INVERSION ATTACK     ")
    print("====================================================")

    n = 8
    samples = 2000       # reduce for speed, can increase
    train_size = 1600

    X = []
    Y = []

    print(f"[INFO] Generating {samples} Wavelock samples...")
    for i in range(samples):
        if i % 200 == 0:
            print(f"  - Sample {i}/{samples}")

        kp = CurvatureKeyPair(n=n, test_mode=True)


        psi_star_flat = cp.asnumpy(kp.psi_star).astype(np.float32).flatten()
        psi_0_flat    = cp.asnumpy(kp.psi_0).astype(np.float32).flatten()

        X.append(psi_star_flat)
        Y.append(psi_0_flat)

    X = np.stack(X)
    Y = np.stack(Y)

    print("[INFO] Data generated.")
    print("      X shape:", X.shape)
    print("      Y shape:", Y.shape)

    model = MLPRegressor(
        hidden_layer_sizes=(512, 512, 256),
        activation="relu",
        solver="adam",
        max_iter=50,         # adjustable to 200+
        random_state=42,
        verbose=True
    )

    print("\n[INFO] Training neural model...")
    model.fit(X[:train_size], Y[:train_size])

    print("\n[INFO] Evaluating model...")
    score = model.score(X[train_size:], Y[train_size:])

    print("\n==================== RESULTS ====================")
    print("Neural inversion score:", score)
    print("Expected behaviour: score < 0.1 (model FAILS to invert)")

    if score < 0.1:
        print("[PASS] Neural model FAILED to learn inverse. Good for security.")
    else:
        print("[WARNING] Model learned partial structure! Investigate.")

    print("====================================================\n")


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Running Wavelock neural inversion test...")
    run_scaled_neural_inversion()
    print("[INFO] Done.")
