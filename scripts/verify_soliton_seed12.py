#!/usr/bin/env python3
"""
Verify that the soliton PDE output for seed=12 matches the stored data files.

This script re-runs the exact PDE from soliton.py and checks bitwise/numerical
agreement with the CSV files in data/wavelock_data/.

Exit code 0 = all checks pass.
Exit code 1 = mismatch detected.
"""

import sys
import os
import numpy as np

# PDE parameters (must match soliton.py exactly)
alpha = 1.50
beta = 2.6e-3
theta = 1.0e-5
eps = 1.0e-12
delta = 1.0e-12
mu = 2.0e-5
dt = 0.1
T = 50
N = 32
seed = 12


def laplacian_periodic(u):
    return (
        np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
        np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4.0 * u
    )


def run_pde():
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal((N, N), dtype=np.float64)
    for _ in range(T):
        L = laplacian_periodic(psi)
        fb = alpha * L / (psi + eps * np.exp(-beta * psi**2))
        ent = theta * psi * laplacian_periodic(np.log(psi**2 + delta))
        psi = psi + dt * (fb - ent) - mu * psi
    return psi


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "wavelock_data")

    psi = run_pde()
    print(f"PDE output: shape={psi.shape}, mean={psi.mean():.6e}, std={psi.std():.6e}")

    ok = True

    # Check matrix file
    matrix_path = os.path.join(data_dir, "soliton_n12_matrix_32x32.csv")
    if os.path.exists(matrix_path):
        expected = np.loadtxt(matrix_path, delimiter=",")
        if np.allclose(psi, expected, rtol=1e-12):
            print(f"[PASS] {matrix_path}: matches (rtol=1e-12)")
        else:
            max_diff = np.max(np.abs(psi - expected))
            print(f"[FAIL] {matrix_path}: max diff = {max_diff:.2e}")
            ok = False
    else:
        print(f"[SKIP] {matrix_path}: file not found")

    # Check flat file
    flat_path = os.path.join(data_dir, "soliton_n12.csv")
    if os.path.exists(flat_path):
        expected_flat = np.loadtxt(flat_path, delimiter=",")
        if np.allclose(psi.ravel(), expected_flat, rtol=1e-12):
            print(f"[PASS] {flat_path}: matches (rtol=1e-12)")
        else:
            max_diff = np.max(np.abs(psi.ravel() - expected_flat))
            print(f"[FAIL] {flat_path}: max diff = {max_diff:.2e}")
            ok = False
    else:
        print(f"[SKIP] {flat_path}: file not found")

    # Determinism check
    psi2 = run_pde()
    if np.array_equal(psi, psi2):
        print("[PASS] Determinism: two runs produce identical output")
    else:
        print("[FAIL] Determinism: runs differ")
        ok = False

    if ok:
        print("\nAll checks passed.")
        return 0
    else:
        print("\nSome checks failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
