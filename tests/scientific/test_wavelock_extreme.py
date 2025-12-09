import numpy as np
import cupy as cp
import hashlib, time, math, sys, os

# ============================================================
# WaveLock imports (ONLY WaveLock)
# ============================================================

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    symbolic_verifier,
    _serialize_commitment_v2,
    laplacian,
)

# ============================================================
# HELPERS
# ============================================================

def commit(psi):
    """Compute WaveLock v2 commitment."""
    return hashlib.sha256(_serialize_commitment_v2(cp.asarray(psi))).hexdigest()

def verify_exact(a, b):
    return commit(a) == commit(b)

# ============================================================
# 1. MASSIVE AVALANCHE ANALYSIS
# ============================================================

def extreme_avalanche_test(n=6, flips=10000):
    kp = CurvatureKeyPair(n=n)
    base = cp.asarray(kp.psi_star, dtype=cp.float64)
    base_hash = commit(base)

    side = base.shape[0]
    failures = 0

    for _ in range(flips):
        perturbed = base.copy()
        x = np.random.randint(0, side)
        y = np.random.randint(0, side)
        perturbed[x, y] += np.random.normal(0, 1e-4)

        if commit(perturbed) == base_hash:
            failures += 1

    return {
        "flip_attempts": flips,
        "failures": failures,
        "failure_rate": failures / flips,
    }

# ============================================================
# 2. LONG-TERM DRIFT ACCUMULATION
# ============================================================

def extreme_drift_test(n=6, steps=1000):
    kp = CurvatureKeyPair(n=n)
    base = cp.asarray(kp.psi_star, dtype=cp.float64)
    c0 = commit(base)

    psi = base.copy()
    for _ in range(steps):
        psi += cp.random.normal(0, 1e-7, psi.shape)

    c1 = commit(psi)

    return {
        "same_commit": c1 == c0,
        "status": "PASS" if c1 != c0 else "FAIL",
    }

# ============================================================
# 3. PRECISION DOWNGRADE ATTACK
# ============================================================

def precision_attack(n=6):
    kp = CurvatureKeyPair(n=n)
    base = cp.asarray(kp.psi_star, dtype=cp.float64)
    c0 = commit(base)

    psi32 = cp.asarray(base, dtype=cp.float32)
    psi16 = cp.asarray(base, dtype=cp.float16)

    return {
        "float32_match": commit(psi32) == c0,
        "float16_match": commit(psi16) == c0,
    }

# ============================================================
# 4. FOURIER-SPACE ADVERSARIAL ATTACK
# ============================================================

def fourier_attack(n=6, attempts=200):
    kp = CurvatureKeyPair(n=n)
    target = cp.asarray(kp.psi_star, dtype=cp.float64)
    h0 = commit(target)

    side = target.shape[0]
    false_accepts = 0

    fx = cp.fft.fftfreq(side)
    fy = cp.fft.fftfreq(side)
    X, Y = cp.meshgrid(fx, fy)

    for _ in range(attempts):
        freq = np.random.randint(5, 200)

        wave = cp.sin(2 * np.pi * freq * X) * cp.cos(2 * np.pi * freq * Y)
        wave = wave / (cp.max(cp.abs(wave)) + 1e-12) * 0.01

        if commit(wave) == h0:
            false_accepts += 1

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / attempts,
    }

# ============================================================
# 5. RANDOM PROJECTION BLEND ATTACK
# ============================================================

def random_projection_attack(n=6, attempts=500):
    kp = CurvatureKeyPair(n=n)
    base = cp.asarray(kp.psi_star, dtype=cp.float64)
    h0 = commit(base)

    false_accepts = 0

    for _ in range(attempts):
        noise = cp.random.normal(0, 1, base.shape)
        mix = 0.01 * noise + 0.01 * base

        if commit(mix) == h0:
            false_accepts += 1

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / attempts,
    }

# ============================================================
# 6. MULTI-GPU CONSISTENCY
# ============================================================

def multi_gpu_test(n=6):
    if cp.cuda.runtime.getDeviceCount() < 2:
        return {"skip": True}

    cp.cuda.Device(0).use()
    kp0 = CurvatureKeyPair(n=n)
    h0 = commit(kp0.psi_star)

    cp.cuda.Device(1).use()
    kp1 = CurvatureKeyPair(n=n)
    h1 = commit(kp1.psi_star)

    return {
        "gpu_count": cp.cuda.runtime.getDeviceCount(),
        "same": h0 == h1,
    }

# ============================================================
# 7. QUANTIZATION (1-BIT ψ-FIELD)
# ============================================================

def quantization_attack(n=6):
    kp = CurvatureKeyPair(n=n)
    psi = cp.asarray(kp.psi_star, dtype=cp.float64)
    h0 = commit(psi)

    threshold = cp.mean(psi)
    q = (psi > threshold).astype(cp.float64)

    return {"quantized_match": commit(q) == h0}

# ============================================================
# 8. THERMAL NOISE MODEL ATTACK
# ============================================================

def thermal_attack(n=6, T=0.1):
    kp = CurvatureKeyPair(n=n)
    psi = cp.asarray(kp.psi_star, dtype=cp.float64)
    h0 = commit(psi)

    thermal = psi + cp.random.normal(0, T, psi.shape)
    return {"thermal_match": commit(thermal) == h0}

# ============================================================
# 9. ψ*-RECOMBINATION ATTACK
# ============================================================

def recombination_attack(n=6, attempts=200):
    kp = CurvatureKeyPair(n=n)
    psi = cp.asarray(kp.psi_star, dtype=cp.float64)
    h0 = commit(psi)

    false_accepts = 0
    for _ in range(attempts):
        noise = cp.random.rand(*psi.shape)
        mix = 0.5 * noise + 0.5 * psi

        if commit(mix) == h0:
            false_accepts += 1

    return {
        "false_accepts": false_accepts,
        "rate": false_accepts / attempts,
    }

# ============================================================
# 10. GRADIENT SURROGATE ATTACK
# ============================================================

def gradient_surrogate_attack(n=6, steps=300, lr=1e-4):
    kp = CurvatureKeyPair(n=n)
    target = cp.asarray(kp.psi_star, dtype=cp.float64)
    h0 = commit(target)

    psi = cp.random.rand(*target.shape)
    psi = cp.asarray(psi, dtype=cp.float64)

    for _ in range(steps):
        grad = laplacian(psi)
        psi -= lr * grad

        if commit(psi) == h0:
            return {"matched": True}

    return {"matched": False}






# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n=== WAVELOCK EXTREME TEST SUITE (no curvature functional) ===\n")

    print("1) Avalanche 10k:")
    print(extreme_avalanche_test())

    print("\n2) Drift Accumulation:")
    print(extreme_drift_test())

    print("\n3) Precision Attack:")
    print(precision_attack())

    print("\n4) Fourier Attack:")
    print(fourier_attack())

    print("\n5) Projection Attack:")
    print(random_projection_attack())

    print("\n6) Multi-GPU Test:")
    print(multi_gpu_test())

    print("\n7) Quantization Attack:")
    print(quantization_attack())

    print("\n8) Thermal Attack:")
    print(thermal_attack())

    print("\n9) Recombination Attack:")
    print(recombination_attack())

    print("\n10) Gradient Surrogate Attack:")
    print(gradient_surrogate_attack())

    print("\n=== DONE ===\n")
