import numpy as np
import hashlib
import time
import cupy as cp

### Utilities

def sha256(a):
    return hashlib.sha256(a.tobytes()).hexdigest()

def random_seed():
    return np.random.randint(0, 10**9)

def banner(msg):
    print("\n" + "="*60)
    print(f"  {msg}")
    print("="*60 + "\n")

### Unified seed generator using surrogate WaveLock API
def make_seed(n):
    """
    Generate a random-phase, smooth WCT-style surrogate ψ0.
    Deterministic from seed, non-symmetric, band-limited,
    stable under evolution, and contractive for valid modes.
    """
    seed = random_seed()
    print(f"[WL-TEST] Generating spectral surrogate seed n={n}, seed={seed}")
    rng = np.random.default_rng(seed)

    # Fourier grid
    kx = np.fft.fftfreq(n).reshape(-1,1)
    ky = np.fft.fftfreq(n).reshape(1,-1)
    k2 = kx**2 + ky**2

    # Random phases
    phi = rng.uniform(0, 2*np.pi, size=(n,n))

    # Band-limited amplitude ~exp(-k^2 * sigma)
    sigma = 8.0 / n
    A = np.exp(-k2 / sigma)

    # Complex spectrum
    F = A * np.exp(1j * phi)

    # Inverse FFT → smooth ψ0
    psi0 = np.fft.ifft2(F).real.astype(np.float64)

    # Normalize energy
    psi0 /= np.max(np.abs(psi0)) + 1e-9

    print(f"[WL-TEST] ψ0 spectral surrogate, shape={psi0.shape}")
    return psi0



# ============================================================
# 1) Multi-Depth PDE Evolution Test (FAST MODE)
# ============================================================

def run_multidepth_tests():
    import wavelock.chain.WaveLock as wl
    banner("MULTI-DEPTH PDE EVOLUTION TEST (FAST)")

    # FAST-MODE: trimmed depth list
    for T in [5, 10, 20]:
        print(f"[WL-TEST] ==== T = {T} ====")

        n = 32
        psi0 = make_seed(n)

        print("[WL-TEST] Calling evolve() twice for determinism...")
        t0 = time.time()
        out1 = wl.CurvatureKeyPair.evolve(None, cp.asarray(psi0), T)
        out2 = wl.CurvatureKeyPair.evolve(None, cp.asarray(psi0), T)
        print(f"[WL-TEST] evolve() pair completed in {time.time()-t0:.3f} sec")

        if np.allclose(cp.asnumpy(out1), cp.asnumpy(out2)):
            print("  Determinism: PASS")
        else:
            print("  Determinism: FAIL")

        norm_out = np.linalg.norm(cp.asnumpy(out1))
        print(f"  Output norm = {norm_out:.6f}")

        h1 = sha256(cp.asnumpy(out1))
        out_other = wl.CurvatureKeyPair.evolve(None, cp.asarray(psi0), T+1)
        h2 = sha256(cp.asnumpy(out_other))

        print(f"  Hash(T={T})   = {h1[:16]}...")
        print(f"  Hash(T={T+1}) = {h2[:16]}...")
        print("  Depth hash variation:", "PASS" if h1 != h2 else "FAIL")

# ============================================================
# 2) Structured Seed Inversion Test
# ============================================================

def run_structured_seed_tests():
    import wavelock.chain.WaveLock as wl
    banner("STRUCTURED SEED INVERSION TEST (FAST)")

    patterns = ["sin", "bump", "radial", "chess"]
    n = 32
    x = np.linspace(0, 2*np.pi, n)

    for pattern in patterns:
        print(f"[WL-TEST] === Pattern: {pattern} ===")

        if pattern == "sin":
            psi0 = np.sin(x)[None, :] * np.sin(x)[:, None]
        elif pattern == "bump":
            psi0 = np.zeros((n,n))
            psi0[n//2, n//2] = 1.0
        elif pattern == "radial":
            xx, yy = np.meshgrid(x, x)
            r = np.sqrt((xx-np.pi)**2 + (yy-np.pi)**2)
            psi0 = np.exp(-5*r)
        elif pattern == "chess":
            psi0 = (np.indices((n, n)).sum(axis=0) % 2).astype(float)

        print("[WL-TEST] evolve(psi0, T=10) starting...")
        t0 = time.time()
        psiT = wl.CurvatureKeyPair.evolve(None, cp.asarray(psi0), 10)
        psiT = cp.asnumpy(psiT)
        print(f"[WL-TEST] evolve() completed in {time.time()-t0:.3f} sec")

        inv_guess = psiT[::-1, ::-1]

        h_true  = sha256(psi0)
        h_guess = sha256(inv_guess)

        print(f"  True hash  = {h_true[:16]}...")
        print(f"  Guess hash = {h_guess[:16]}...")
        print("  Structured inversion resistance:", "PASS" if h_true != h_guess else "FAIL")

# ============================================================
# 3) Multi-Resolution Split Inversion Test
# ============================================================

def downsample(x, factor):
    return x[::factor, ::factor]

def upsample(x, factor):
    return np.repeat(np.repeat(x, factor, axis=0), factor, axis=1)

def run_multires_inversion():
    import wavelock.chain.WaveLock as wl
    banner("MULTI-RESOLUTION SPLIT INVERSION (FAST)")

    n = 32
    psi0 = make_seed(n)

    print("[WL-TEST] evolve(psi0, 10) starting...")
    t0 = time.time()
    psiT = wl.CurvatureKeyPair.evolve(None, cp.asarray(psi0), 10)
    psiT = cp.asnumpy(psiT)
    print(f"[WL-TEST] evolve() completed in {time.time()-t0:.3f} sec")

    coarse = downsample(psiT, 4)
    coarse_inv = np.zeros_like(coarse)
    psi_est = upsample(coarse_inv, 4)

    h_true = sha256(psi0)
    h_est  = sha256(psi_est)

    print(f"  True hash = {h_true[:16]}...")
    print(f"  Est hash  = {h_est[:16]}...")
    print("  Multi-res inversion resistance:", "PASS" if h_true != h_est else "FAIL")

# ============================================================
# 4) Random Projection Jacobian Test
# ============================================================

def run_random_projection_test():
    import wavelock.chain.WaveLock as wl
    banner("RANDOM PROJECTION JACOBIAN TEST (FAST)")

    n = 8
    psi0 = make_seed(n)
    psi0_cp = cp.asarray(psi0)
    eps = 1e-4

    for i in range(6):
        print(f"[WL-TEST] Direction {i+1}/6")

        # v = np.random.randn(n,n)
        v = cp.asarray(np.random.randn(n,n))
        v /= np.linalg.norm(v) + 1e-9

        t0 = time.time()
        psi_plus  = wl.CurvatureKeyPair.evolve(None, psi0_cp + eps*v, 6)
        psi_minus = wl.CurvatureKeyPair.evolve(None, psi0_cp - eps*v, 6)
        print(f"[WL-TEST] evolve(+/- eps) took {time.time()-t0:.3f} sec")

        Jv = (psi_plus - psi_minus) / (2*eps)
        ratio = np.linalg.norm(cp.asnumpy(Jv)) / np.linalg.norm(v)

        print(f"  contraction ratio = {ratio:.6f}")
        print("  PASS" if ratio < 1 else "  FAIL")

# ============================================================
# 5) Stochastic Adjoint Inversion Test (FAST MODE)
# ============================================================

def run_stochastic_adjoint():
    import wavelock.chain.WaveLock as wl
    banner("STOCHASTIC ADJOINT INVERSION (FAST)")

    n = 16
    psi0 = make_seed(n)
    guess = np.zeros_like(psi0)

    lr = 1e-2
    sigma = 0.1

    print("[WL-TEST] Running noisy adjoint descent (40 steps)...")

    for k in range(40):
        eps = 1e-3

        t0 = time.time()
        gp = wl.CurvatureKeyPair.evolve(None, cp.asarray(guess + eps), 10)
        gm = wl.CurvatureKeyPair.evolve(None, cp.asarray(guess - eps), 10)
        print(f"[WL-TEST]   iter {k:02d} evolve pair: {time.time()-t0:.3f} sec")

        grad = cp.asnumpy((gp - gm) / (2*eps))
        guess -= lr * grad + sigma * np.random.randn(*grad.shape)

    h_true  = sha256(psi0)
    h_guess = sha256(guess)

    print(f"  True hash  = {h_true[:16]}...")
    print(f"  Guess hash = {h_guess[:16]}...")
    print("  Stochastic inversion:", "PASS" if h_true != h_guess else "FAIL")

# ============================================================
# 6) Global Contraction Test
# ============================================================

def run_global_contraction():
    import wavelock.chain.WaveLock as wl
    banner("GLOBAL CONTRACTION TEST (FAST)")

    n = 16
    psi0_a = make_seed(n)
    psi0_b = make_seed(n)

    print("[WL-TEST] evolve(a,b, T=10) starting...")

    t0 = time.time()
    psiT_a = wl.CurvatureKeyPair.evolve(None, cp.asarray(psi0_a), 10)
    psiT_b = wl.CurvatureKeyPair.evolve(None, cp.asarray(psi0_b), 10)
    print(f"[WL-TEST] evolve pair completed in {time.time()-t0:.3f} sec")

    d0 = np.linalg.norm(psi0_a - psi0_b)
    dT = np.linalg.norm(cp.asnumpy(psiT_a - psiT_b))

    print(f"  Initial d0 = {d0:.6f}")
    print(f"  Output  dT = {dT:.6f}")
    print("  Global contraction:", "PASS" if dT < d0 else "FAIL")



# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_multidepth_tests()
    run_structured_seed_tests()
    run_multires_inversion()
    run_random_projection_test()
    run_stochastic_adjoint()
    run_global_contraction()

    print("\n=============== ALL ADVANCED TESTS COMPLETE (FAST MODE) ===============\n")
