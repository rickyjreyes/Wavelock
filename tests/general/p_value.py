import cupy as cp
import numpy as np
import time

# ============================================================
#  GPU Observables (Aliasing-Safe, Fixed Set)
# ============================================================

def get_observables_gpu(batch_psi):
    """
    Compute observables for an entire batch.
    Input shape: (Batch_Size, H, W)
    """
    # 1. L2 Norm
    l2 = cp.linalg.norm(batch_psi, axis=(1, 2))
    
    # 2. Mean & Variance
    mean = cp.mean(batch_psi, axis=(1, 2))
    var  = cp.var(batch_psi, axis=(1, 2))
    
    # 3. Gradient Energy
    gy, gx = cp.gradient(batch_psi, axis=(1, 2))
    grad_energy = cp.sum(gx**2 + gy**2, axis=(1, 2))
    
    # 4. Entropy Proxy
    sq = batch_psi**2
    energy = cp.sum(sq, axis=(1, 2), keepdims=True) + 1e-12
    prob = sq / energy
    entropy = -cp.sum(prob * cp.log(prob + 1e-12), axis=(1, 2))
    
    # 5. Spectral Power
    spectrum = cp.abs(cp.fft.rfft2(batch_psi, axes=(1, 2)))
    low_freq = cp.sum(spectrum[:, :5, :5], axis=(1, 2)) 
    mid_freq = cp.sum(spectrum[:, 5:15, 5:15], axis=(1, 2))
    
    return {
        "L2_norm": l2,
        "mean": mean,
        "variance": var,
        "grad_energy": grad_energy,
        "entropy_proxy": entropy,
        "low_freq_power": low_freq,
        "mid_freq_power": mid_freq,
    }

# ============================================================
#  High-Resolution Extreme Perturbation Test
# ============================================================

def run_stress_test(psi_cpu, n_perturbations=250_000, batch_size=1_000):
    H, W = psi_cpu.shape
    print(f"\n[GPU] Starting HIGH-RES EXTREME Test (N={n_perturbations}, Grid={H}x{W})...")
    
    psi_star = cp.asarray(psi_cpu)
    base_obs = get_observables_gpu(psi_star[None, ...])
    
    samples = {k: cp.empty(n_perturbations) for k in base_obs.keys()}
    start_t = time.time()
    
    for i in range(0, n_perturbations, batch_size):
        current_n = min(batch_size, n_perturbations - i)
        
        batch = cp.tile(psi_star, (current_n, 1, 1))
        
        # Variable noise (primary stressor)
        scale = float(cp.random.uniform(1e-5, 1e-3))
        batch += cp.random.normal(0, scale, batch.shape)
        
        # Integer rolls (aliasing-free)
        max_shift = int(H * 0.1)
        sx = int(cp.random.randint(-max_shift, max_shift))
        sy = int(cp.random.randint(-max_shift, max_shift))
        batch = cp.roll(batch, shift=(sy, sx), axis=(1, 2))
        
        # Orthogonal rotations
        k_rot = int(cp.random.randint(0, 4))
        batch = cp.rot90(batch, k=k_rot, axes=(1, 2))
        
        # Flips
        if cp.random.random() > 0.5:
            batch = cp.flip(batch, axis=2)
        
        obs = get_observables_gpu(batch)
        for k in samples:
            samples[k][i:i+current_n] = obs[k]
        
        if i % 10000 == 0:
            print(f"   Processed {i + current_n}/{n_perturbations} perturbations...", end="\r")
    
    print(f"\n[GPU] Done in {time.time() - start_t:.2f}s")
    return base_obs, samples

# ============================================================
#  Analysis & Reporting (CORRECT NULL HYPOTHESIS)
# ============================================================
def analyze_results(base, samples, alpha=0.05):
    print("\n" + "=" * 72)
    print(f"{'OBSERVABLE':<20} | {'P-VALUE':<10} | STATUS")
    print("-" * 72)
    
    for k in base.keys():
        original = float(base[k][0])
        dist = samples[k]
        
        median = float(cp.median(dist))
        diffs = cp.abs(dist - median)
        deviation = abs(original - median)
        
        # P-Value: How "normal" is the original compared to the stressed versions?
        # 1.0 = Perfectly Normal (Robust)
        # 0.0 = Outlier (Broken)
        p_val = float(cp.mean(diffs >= deviation))
        p_val = max(p_val, 1.0 / len(dist))
        
        # CORRECTED GRADING SCALE
        if p_val > 0.90:
            status = "✅ ROBUST (Invariant)"
        elif p_val > 0.50:
            status = "⚠️ STABLE (Noise Drift)" # 0.58 falls here (Expected for L2)
        else:
            status = "❌ BROKEN (Sensitive)"
        
        print(f"{k:<20} | {p_val:.4f}     | {status}")

# ============================================================
#  Entry Point (256x256 High-Res Soliton)
# ============================================================

if __name__ == "__main__":
    H, W = 256, 256
    
    # Envelope chosen so signal dies before boundary
    x = np.linspace(-6, 6, W)
    y = np.linspace(-6, 6, H)
    X, Y = np.meshgrid(x, y)
    
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    print(f"Generating High-Res Soliton Model ({H}x{W})...")
    psi_cpu = np.exp(-R**2) * np.cos(12 * Theta)
    
    base, perturbed = run_stress_test(
        psi_cpu,
        n_perturbations=250_000,
        batch_size=1_000
    )
    
    analyze_results(base, perturbed)
import cupy as cp
import numpy as np
import time

# ============================================================
#  GPU Observables (Aliasing-Safe, Fixed Set)
# ============================================================

def get_observables_gpu(batch_psi):
    """
    Compute observables for an entire batch.
    Input shape: (Batch_Size, H, W)
    """
    # 1. L2 Norm (Conservation)
    l2 = cp.linalg.norm(batch_psi, axis=(1, 2))
    
    # 2. Mean & Variance (Distribution)
    mean = cp.mean(batch_psi, axis=(1, 2))
    var  = cp.var(batch_psi, axis=(1, 2))
    
    # 3. Gradient Energy (Topological Tightness)
    gy, gx = cp.gradient(batch_psi, axis=(1, 2))
    grad_energy = cp.sum(gx**2 + gy**2, axis=(1, 2))
    
    # 4. Entropy Proxy (Information Density)
    sq = batch_psi**2
    energy = cp.sum(sq, axis=(1, 2), keepdims=True) + 1e-12
    prob = sq / energy
    entropy = -cp.sum(prob * cp.log(prob + 1e-12), axis=(1, 2))
    
    # 5. Spectral Power (Frequency Fingerprint)
    spectrum = cp.abs(cp.fft.rfft2(batch_psi, axes=(1, 2)))
    low_freq = cp.sum(spectrum[:, :5, :5], axis=(1, 2)) 
    mid_freq = cp.sum(spectrum[:, 5:15, 5:15], axis=(1, 2))
    
    return {
        "L2_norm": l2,
        "mean": mean,
        "variance": var,
        "grad_energy": grad_energy,
        "entropy_proxy": entropy,
        "low_freq_power": low_freq,
        "mid_freq_power": mid_freq,
    }

# ============================================================
#  High-Resolution Extreme Perturbation Test
# ============================================================

def run_stress_test(psi_cpu, n_perturbations=250_000, batch_size=1_000):
    H, W = psi_cpu.shape
    print(f"\n[GPU] Starting HIGH-RES EXTREME Test (N={n_perturbations}, Grid={H}x{W})...")
    
    psi_star = cp.asarray(psi_cpu)
    base_obs = get_observables_gpu(psi_star[None, ...])
    
    samples = {k: cp.empty(n_perturbations) for k in base_obs.keys()}
    start_t = time.time()
    
    for i in range(0, n_perturbations, batch_size):
        current_n = min(batch_size, n_perturbations - i)
        
        batch = cp.tile(psi_star, (current_n, 1, 1))
        
        # 1. Variable Noise
        scale = float(cp.random.uniform(1e-5, 1e-3))
        batch += cp.random.normal(0, scale, batch.shape)
        
        # 2. Integer Rolls (Aliasing-Free)
        max_shift = int(H * 0.1)
        sx = int(cp.random.randint(-max_shift, max_shift))
        sy = int(cp.random.randint(-max_shift, max_shift))
        batch = cp.roll(batch, shift=(sy, sx), axis=(1, 2))
        
        # 3. Orthogonal Rotations
        k_rot = int(cp.random.randint(0, 4))
        batch = cp.rot90(batch, k=k_rot, axes=(1, 2))
        
        # 4. Flips
        if cp.random.random() > 0.5:
            batch = cp.flip(batch, axis=2)
        
        obs = get_observables_gpu(batch)
        for k in samples:
            samples[k][i:i+current_n] = obs[k]
        
        if i % 10000 == 0:
            print(f"   Processed {i + current_n}/{n_perturbations} perturbations...", end="\r")
    
    print(f"\n[GPU] Done in {time.time() - start_t:.2f}s")
    return base_obs, samples

# ============================================================
#  Analysis & Reporting (CORRECTED LOGIC)
# ============================================================

def analyze_results(base, samples, alpha=0.05):
    """
    Standard Stress Test Interpretation:
      P-Value ~ 1.0 -> Robust Invariant (PASS)
      P-Value < 0.05 -> Broken / Sensitive (FAIL)
    """
    print("\n" + "=" * 72)
    print(f"{'OBSERVABLE':<20} | {'P-VALUE':<10} | INTERPRETATION")
    print("-" * 72)
    
    for k in base.keys():
        original = float(base[k][0])
        dist = samples[k]
        
        median = float(cp.median(dist))
        diffs = cp.abs(dist - median)
        deviation = abs(original - median)
        
        p_val = float(cp.mean(diffs >= deviation))
        p_val = max(p_val, 1.0 / len(dist))
        
        # --- THE FIX: High P means Stability ---
        if p_val > (1.0 - alpha):
            status = "✅ ROBUST (Invariant)"
        elif p_val < alpha:
            status = "❌ BROKEN (Sensitive)"
        else:
            status = "⚠️ WEAK (Drifting)"
        
        print(f"{k:<20} | {p_val:.4f}     | {status}")

# ============================================================
#  Entry Point (256x256 High-Res)
# ============================================================

if __name__ == "__main__":
    H, W = 256, 256
    
    # Envelope ensures signal dies before boundary (prevents wrapping errors)
    x = np.linspace(-6, 6, W)
    y = np.linspace(-6, 6, H)
    X, Y = np.meshgrid(x, y)
    
    R = np.sqrt(X**2 + Y**2)
    Theta = np.arctan2(Y, X)
    
    print(f"Generating High-Res Soliton Model ({H}x{W})...")
    psi_cpu = np.exp(-R**2) * np.cos(12 * Theta)
    
    base, perturbed = run_stress_test(
        psi_cpu,
        n_perturbations=250_000,
        batch_size=1_000
    )
    
    analyze_results(base, perturbed)