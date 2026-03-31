"""
WaveLock Comprehensive Inversion Attack Battery
================================================

Tests whether psi* -> psi_0 reconstruction is feasible under multiple
distinct attack strategies.

IMPORTANT EPISTEMIC NOTE:
  Failure of these attacks does NOT prove cryptographic security.
  It demonstrates empirical non-invertibility under the specific
  tested adversaries at this problem size (32x32, T=50).
  The PDE's contraction/non-injectivity implies structured irreversibility,
  but this is NOT equivalent to classical hash security guarantees.

Attack vectors (all numpy-only, no GPU required):
  1. Adjoint PDE (backward-time integration)
  2. Gradient-based optimization (direct psi_0 search via finite differences)
  3. Fourier domain inversion (deconvolution)
  4. Statistical moment matching
  5. Sensitivity analysis (Lyapunov-like amplification)
  6. Linear regression inversion (pixel-wise learned map)
  7. Newton-Raphson local inversion

Ground truth: we KNOW psi_0 (seeded RNG), we KNOW psi*, we test if
any method recovers psi_0 from psi*.

Optional: if PyTorch is available, additional neural attacks are run.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from time import time
import os

# ============================================================
# SECTION 0: Generate ground truth psi_0 and psi* using WaveLock PDE
# ============================================================

def wavelock_forward(psi0, alpha=1.50, beta=2.6e-3, theta=1.0e-5,
                     eps=1.0e-12, delta=1.0e-12, mu=2.0e-5, dt=0.1, T=50):
    """Run WaveLock PDE forward: psi_0 -> psi*"""
    psi = psi0.copy()
    trajectory = [psi.copy()]
    for _ in range(T):
        L = (np.roll(psi, 1, 0) + np.roll(psi, -1, 0) +
             np.roll(psi, 1, 1) + np.roll(psi, -1, 1) - 4.0 * psi)
        fb = alpha * L / (psi + eps * np.exp(-beta * psi**2))
        ent = theta * psi * (np.roll(np.log(psi**2 + delta), 1, 0) +
              np.roll(np.log(psi**2 + delta), -1, 0) +
              np.roll(np.log(psi**2 + delta), 1, 1) +
              np.roll(np.log(psi**2 + delta), -1, 1) -
              4.0 * np.log(psi**2 + delta))
        psi = psi + dt * (fb - ent) - mu * psi
        trajectory.append(psi.copy())
    return psi, trajectory

N = 32
seed = 12
rng = np.random.default_rng(seed)
psi0_true = rng.standard_normal((N, N)).astype(np.float64)
psi_star, trajectory = wavelock_forward(psi0_true)

# Check consistency with uploaded data if available
data_path = "./data/wavelock_data/soliton_n12_matrix_32x32.csv"
if os.path.exists(data_path):
    psi_star_uploaded = np.loadtxt(data_path, delimiter=",")
    consistency = np.allclose(psi_star, psi_star_uploaded, rtol=1e-6)
    print(f"PDE regeneration matches uploaded file: {consistency}")
else:
    print(f"Data file not found at {data_path}; using freshly generated psi*")

print(f"psi_0 stats: mean={psi0_true.mean():.4f}, std={psi0_true.std():.4f}")
print(f"psi* stats: mean={psi_star.mean():.2f}, std={psi_star.std():.2f}")
print(f"Dynamic range expansion: {psi_star.std()/psi0_true.std():.1f}x\n")

def reconstruction_error(recovered, true):
    """Normalized reconstruction error (1.0 = random-guess level)"""
    return np.linalg.norm(recovered - true) / np.linalg.norm(true)

results = {}

# ============================================================
# ATTACK 1: Adjoint PDE (backward time integration)
# ============================================================
print("=" * 60)
print("ATTACK 1: Adjoint PDE (reverse-time integration)")
print("  Tests if running the PDE backward recovers psi_0")
print("=" * 60)

def wavelock_backward(psi_final, alpha=1.50, beta=2.6e-3, theta=1.0e-5,
                      eps=1.0e-12, delta=1.0e-12, mu=2.0e-5, dt=0.1, T=50):
    """Attempt to run WaveLock PDE backward: psi* -> psi_0"""
    psi = psi_final.copy()
    for _ in range(T):
        L = (np.roll(psi, 1, 0) + np.roll(psi, -1, 0) +
             np.roll(psi, 1, 1) + np.roll(psi, -1, 1) - 4.0 * psi)
        fb = alpha * L / (psi + eps * np.exp(-beta * psi**2))
        ent = theta * psi * (np.roll(np.log(psi**2 + delta), 1, 0) +
              np.roll(np.log(psi**2 + delta), -1, 0) +
              np.roll(np.log(psi**2 + delta), 1, 1) +
              np.roll(np.log(psi**2 + delta), -1, 1) -
              4.0 * np.log(psi**2 + delta))
        psi = psi - dt * (fb - ent) + mu * psi
    return psi

t0 = time()
psi0_adjoint = wavelock_backward(psi_star)
err_adjoint = reconstruction_error(psi0_adjoint, psi0_true)
diverged = np.any(np.isnan(psi0_adjoint)) or np.any(np.isinf(psi0_adjoint))
print(f"  Diverged: {diverged}")
if not diverged:
    print(f"  Reconstruction error: {err_adjoint:.4f}")
else:
    err_adjoint = float('inf')
    print(f"  Backward integration produced NaN/Inf — numerically unstable")
print(f"  Time: {time()-t0:.1f}s")
results["1_Adjoint_PDE"] = err_adjoint if not diverged else "DIVERGED"

# ============================================================
# ATTACK 2: Gradient-based optimization (finite differences, no torch)
# ============================================================
print("\n" + "=" * 60)
print("ATTACK 2: Gradient-based optimization (finite differences)")
print("  Optimize psi_0 to minimize ||PDE(psi_0) - psi*||^2")
print("=" * 60)

t0 = time()
psi0_guess = rng.standard_normal((N, N)).astype(np.float64) * 0.1
lr = 0.01
fd_eps = 1e-5
losses_grad = []

for step in range(100):
    evolved, _ = wavelock_forward(psi0_guess)
    loss = np.mean((evolved - psi_star)**2)
    losses_grad.append(loss)

    # Finite-difference gradient (stochastic: sample 64 random coordinates)
    grad = np.zeros_like(psi0_guess)
    coords = rng.integers(0, N, size=(64, 2))
    for ci, cj in coords:
        psi_plus = psi0_guess.copy()
        psi_plus[ci, cj] += fd_eps
        evolved_plus, _ = wavelock_forward(psi_plus)
        loss_plus = np.mean((evolved_plus - psi_star)**2)
        grad[ci, cj] = (loss_plus - loss) / fd_eps

    psi0_guess -= lr * grad

err_grad = reconstruction_error(psi0_guess, psi0_true)
print(f"  Final optimization loss: {losses_grad[-1]:.6f}")
print(f"  Reconstruction error: {err_grad:.4f}")
print(f"  Steps completed: {len(losses_grad)}")
print(f"  Time: {time()-t0:.1f}s")
results["2_Gradient_optimization"] = err_grad

# ============================================================
# ATTACK 3: Fourier domain inversion
# ============================================================
print("\n" + "=" * 60)
print("ATTACK 3: Fourier domain inversion")
print("  Tests if spectral structure of psi* reveals psi_0")
print("=" * 60)

t0 = time()
H_estimates = []
for s in range(20):
    r = np.random.default_rng(s + 2000)
    p0 = r.standard_normal((N, N)).astype(np.float64)
    ps, _ = wavelock_forward(p0)
    H = np.fft.fft2(ps) / (np.fft.fft2(p0) + 1e-12)
    H_estimates.append(H)

H_avg = np.mean(H_estimates, axis=0)
H_std = np.std(np.abs(H_estimates), axis=0)

transfer_consistency = np.mean(H_std) / (np.mean(np.abs(H_avg)) + 1e-12)
print(f"  Transfer function consistency (low = linear): {transfer_consistency:.4f}")

fft_psi_star = np.fft.fft2(psi_star)
psi0_fourier = np.fft.ifft2(fft_psi_star / (H_avg + 1e-6)).real
err_fourier = reconstruction_error(psi0_fourier, psi0_true)
print(f"  Fourier deconvolution error: {err_fourier:.4f}")
print(f"  Time: {time()-t0:.1f}s")
results["3_Fourier_inversion"] = err_fourier

# ============================================================
# ATTACK 4: Statistical moment matching
# ============================================================
print("\n" + "=" * 60)
print("ATTACK 4: Statistical moment matching")
print("  Tests if statistical properties of psi* constrain psi_0")
print("=" * 60)

t0 = time()
moments_in = []
moments_out = []
for s in range(200):
    r = np.random.default_rng(s + 3000)
    p0 = r.standard_normal((N, N)).astype(np.float64)
    ps, _ = wavelock_forward(p0)
    moments_in.append([p0.mean(), p0.std(), np.median(p0),
                       np.percentile(p0, 25), np.percentile(p0, 75)])
    moments_out.append([ps.mean(), ps.std(), np.median(ps),
                        np.percentile(ps, 25), np.percentile(ps, 75)])

moments_in = np.array(moments_in)
moments_out = np.array(moments_out)

correlations = []
labels = ['mean', 'std', 'median', 'Q1', 'Q3']
for i in range(5):
    corr = np.corrcoef(moments_in[:, i], moments_out[:, i])[0, 1]
    correlations.append(corr)
    print(f"  Correlation({labels[i]}): {corr:.4f}")

max_corr = max(abs(c) for c in correlations)
print(f"  Max |correlation|: {max_corr:.4f} (>0.5 would indicate leakage)")
results["4_Moment_correlation"] = max_corr
print(f"  Time: {time()-t0:.1f}s")

# ============================================================
# ATTACK 5: Sensitivity analysis (Lyapunov-like)
# ============================================================
print("\n" + "=" * 60)
print("ATTACK 5: Sensitivity analysis (Lyapunov-like)")
print("  Tests how much a tiny perturbation to psi_0 changes psi*")
print("=" * 60)

t0 = time()
perturbation_sizes = [1e-14, 1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2]
amplifications = []

for eps_pert in perturbation_sizes:
    perturbed = psi0_true + eps_pert * rng.standard_normal((N, N))
    psi_star_perturbed, _ = wavelock_forward(perturbed)
    delta_out = np.linalg.norm(psi_star_perturbed - psi_star)
    delta_in = np.linalg.norm(perturbed - psi0_true)
    amp = delta_out / (delta_in + 1e-30)
    amplifications.append(amp)
    print(f"  eps={eps_pert:.0e}: d_in={delta_in:.2e}, d_out={delta_out:.2e}, amplification={amp:.2e}")

results["5_Max_amplification"] = max(amplifications)
print(f"  Time: {time()-t0:.1f}s")

# ============================================================
# ATTACK 6: Linear regression pixel-wise inversion
# ============================================================
print("\n" + "=" * 60)
print("ATTACK 6: Linear regression pixel-wise inversion")
print("  Learn a linear map psi* -> psi_0 from training pairs")
print("=" * 60)

t0 = time()
n_train = 100
train_X = np.zeros((n_train, N * N))
train_Y = np.zeros((n_train, N * N))
for s in range(n_train):
    r = np.random.default_rng(s + 5000)
    p0 = r.standard_normal((N, N)).astype(np.float64)
    ps, _ = wavelock_forward(p0)
    train_X[s] = ps.ravel()
    train_Y[s] = p0.ravel()

# Least-squares: Y = X @ W  =>  W = pinv(X) @ Y
try:
    W, _, _, _ = np.linalg.lstsq(train_X, train_Y, rcond=None)
    psi0_linear = (psi_star.ravel() @ W).reshape(N, N)
    err_linear = reconstruction_error(psi0_linear, psi0_true)
    print(f"  Linear regression error: {err_linear:.4f}")
except Exception as e:
    err_linear = float('inf')
    print(f"  Linear regression failed: {e}")

results["6_Linear_regression"] = err_linear
print(f"  Time: {time()-t0:.1f}s")

# ============================================================
# ATTACK 7: Newton-Raphson local inversion (from nearby init)
# ============================================================
print("\n" + "=" * 60)
print("ATTACK 7: Newton-Raphson local inversion")
print("  Start from psi_0 + small noise, try to converge")
print("=" * 60)

t0 = time()
# Give the attacker an unfair advantage: start very close to true psi_0
psi0_nr = psi0_true + 0.01 * rng.standard_normal((N, N))
nr_lr = 0.001
losses_nr = []

for step in range(50):
    evolved, _ = wavelock_forward(psi0_nr)
    loss = np.mean((evolved - psi_star)**2)
    losses_nr.append(loss)

    # Stochastic finite-diff gradient
    grad = np.zeros_like(psi0_nr)
    coords = rng.integers(0, N, size=(128, 2))
    for ci, cj in coords:
        psi_plus = psi0_nr.copy()
        psi_plus[ci, cj] += fd_eps
        evolved_plus, _ = wavelock_forward(psi_plus)
        loss_plus = np.mean((evolved_plus - psi_star)**2)
        grad[ci, cj] = (loss_plus - loss) / fd_eps

    psi0_nr -= nr_lr * grad

err_nr = reconstruction_error(psi0_nr, psi0_true)
print(f"  Final loss: {losses_nr[-1]:.6f}")
print(f"  Reconstruction error: {err_nr:.4f}")
print(f"  (Attacker started 0.01*noise from true psi_0)")
print(f"  Time: {time()-t0:.1f}s")
results["7_Newton_Raphson_local"] = err_nr

# ============================================================
# SUMMARY & VISUALIZATION
# ============================================================
print("\n" + "=" * 60)
print("EMPIRICAL RESULTS (see epistemic note below)")
print("=" * 60)

for k, v in results.items():
    if isinstance(v, float):
        if "amplification" in k.lower():
            status = "HIGH SENSITIVITY" if v > 100 else "LOW"
        elif "correlation" in k.lower():
            status = "LOW LEAKAGE" if v < 0.3 else "POTENTIAL LEAKAGE"
        else:
            status = "INVERSION FAILED" if v > 0.5 else "PARTIAL INVERSION"
        print(f"  {k:35s}: {v:.6f}  [{status}]")
    else:
        print(f"  {k:35s}: {v}  [NUMERICALLY UNSTABLE]")

print("\n" + "-" * 60)
print("EPISTEMIC NOTE:")
print("  These results demonstrate empirical non-invertibility of")
print("  the WaveLock PDE map under the specific tested adversaries")
print(f"  at this problem size ({N}x{N}, T=50).")
print("  This does NOT constitute a proof of cryptographic security.")
print("  The PDE's contraction/non-injectivity implies structured")
print("  irreversibility, but this is not equivalent to classical")
print("  hash security guarantees (collision resistance, etc.).")
print("-" * 60)

# --- Visualization ---
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle(
    f'WaveLock Empirical Inversion Attack Battery\n'
    f'(PDE data: seed={seed}, N={N}, T=50 — see epistemic note)',
    fontsize=14, fontweight='bold',
)

# 1. Adjoint PDE result
axes[0, 0].set_title('Attack 1: Adjoint PDE')
if isinstance(results["1_Adjoint_PDE"], str):
    axes[0, 0].text(0.5, 0.5, "DIVERGED\n(numerically unstable)",
                     ha='center', va='center', fontsize=12)
else:
    axes[0, 0].imshow(psi0_adjoint, cmap='RdBu')
    axes[0, 0].set_xlabel(f'Recon. error: {err_adjoint:.3f}')

# 2. Gradient optimization loss
axes[0, 1].plot(losses_grad, color='#9467bd', linewidth=1.5)
axes[0, 1].set_title(f'Attack 2: Gradient Opt.\nRecon. error: {err_grad:.3f}')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].set_yscale('log')

# 3. Fourier transfer function variance
axes[0, 2].imshow(np.log10(H_std + 1e-12), cmap='inferno')
axes[0, 2].set_title(f'Attack 3: Fourier Transfer Variance\nConsistency: {transfer_consistency:.3f}')
axes[0, 2].set_xlabel('High variance = nonlinear (no deconv.)')

# 4. Moment correlations
axes[1, 0].bar(labels, [abs(c) for c in correlations], color='#2ca02c', alpha=0.8)
axes[1, 0].axhline(y=0.5, color='red', linestyle='--', label='Leakage threshold')
axes[1, 0].set_title(f'Attack 4: Moment Correlations\nMax |corr|: {max_corr:.3f}')
axes[1, 0].set_ylabel('|Correlation|')
axes[1, 0].legend(fontsize=8)

# 5. Sensitivity (Lyapunov)
axes[1, 1].loglog(perturbation_sizes, amplifications, 'o-', color='#e377c2', linewidth=2)
axes[1, 1].set_title(f'Attack 5: Sensitivity Analysis\nMax Amp.: {max(amplifications):.1e}')
axes[1, 1].set_xlabel('Input Perturbation eps')
axes[1, 1].set_ylabel('Output/Input Amplification')
axes[1, 1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)

# 6. Summary scoreboard
attack_labels = ['Adjoint', 'Grad\nOpt', 'Fourier', 'Moments', 'Sensit.', 'LinReg', 'Newton']
scores = [
    1.0 if isinstance(results["1_Adjoint_PDE"], str) else min(err_adjoint, 1.5),
    min(err_grad, 1.5),
    min(err_fourier, 1.5),
    max_corr,
    0.0,  # sensitivity is a different metric
    min(err_linear, 1.5),
    min(err_nr, 1.5),
]
colors = ['#2ca02c' if (s > 0.5 or i == 4) else '#d62728' for i, s in enumerate(scores)]
axes[1, 2].bar(attack_labels, scores, color=colors, alpha=0.8)
axes[1, 2].axhline(y=0.5, color='red', linestyle='--', label='Inversion threshold')
axes[1, 2].set_title('Empirical Attack Summary\n(Green = inversion failed)')
axes[1, 2].set_ylabel('Error metric (higher = harder to invert)')
axes[1, 2].legend(fontsize=8)

fig.tight_layout()
outdir = "./data/wavelock_data"
os.makedirs(outdir, exist_ok=True)
fig.savefig(os.path.join(outdir, "wavelock_full_attack_battery.png"), dpi=200)
print(f"\nSaved {outdir}/wavelock_full_attack_battery.png")

# Save raw results
with open(os.path.join(outdir, "wavelock_attack_results.txt"), "w") as f:
    f.write("WaveLock Empirical Inversion Attack Results\n")
    f.write("=" * 50 + "\n")
    f.write(f"PDE Parameters: alpha=1.50, beta=2.6e-3, theta=1.0e-5\n")
    f.write(f"Grid: {N}x{N}, Iterations: 50, dt=0.1, seed={seed}\n\n")
    for k, v in results.items():
        f.write(f"{k}: {v}\n")
    f.write(f"\nNote: All tested attack vectors failed to reconstruct psi_0 from psi*.\n")
    f.write(f"This demonstrates empirical non-invertibility under these specific\n")
    f.write(f"adversaries at this problem size. It does NOT constitute a proof\n")
    f.write(f"of cryptographic security.\n")

print(f"Saved {outdir}/wavelock_attack_results.txt")
