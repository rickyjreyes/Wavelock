# ultimate_attack_tangent_space.py
# ============================================================
# Tangent-Space Collapse Analysis:
#   Approximate leading singular values of J = ∂ψ*/∂ψ0
#   using Hutchinson-style power iteration (J^T J on R^N).
# Requires: torch with CUDA (or CPU, slower).
# ============================================================

import torch
import numpy as np
import cupy as cp
from wavelock.chain.WaveLock import CurvatureKeyPair

# ---------- PDE evolution (same as TBJA) ----------

def laplacian_torch(x):
    return (
        -4 * x
        + torch.roll(x, 1, dims=0)
        + torch.roll(x, -1, dims=0)
        + torch.roll(x, 1, dims=1)
        + torch.roll(x, -1, dims=1)
    )

def evolve_torch(psi, T=50, dt=0.01, alpha=1.0, beta=1.0, gamma=0.3, eps=1e-12):
    for _ in range(T):
        L = laplacian_torch(psi)
        Fterm = alpha * L - beta * (psi**3 - psi) - gamma * torch.log(psi * psi + eps)
        psi = psi + dt * Fterm
    return psi

# ---------- J^T J power iteration ----------

def jtj_power_iteration(psi0, iters=30):
    """
    psi0: torch tensor, requires_grad=True, shape [H,W]
    returns approximate top singular value of J = ∂ψ*/∂ψ0
    """
    device = psi0.device
    H, W = psi0.shape
    N = H * W

    psi_star = evolve_torch(psi0)
    psi_star_flat = psi_star.view(-1)

    # initialize random vector v in output space
    v = torch.randn(N, dtype=torch.float64, device=device)
    v = v / torch.norm(v)

    for k in range(iters):
        # J^T v via VJP
        psi0.grad = None
        psi_star_flat.backward(v, retain_graph=True)
        jt_v = psi0.grad.view(-1)

        # now J (J^T v) via JVP
        psi0.grad = None
        with torch.no_grad():
            u = jt_v.clone()
        psi0.grad = None
        psi_star2 = evolve_torch(psi0)
        psi_star2_flat = psi_star2.view(-1)

        # J u: treat u as "input tangent" using autograd.functional.jvp
        # but we can approximate by backward on random scalar
        # Simpler: compute gradient of (ψ*, u) dot product
        psi0.grad = None
        dot = torch.dot(psi_star2_flat, u)
        dot.backward()
        j_jt_v = psi0.grad.view(-1)

        # update v ∝ J J^T v
        v = j_jt_v / (torch.norm(j_jt_v) + 1e-12)

    # Rayleigh quotient approximation of λ_max
    psi0.grad = None
    psi_star_flat = evolve_torch(psi0).view(-1)
    dot = torch.dot(psi_star_flat, v)
    dot.backward()
    jt_v = psi0.grad.view(-1)
    num = torch.dot(jt_v, jt_v)
    denom = torch.dot(v, v)
    lambda_max = (num / (denom + 1e-12)).item()
    sigma_max = np.sqrt(max(lambda_max, 0.0))
    return sigma_max

def run_tangent_space_analysis(n=4):
    # Small n=4 (16x16) to keep J manageable
    kp = CurvatureKeyPair(n=n)
    psi0_np = cp.asnumpy(kp.psi_0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    psi0 = torch.tensor(psi0_np, dtype=torch.float64, device=device, requires_grad=True)
    sigma_max = jtj_power_iteration(psi0, iters=20)

    print(f"=== Tangent-Space Collapse Analysis (n={n}) ===")
    print(f"Approximate top singular value σ_max ≈ {sigma_max}")
    print("If σ_max << 1 or many directions appear null, J is strongly contractive / rank-deficient.")

if __name__ == "__main__":
    run_tangent_space_analysis(n=4)
# ultimate_attack_tangent_space.py
# ============================================================
# Tangent-Space Collapse Analysis:
#   Approximate leading singular values of J = ∂ψ*/∂ψ0
#   using Hutchinson-style power iteration (J^T J on R^N).
# Requires: torch with CUDA (or CPU, slower).
# ============================================================

import torch
import numpy as np
import cupy as cp
from wavelock.chain.WaveLock import CurvatureKeyPair

# ---------- PDE evolution (same as TBJA) ----------

def laplacian_torch(x):
    return (
        -4 * x
        + torch.roll(x, 1, dims=0)
        + torch.roll(x, -1, dims=0)
        + torch.roll(x, 1, dims=1)
        + torch.roll(x, -1, dims=1)
    )

def evolve_torch(psi, T=50, dt=0.01, alpha=1.0, beta=1.0, gamma=0.3, eps=1e-12):
    for _ in range(T):
        L = laplacian_torch(psi)
        Fterm = alpha * L - beta * (psi**3 - psi) - gamma * torch.log(psi * psi + eps)
        psi = psi + dt * Fterm
    return psi

# ---------- J^T J power iteration ----------

def jtj_power_iteration(psi0, iters=30):
    """
    psi0: torch tensor, requires_grad=True, shape [H,W]
    returns approximate top singular value of J = ∂ψ*/∂ψ0
    """
    device = psi0.device
    H, W = psi0.shape
    N = H * W

    psi_star = evolve_torch(psi0)
    psi_star_flat = psi_star.view(-1)

    # initialize random vector v in output space
    v = torch.randn(N, dtype=torch.float64, device=device)
    v = v / torch.norm(v)

    for k in range(iters):
        # J^T v via VJP
        psi0.grad = None
        psi_star_flat.backward(v, retain_graph=True)
        jt_v = psi0.grad.view(-1)

        # now J (J^T v) via JVP
        psi0.grad = None
        with torch.no_grad():
            u = jt_v.clone()
        psi0.grad = None
        psi_star2 = evolve_torch(psi0)
        psi_star2_flat = psi_star2.view(-1)

        # J u: treat u as "input tangent" using autograd.functional.jvp
        # but we can approximate by backward on random scalar
        # Simpler: compute gradient of (ψ*, u) dot product
        psi0.grad = None
        dot = torch.dot(psi_star2_flat, u)
        dot.backward()
        j_jt_v = psi0.grad.view(-1)

        # update v ∝ J J^T v
        v = j_jt_v / (torch.norm(j_jt_v) + 1e-12)

    # Rayleigh quotient approximation of λ_max
    psi0.grad = None
    psi_star_flat = evolve_torch(psi0).view(-1)
    dot = torch.dot(psi_star_flat, v)
    dot.backward()
    jt_v = psi0.grad.view(-1)
    num = torch.dot(jt_v, jt_v)
    denom = torch.dot(v, v)
    lambda_max = (num / (denom + 1e-12)).item()
    sigma_max = np.sqrt(max(lambda_max, 0.0))
    return sigma_max

def run_tangent_space_analysis(n=4):
    # Small n=4 (16x16) to keep J manageable
    kp = CurvatureKeyPair(n=n)
    psi0_np = cp.asnumpy(kp.psi_0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    psi0 = torch.tensor(psi0_np, dtype=torch.float64, device=device, requires_grad=True)
    sigma_max = jtj_power_iteration(psi0, iters=20)

    print(f"=== Tangent-Space Collapse Analysis (n={n}) ===")
    print(f"Approximate top singular value σ_max ≈ {sigma_max}")
    print("If σ_max << 1 or many directions appear null, J is strongly contractive / rank-deficient.")









if __name__ == "__main__":
    run_tangent_space_analysis(n=4)
