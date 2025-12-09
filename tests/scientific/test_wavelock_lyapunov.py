# ultimate_attack_lyapunov_sync.py
# ============================================================
# Lyapunov–Perron Inversion Attempt:
#   Reverse-time chaos synchronization on WaveLock PDE.
#   Optimize ψ0 and intermediate ψ_t to match ψ* at final time.
# ============================================================

import torch
import torch.nn.functional as F
import cupy as cp
from wavelock.chain.WaveLock import CurvatureKeyPair

def laplacian_torch(x):
    return (
        -4 * x
        + torch.roll(x, 1, dims=0)
        + torch.roll(x, -1, dims=0)
        + torch.roll(x, 1, dims=1)
        + torch.roll(x, -1, dims=1)
    )

def evolve_step(psi, dt=0.01, alpha=1.0, beta=1.0, gamma=0.3, eps=1e-12):
    L = laplacian_torch(psi)
    Fterm = alpha * L - beta * (psi**3 - psi) - gamma * torch.log(psi * psi + eps)
    return psi + dt * Fterm

def run_lyapunov_perron_attack(n=6, T=50, outer_steps=1500, lr=1e-3):
    kp = CurvatureKeyPair(n=n)
    psi_star_np = cp.asnumpy(kp.psi_star)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    side = psi_star_np.shape[0]

    psi_star = torch.tensor(psi_star_np, dtype=torch.float64, device=device)

    # Initialize ψ0 and all intermediate ψ_t as independent parameters
    psi_traj = [
        torch.randn((side, side), dtype=torch.float64, device=device, requires_grad=True)
        for _ in range(T + 1)
    ]
    # psi_traj[0] ~ ψ0, psi_traj[T] ~ ψ*

    opt = torch.optim.Adam(psi_traj, lr=lr)

    for step in range(outer_steps):
        opt.zero_grad()

        # forward consistency loss: ψ_{t+1} ≈ evolve_step(ψ_t)
        cons_loss = 0.0
        for t in range(T):
            psi_pred = evolve_step(psi_traj[t])
            cons_loss = cons_loss + F.mse_loss(psi_pred, psi_traj[t + 1])

        # endpoint anchoring: ψ_T ≈ ψ*
        end_loss = F.mse_loss(psi_traj[-1], psi_star)

        loss = cons_loss + 10.0 * end_loss
        loss.backward()
        opt.step()

        if step % 100 == 0 or step == outer_steps - 1:
            print(f"[Lyapunov-Perron] step={step}, total_loss={float(loss):.4e}, "
                  f"cons={float(cons_loss):.4e}, end={float(end_loss):.4e}")

    print("\n=== Lyapunov–Perron Attack Result ===")
    print(f"Final total loss={float(loss):.6f}")
    print("If this cannot drive end_loss near 0, backward synchronization fails (good for one-wayness).")











if __name__ == "__main__":
    run_lyapunov_perron_attack()
