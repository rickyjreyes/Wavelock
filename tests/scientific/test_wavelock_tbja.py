# ultimate_attack_tbja_cluster.py
# ============================================================
# WaveLock TBJA-Cluster:
#   Multi-GPU, multi-ψ0 true-backprop Jacobian inversion.
#   Each GPU runs an independent TBJA instance with a different ψ0.
# Requires: torch with CUDA.
# ============================================================

import os
import math
import hashlib
import cupy as cp
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    _serialize_commitment_v2,
)

# ----------------- Shared PDE evolution (Torch) -----------------

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

# ----------------- Commitment helper -----------------

def commit_np(psi_np):
    psi_cp = cp.asarray(psi_np)
    raw = _serialize_commitment_v2(psi_cp)
    return hashlib.sha256(raw).hexdigest()

# ----------------- Worker TBJA -----------------

def tbja_worker(rank, world_size, psi_target_np, target_hash, steps=2000, lr=5e-3, results_dict=None):
    if not torch.cuda.is_available():
        print(f"[GPU {rank}] CUDA not available, skipping.")
        return

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    side = psi_target_np.shape[0]

    psi_target = torch.tensor(psi_target_np, dtype=torch.float64, device=device)
    psi0 = torch.randn((side, side), dtype=torch.float64, device=device, requires_grad=True)

    opt = torch.optim.Adam([psi0], lr=lr)

    best_loss = float("inf")
    best_iter = -1

    for step in range(steps):
        opt.zero_grad()
        psi_star_hat = evolve_torch(psi0)
        loss = F.mse_loss(psi_star_hat, psi_target)
        loss.backward()
        opt.step()

        loss_val = float(loss.detach().cpu().numpy())
        if loss_val < best_loss:
            best_loss = loss_val
            best_iter = step

        if step % 100 == 0 or step == steps - 1:
            cur_np = psi_star_hat.detach().cpu().numpy()
            if commit_np(cur_np) == target_hash:
                print(f"[GPU {rank}] MATCH FOUND at step {step}, loss={loss_val}")
                if results_dict is not None:
                    results_dict[rank] = {
                        "matched": True,
                        "iteration": step,
                        "loss": loss_val,
                    }
                return

    print(f"[GPU {rank}] Finished. Best loss={best_loss} at iter {best_iter}")
    if results_dict is not None:
        results_dict[rank] = {
            "matched": False,
            "best_loss": best_loss,
            "best_iter": best_iter,
        }

# ----------------- Entry point -----------------

def run_tbja_cluster(n=6, steps=2000, lr=5e-3):
    kp = CurvatureKeyPair(n=n)
    psi_star = cp.asnumpy(kp.psi_star)
    target_hash = commit_np(psi_star)

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No CUDA devices. TBJA-Cluster cannot run.")
        return

    print(f"=== TBJA-Cluster: {world_size} GPUs, n={n}, steps={steps} ===")
    print(f"Target hash: {target_hash}")

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    results_dict = manager.dict()

    procs = []
    for rank in range(world_size):
        p = mp.Process(
            target=tbja_worker,
            args=(rank, world_size, psi_star, target_hash, steps, lr, results_dict),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print("\n=== TBJA-Cluster Results ===")
    for k, v in results_dict.items():
        print(f"GPU {k}: {v}")









if __name__ == "__main__":
    run_tbja_cluster(n=6, steps=2000, lr=5e-3)
