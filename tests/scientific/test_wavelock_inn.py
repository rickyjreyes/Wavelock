#!/usr/bin/env python3
# test_wavelock_inn.py / neural inversion benchmark
# ============================================================
# Neural Inversion Attack:
#   Learn f_θ : ψ* → ψ0 from a dataset of deterministic seeds.
#
# Updated benchmark version:
#   - no environment commands required
#   - internal fast/standard/deep profiles
#   - uses GPU torch when available
#   - optional on-disk dataset cache
#   - train/test metrics: MSE, MAE, R2, correlation
#   - baseline comparison against zero/mean predictor
#   - fail-closed only if model learns meaningful inverse
#   - RISK_METRICS_BEGIN/END for run_benchmarks.py
# ============================================================

import os
import sys
import time
import json
import math
import hashlib
import traceback
from pathlib import Path

# Keep CPU numeric libs under control when benchmark runner is parallel.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import numpy as np
import cupy as cp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

# ============================================================
# CONFIG — edit here only
# ============================================================

# "fast"     = smoke
# "standard" = several-minute benchmark
# "deep"     = stronger long validation
PROFILE = "standard"

PROFILE_CONFIG = {
    "fast": {
        "n": 6,
        "train_samples": 512,
        "test_samples": 128,
        "epochs": 20,
        "batch_size": 16,
        "lr": 1e-3,
        "width": 32,
        "depth": 4,
    },
    "standard": {
        "n": 6,
        "train_samples": 2_000,
        "test_samples": 500,
        "epochs": 60,
        "batch_size": 32,
        "lr": 8e-4,
        "width": 64,
        "depth": 6,
    },
    "deep": {
        "n": 8,
        "train_samples": 8_000,
        "test_samples": 2_000,
        "epochs": 120,
        "batch_size": 64,
        "lr": 5e-4,
        "width": 96,
        "depth": 8,
    },
}

# If R2 rises above this, the attack learned structure worth investigating.
# Prior sklearn attack had strongly negative score, so 0.10 is conservative.
R2_DANGER_THRESHOLD = 0.10

# Optional cap. None = full selected profile.
MAX_SECONDS = None

# Cache generated datasets to avoid regenerating WaveLock samples every run.
USE_CACHE = True

# Deterministic dataset offsets.
TRAIN_SEED_OFFSET = 0
TEST_SEED_OFFSET = 1_000_000

# ============================================================
# Path setup
# ============================================================

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[2]
CACHE_DIR = REPO_ROOT / "scripts" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ============================================================
# Import WaveLock
# ============================================================

print("[INFO] Importing WaveLock CurvatureKeyPair...")
try:
    from wavelock.chain.WaveLock import CurvatureKeyPair
    print("[INFO] Import successful.")
except Exception as e:
    print("[ERROR] Could not import WaveLock:", repr(e))
    traceback.print_exc()
    sys.exit(1)

# ============================================================
# Utilities
# ============================================================

def sync_all():
    try:
        cp.cuda.Stream.null.synchronize()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def array_to_np_float32(x):
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x).astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def sha_config(profile, cfg, kind):
    payload = json.dumps(
        {
            "profile": profile,
            "cfg": cfg,
            "kind": kind,
            "train_offset": TRAIN_SEED_OFFSET,
            "test_offset": TEST_SEED_OFFSET,
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def cache_path(profile, cfg, kind):
    return CACHE_DIR / f"wavelock_inverse_{kind}_{sha_config(profile, cfg, kind)}.npz"


def finite_guard_np(arr, label):
    if not np.all(np.isfinite(arr)):
        raise FloatingPointError(f"non-finite values detected in {label}")


# ============================================================
# Dataset generation
# ============================================================

def generate_inverse_arrays(n, num_samples, seed_offset):
    """
    Returns:
        X = psi_star [N, 1, H, W]
        Y = psi_0    [N, 1, H, W]
    """
    X = []
    Y = []

    print(f"[INFO] Generating {num_samples:,} WaveLock samples n={n}, seed_offset={seed_offset:,}")

    t0 = time.time()
    for i in range(num_samples):
        if i % max(1, num_samples // 10) == 0:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 and i > 0 else 0.0
            print(f"  - Sample {i:,}/{num_samples:,} | rate={rate:,.1f}/s", flush=True)

        kp = CurvatureKeyPair(n=n, seed=int(seed_offset + i), test_mode=True)

        psi0 = array_to_np_float32(kp.psi_0)
        psi_star = array_to_np_float32(kp.psi_star)

        finite_guard_np(psi0, f"psi0 sample {i}")
        finite_guard_np(psi_star, f"psi_star sample {i}")

        X.append(psi_star[None, ...])
        Y.append(psi0[None, ...])

    X = np.stack(X).astype(np.float32, copy=False)
    Y = np.stack(Y).astype(np.float32, copy=False)

    return X, Y


def load_or_generate(profile, cfg, kind, n, num_samples, seed_offset):
    path = cache_path(profile, cfg, kind)

    if USE_CACHE and path.exists():
        print(f"[INFO] Loading cached {kind} dataset: {path}")
        data = np.load(path)
        return data["X"].astype(np.float32, copy=False), data["Y"].astype(np.float32, copy=False)

    X, Y = generate_inverse_arrays(n=n, num_samples=num_samples, seed_offset=seed_offset)

    if USE_CACHE:
        print(f"[INFO] Saving {kind} dataset cache: {path}")
        np.savez_compressed(path, X=X, Y=Y)

    return X, Y


# ============================================================
# Model
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(4 if channels >= 4 else 1, channels)
        self.norm2 = nn.GroupNorm(4 if channels >= 4 else 1, channels)

    def forward(self, x):
        y = F.gelu(self.norm1(self.conv1(x)))
        y = self.norm2(self.conv2(y))
        return F.gelu(x + y)


class InverseNet(nn.Module):
    def __init__(self, width=64, depth=6):
        super().__init__()

        blocks = [
            nn.Conv2d(1, width, 3, padding=1),
            nn.GELU(),
        ]

        for _ in range(depth):
            blocks.append(ResidualBlock(width))

        blocks += [
            nn.Conv2d(width, width // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, 1, 3, padding=1),
        ]

        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)


# ============================================================
# Metrics
# ============================================================

def compute_regression_metrics(pred, target):
    """
    pred/target torch tensors on any device.
    """
    p = pred.detach().float().flatten()
    y = target.detach().float().flatten()

    mse = float(F.mse_loss(p, y).item())
    mae = float(torch.mean(torch.abs(p - y)).item())

    y_mean = torch.mean(y)
    ss_res = torch.sum((y - p) ** 2)
    ss_tot = torch.sum((y - y_mean) ** 2) + 1e-12
    r2 = float((1.0 - ss_res / ss_tot).item())

    p_center = p - torch.mean(p)
    y_center = y - y_mean
    corr = float(
        (torch.sum(p_center * y_center) / (
            torch.sqrt(torch.sum(p_center ** 2) + 1e-12)
            * torch.sqrt(torch.sum(y_center ** 2) + 1e-12)
        )).item()
    )

    return {
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "corr": corr,
    }


def evaluate_model(model, loader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for psi_star, psi0 in loader:
            psi_star = psi_star.to(device, non_blocking=True)
            psi0 = psi0.to(device, non_blocking=True)
            pred = model(psi_star)
            preds.append(pred.detach())
            targets.append(psi0.detach())

    pred = torch.cat(preds, dim=0)
    target = torch.cat(targets, dim=0)

    return compute_regression_metrics(pred, target)


def baseline_metrics(Y_test, device):
    y = torch.from_numpy(Y_test).to(device)
    zero = torch.zeros_like(y)
    mean_pred = torch.full_like(y, float(torch.mean(y).item()))

    return {
        "zero": compute_regression_metrics(zero, y),
        "mean": compute_regression_metrics(mean_pred, y),
    }


# ============================================================
# Attack
# ============================================================

def run_neural_inversion_attack(cfg, profile):
    n = cfg["n"]
    train_samples = cfg["train_samples"]
    test_samples = cfg["test_samples"]
    epochs = cfg["epochs"]
    batch_size = cfg["batch_size"]
    lr = cfg["lr"]
    width = cfg["width"]
    depth = cfg["depth"]

    print("\n====================================================")
    print("      WAVELOCK — CNN NEURAL INVERSION ATTACK        ")
    print("====================================================")
    print(f"[INFO] Profile: {profile}")
    print(f"[INFO] Config: {json.dumps(cfg, indent=2)}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Torch device: {device}")
    if torch.cuda.is_available():
        print(f"[INFO] CUDA device: {torch.cuda.get_device_name(0)}")

    start = time.time()

    X_train, Y_train = load_or_generate(
        profile, cfg, "train", n, train_samples, TRAIN_SEED_OFFSET
    )
    X_test, Y_test = load_or_generate(
        profile, cfg, "test", n, test_samples, TEST_SEED_OFFSET
    )

    print("[INFO] Dataset shapes:")
    print("      X_train:", X_train.shape)
    print("      Y_train:", Y_train.shape)
    print("      X_test :", X_test.shape)
    print("      Y_test :", Y_test.shape)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(Y_test))

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=0,
    )

    model = InverseNet(width=width, depth=depth).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    base = baseline_metrics(Y_test, device)
    print("[INFO] Baselines:")
    print(json.dumps(base, indent=2, sort_keys=True))

    best_test_r2 = -1e9
    best_test_mse = float("inf")
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        seen = 0

        for psi_star, psi0 in train_loader:
            psi_star = psi_star.to(device, non_blocking=True)
            psi0 = psi0.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            pred = model(psi_star)
            loss = F.mse_loss(pred, psi0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = psi_star.size(0)
            total_loss += float(loss.item()) * bs
            seen += bs

        scheduler.step()
        sync_all()

        avg_train_loss = total_loss / max(1, seen)
        test_metrics = evaluate_model(model, test_loader, device)

        if test_metrics["r2"] > best_test_r2:
            best_test_r2 = test_metrics["r2"]
            best_test_mse = test_metrics["mse"]
            best_epoch = epoch + 1

        print(
            f"[Epoch {epoch+1:03d}/{epochs}] "
            f"train_loss={avg_train_loss:.4e} "
            f"test_mse={test_metrics['mse']:.4e} "
            f"test_mae={test_metrics['mae']:.4e} "
            f"test_r2={test_metrics['r2']:.4f} "
            f"corr={test_metrics['corr']:.4f}"
        )

        if MAX_SECONDS is not None and (time.time() - start) >= MAX_SECONDS:
            print("[WARNING] MAX_SECONDS reached; stopping early.")
            break

    final_metrics = evaluate_model(model, test_loader, device)
    elapsed = time.time() - start

    learned_inverse = best_test_r2 >= R2_DANGER_THRESHOLD

    print("\n=== Neural Inversion Attack Result ===")
    print(f"Best test R2: {best_test_r2:.6f} at epoch {best_epoch}")
    print(f"Best test MSE: {best_test_mse:.6e}")
    print(f"Final metrics: {json.dumps(final_metrics, indent=2, sort_keys=True)}")

    if learned_inverse:
        print("[WARNING] CNN learned nontrivial inverse structure. Investigate.")
    else:
        print("[PASS] CNN failed to learn meaningful inverse. Good for one-way evidence.")

    metrics = {
        "test": "cnn_neural_inversion",
        "profile": profile,
        "matched": bool(learned_inverse),
        "collisions": 0,
        "forgeries": 0,
        "false_accepts": 0,
        "accepted": 0,
        "nan_detected": False,
        "learned_inverse": bool(learned_inverse),
        "r2_danger_threshold": float(R2_DANGER_THRESHOLD),
        "best_test_r2": float(best_test_r2),
        "best_test_mse": float(best_test_mse),
        "best_epoch": int(best_epoch),
        "final_metrics": final_metrics,
        "baseline_metrics": base,
        "elapsed_seconds": float(elapsed),
        "n": int(n),
        "train_samples": int(train_samples),
        "test_samples": int(test_samples),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "device": str(device),
    }

    print("\nRISK_METRICS_BEGIN")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("RISK_METRICS_END")

    return not learned_inverse


# ============================================================
# Main
# ============================================================

def main():
    if PROFILE not in PROFILE_CONFIG:
        print(f"[ERROR] Unknown PROFILE={PROFILE!r}. Valid profiles: {sorted(PROFILE_CONFIG)}")
        return False

    cfg = PROFILE_CONFIG[PROFILE]

    try:
        return run_neural_inversion_attack(cfg, PROFILE)
    except Exception as e:
        print("[ERROR] Neural inversion benchmark crashed:", repr(e))
        traceback.print_exc()

        metrics = {
            "test": "cnn_neural_inversion",
            "profile": PROFILE,
            "matched": False,
            "collisions": 0,
            "forgeries": 0,
            "false_accepts": 0,
            "accepted": 0,
            "nan_detected": False,
            "crashed": True,
            "error": repr(e),
        }

        print("\nRISK_METRICS_BEGIN")
        print(json.dumps(metrics, indent=2, sort_keys=True))
        print("RISK_METRICS_END")

        return False


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
