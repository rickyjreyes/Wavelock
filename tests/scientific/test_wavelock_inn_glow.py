import os, sys, numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# ============================================================
# Dynamic WaveLock Import Bootstrap
# ============================================================
def find_wavelock_root(start_dir=None):
    if start_dir is None:
        start_dir = os.path.abspath(os.getcwd())
    for root, dirs, files in os.walk(start_dir):
        if "WaveLockCore.py" in files or "WaveLock.py" in files:
            return root
    parent = os.path.dirname(start_dir)
    if parent != start_dir:
        return find_wavelock_root(parent)
    return None

WL_ROOT = find_wavelock_root()
if WL_ROOT is None:
    raise RuntimeError("ERROR: Could not locate WaveLock codebase automatically.")

sys.path.insert(0, WL_ROOT)

try:
    import WaveLockCore as wl
except ImportError:
    try:
        import WaveLock as wl
    except ImportError:
        wl = None
        for fname in os.listdir(WL_ROOT):
            if fname.endswith(".py"):
                mod = __import__(fname[:-3])
                if hasattr(mod, "evolve_field") or hasattr(mod, "evolve"):
                    wl = mod
                    break

if wl is None:
    raise ImportError("FATAL: Could not import WaveLock module")


# ============================================================
# Deterministic seed field (WaveLock does NOT define these)
# ============================================================
def generate_seed_field(seed, n):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n)).astype(np.float64)


# ============================================================
# Surrogate PDE (fallback if WaveLock does not expose a PDE)
# ============================================================
DT = 0.10
ALPHA = 0.25
BETA = 0.05

def _laplacian_np(psi):
    psi = np.asarray(psi, dtype=np.float64)
    return (
        -4.0 * psi
        + np.roll(psi, 1, axis=0)
        + np.roll(psi, -1, axis=0)
        + np.roll(psi, 1, axis=1)
        + np.roll(psi, -1, axis=1)
    )

def _surrogate_step_np(psi):
    psi = np.asarray(psi, dtype=np.float64)
    lap = _laplacian_np(psi)
    nonlin = -BETA * (psi**3 - psi)
    return psi + DT * (ALPHA * lap + nonlin)

def _forward_pde_np(psi0, T=30):
    psi = np.asarray(psi0, dtype=np.float64)
    for _ in range(T):
        psi = _surrogate_step_np(psi)
    return psi


# ============================================================
# Unified evolution function (WaveLock or surrogate)
# ============================================================
def evolve_field_safe(psi0, T=30):
    psi0 = np.asarray(psi0, dtype=np.float64)

    # Use WaveLock if available
    if hasattr(wl, "evolve_field"):
        try:
            return np.asarray(wl.evolve_field(psi0.copy(), T), dtype=np.float64)
        except:
            return np.asarray(wl.evolve_field(psi0.copy()), dtype=np.float64)

    if hasattr(wl, "evolve"):
        try:
            return np.asarray(wl.evolve(psi0.copy(), T), dtype=np.float64)
        except:
            return np.asarray(wl.evolve(psi0.copy()), dtype=np.float64)

    # Otherwise use surrogate PDE
    return _forward_pde_np(psi0, T=T)


# ============================================================
# Glow-style Affine Coupling Layer (Requires ≥2 Channels)
# ============================================================
class AffineCoupling(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels // 2, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, channels, 3, padding=1),
        )

    def forward(self, x, reverse=False):
        x_a, x_b = x.chunk(2, dim=1)
        params = self.net(x_a)
        s, t = params.chunk(2, dim=1)

        if reverse:
            x_b = (x_b - t) * torch.exp(-s)
        else:
            x_b = x_b * torch.exp(s) + t

        return torch.cat([x_a, x_b], dim=1)


# ============================================================
# Tiny Glow Flow
# ============================================================
class MiniGlow(nn.Module):
    def __init__(self, channels=2, depth=3):
        super().__init__()
        self.layers = nn.ModuleList([
            AffineCoupling(channels) for _ in range(depth)
        ])

    def forward(self, x, reverse=False):
        if reverse:
            for layer in reversed(self.layers):
                x = layer(x, reverse=True)
            return x
        else:
            for layer in self.layers:
                x = layer(x, reverse=False)
            return x


# ============================================================
# Dataset
# ============================================================
def generate_dataset(num_samples=300, n=32, T=30):
    X0 = []
    XT = []
    for s in range(num_samples):
        psi0 = generate_seed_field(s, n)
        psiT = evolve_field_safe(psi0.copy(), T=T)
        X0.append(psi0)
        XT.append(psiT)
    X0 = torch.tensor(np.array(X0)).float().unsqueeze(1)
    XT = torch.tensor(np.array(XT)).float().unsqueeze(1)
    return X0, XT


# ============================================================
# Training Loop
# ============================================================
def train_invertible_model(epochs=20, batch=16, n=32, T=30):
    X0, XT = generate_dataset(300, n, T=T)
    train_set = TensorDataset(XT[:250], X0[:250])
    test_set  = TensorDataset(XT[250:], X0[250:])

    train_loader = DataLoader(train_set, batch_size=batch, shuffle=True)
    test_loader  = DataLoader(test_set, batch_size=batch)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    flow = MiniGlow(channels=2, depth=3).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)

    for ep in range(1, epochs + 1):
        flow.train()
        train_loss = 0.0

        for XT_batch, X0_batch in train_loader:
            XT_batch = XT_batch.to(device)
            X0_batch = X0_batch.to(device)

            # Duplicate 1-channel → 2 channels
            XT2 = XT_batch.repeat(1, 2, 1, 1)
            X02 = X0_batch.repeat(1, 2, 1, 1)

            X0_pred = flow(XT2, reverse=True)
            loss = torch.mean((X0_pred - X02) ** 2)

            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()

        # Evaluation
        flow.eval()
        test_loss = 0.0
        with torch.no_grad():
            for XT_batch, X0_batch in test_loader:
                XT2 = XT_batch.to(device).repeat(1, 2, 1, 1)
                X02 = X0_batch.to(device).repeat(1, 2, 1, 1)
                X0_pred = flow(XT2, reverse=True)
                test_loss += torch.mean((X0_pred - X02) ** 2).item()

        print(f"[Epoch {ep}] train_loss={train_loss:.4f}, test_loss={test_loss:.4f}")

    print("\n=== INN/GLOW ATTACK COMPLETE ===")
    print("Final test_loss:", test_loss)
    print("If test_loss remains O(1), inversion is NOT learnable.\n")

    return test_loss






# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    train_invertible_model(epochs=20, batch=16, n=32, T=30)
