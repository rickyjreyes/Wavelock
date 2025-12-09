# ultimate_attack_neural_inversion.py
# ============================================================
# Neural Inversion Attack:
#   Learn f_θ : ψ* → ψ0 from a dataset of seeds.
#   If WaveLock is structurally invertible, a small CNN should learn it.
# ============================================================

import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from wavelock.chain.WaveLock import CurvatureKeyPair

# ----------------- Dataset -----------------

class WaveLockInverseDataset(Dataset):
    def __init__(self, n=6, num_samples=512, seed_offset=0):
        self.n = n
        self.num_samples = num_samples
        self.seed_offset = seed_offset
        self.data = []
        self._generate()

    def _generate(self):
        for i in range(self.num_samples):
            kp = CurvatureKeyPair(n=self.n, seed=self.seed_offset + i)
            psi0 = cp.asnumpy(kp.psi_0).astype(np.float32)
            psi_star = cp.asnumpy(kp.psi_star).astype(np.float32)
            self.data.append((psi_star, psi0))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        psi_star, psi0 = self.data[idx]
        # shape: [1, H, W] for CNN
        return (
            torch.from_numpy(psi_star[None, ...]),
            torch.from_numpy(psi0[None, ...])
        )

# ----------------- Simple CNN Inverter -----------------

class InverseNet(nn.Module):
    def __init__(self, side):
        super().__init__()
        self.side = side
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1),
        )

    def forward(self, x):
        return self.net(x)

def run_neural_inversion_attack(n=6, train_samples=512, test_samples=128, epochs=20, batch_size=16, lr=1e-3):
    # Build datasets
    train_ds = WaveLockInverseDataset(n=n, num_samples=train_samples, seed_offset=0)
    test_ds  = WaveLockInverseDataset(n=n, num_samples=test_samples, seed_offset=10000)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    side = train_ds[0][0].shape[-1]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = InverseNet(side=side).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Train
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for psi_star, psi0 in train_loader:
            psi_star = psi_star.to(device)
            psi0 = psi0.to(device)

            opt.zero_grad()
            pred = model(psi_star)
            loss = F.mse_loss(pred, psi0)
            loss.backward()
            opt.step()
            total_loss += float(loss) * psi_star.size(0)

        avg_train_loss = total_loss / len(train_ds)

        # Test
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for psi_star, psi0 in test_loader:
                psi_star = psi_star.to(device)
                psi0 = psi0.to(device)
                pred = model(psi_star)
                loss = F.mse_loss(pred, psi0)
                total_test_loss += float(loss) * psi_star.size(0)
        avg_test_loss = total_test_loss / len(test_ds)

        print(f"[Epoch {epoch+1}/{epochs}] train_loss={avg_train_loss:.4e}, test_loss={avg_test_loss:.4e}")

    print("\n=== Neural Inversion Attack Result ===")
    print("If test_loss stays high (O(1) or larger), f_θ fails to learn an inverse → strong one-way evidence.")





if __name__ == "__main__":
    run_neural_inversion_attack()
