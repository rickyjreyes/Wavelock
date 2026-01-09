# tests/inn_attack.py
# ============================================================
# Neural Inversion Attack Module for WCT-Bench
# Wraps your original logic to test external RNG streams.
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ----------------- The Network (Same as your original) -----------------
class InverseNet(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        # Adapted for 1D streams (RNG testing) instead of 2D grids
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim), # Try to reconstruct input
        )

    def forward(self, x):
        return self.net(x)

# ----------------- The Attack Logic -----------------
def run_attack(stream_data, epochs=10):
    """
    Attempts to train a Neural Network to predict the next number in the stream.
    Low Loss = PREDICTABLE (FAIL)
    High Loss = RANDOM (PASS)
    """
    print(f"   [INN] Training Neural Inverter on {len(stream_data)} samples...")
    
    # Prepare Data (Split into X -> Y pairs for prediction)
    # We try to predict value[i] given value[i-1]
    data_x = []
    data_y = []
    for i in range(len(stream_data) - 1):
        data_x.append([stream_data[i]])
        data_y.append([stream_data[i+1]])
        
    # Convert to Tensor
    x_tensor = torch.tensor(data_x, dtype=torch.float32)
    y_tensor = torch.tensor(data_y, dtype=torch.float32)
    
    # Setup Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InverseNet(input_dim=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)

    # Train Loop
    final_loss = 0.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(x_tensor)
        loss = loss_fn(prediction, y_tensor)
        loss.backward()
        optimizer.step()
        
        final_loss = loss.item()
        if epoch % 2 == 0:
            print(f"      Epoch {epoch}: Loss = {final_loss:.5f}")

    # Verdict Logic
    print(f"   [INN] Final Prediction Loss: {final_loss:.5f}")
    
    if final_loss < 0.01:
        return "FAIL (AI learned the pattern)"
    else:
        return "PASS (AI could not learn)"

# Only run if called directly
if __name__ == "__main__":
    # Test with dummy data
    print("Testing with simple sine wave (Should FAIL)...")
    sine_data = [np.sin(i) for i in range(100)]
    print(run_attack(sine_data))