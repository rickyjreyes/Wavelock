# tests/inn_attack_v2.py
# ============================================================
# Neural Inversion Attack Module (Aggressive Tuning)
# Purpose: Must be smart enough to crack a sine wave,
#          so its failure on Wavelock is meaningful.
# ============================================================

import torch
import torch.nn as nn
import numpy as np

class InverseNet(nn.Module):
    def __init__(self, input_dim=1):
        super().__init__()
        # Wider, deeper network to capture non-linear patterns quickly
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        return self.net(x)

def run_attack(stream_data, epochs=300): # Increased epochs significantly
    """
    Attempts to train a Neural Network to predict the next number.
    """
    print(f"   [INN] Training Neural Inverter on {len(stream_data)} samples...")
    
    # Normalize Data (Crucial for Neural Nets to converge)
    data = np.array(stream_data)
    mean = data.mean()
    std = data.std() + 1e-6
    data = (data - mean) / std
    
    # Prepare X -> Y pairs
    data_x = []
    data_y = []
    for i in range(len(data) - 1):
        data_x.append([data[i]])
        data_y.append([data[i+1]])
        
    x_tensor = torch.tensor(data_x, dtype=torch.float32)
    y_tensor = torch.tensor(data_y, dtype=torch.float32)
    
    # Setup Model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InverseNet(input_dim=1).to(device)
    # Higher learning rate for faster convergence on simple patterns
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01) 
    loss_fn = nn.MSELoss()
    
    x_tensor = x_tensor.to(device)
    y_tensor = y_tensor.to(device)

    # Train Loop
    final_loss = 1.0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        prediction = model(x_tensor)
        loss = loss_fn(prediction, y_tensor)
        loss.backward()
        optimizer.step()
        
        final_loss = loss.item()
        if epoch % 50 == 0:
            print(f"      Epoch {epoch}: Loss = {final_loss:.5f}")
            # Early stopping if it cracked it
            if final_loss < 0.005: 
                print(f"      [!] Pattern Cracked early at Epoch {epoch}")
                break

    # Verdict Logic
    print(f"   [INN] Final Prediction Loss: {final_loss:.5f}")
    
    # Threshold: If it can predict better than random guessing
    if final_loss < 0.1: 
        return "FAIL (AI learned the pattern)"
    else:
        return "PASS (AI could not learn)"

if __name__ == "__main__":
    # Test 1: Simple Sine Wave (Should FAIL / Be Cracked)
    print("Test 1: Sine Wave (Predictable)")
    sine_data = [np.sin(i * 0.1) for i in range(500)] # More samples, smooth curve
    print(run_attack(sine_data))
    
    print("-" * 30)
    
    # Test 2: Random Noise (Should PASS / Resist)
    print("Test 2: Random Noise (Unpredictable)")
    rand_data = [np.random.random() for _ in range(500)]
    print(run_attack(rand_data))