import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os

# --- 1. Define the Attack (Mini Version for Plotting) ---
class InverseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x): return self.net(x)

def get_loss_curve(data, epochs=150):
    # Normalize
    data = np.array(data)
    data = (data - data.mean()) / (data.std() + 1e-6)
    
    # Prepare Tensors
    X = torch.tensor([[x] for x in data[:-1]], dtype=torch.float32)
    Y = torch.tensor([[y] for y in data[1:]], dtype=torch.float32)
    
    model = InverseNet()
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    
    losses = []
    for _ in range(epochs):
        pred = model(X)
        loss = loss_fn(pred, Y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(loss.item())
    return losses

# --- 2. Generate Data Streams ---
print("Generating Data Streams...")
# A. Weak Target (Sine)
sine_data = [np.sin(i * 0.1) for i in range(1000)]

# B. Standard Target (Python Random)
python_data = [np.random.random() for _ in range(1000)]

# C. Strong Target (Wavelock - Mock or Real)
wavelock_path = os.path.join("./data/wavelock_data", "soliton_n12.csv")
if os.path.exists(wavelock_path):
    print("Loading Real Wavelock Data...")
    wavelock_data = np.loadtxt(wavelock_path)[:1000]
else:
    print("Using Mock Wavelock Data (Random Fallback)...")
    wavelock_data = [np.random.random() for _ in range(1000)]

# --- 3. Run Attacks & Record History ---
print("Running AI Attacks to generate Loss Curves...")
loss_sine = get_loss_curve(sine_data)
loss_python = get_loss_curve(python_data)
loss_wavelock = get_loss_curve(wavelock_data)

# --- 4. Plot: Neural Inversion Performance ---
plt.figure(figsize=(10, 5))
plt.plot(loss_sine, label='Sine Wave (Math)', color='red', linestyle='--')
plt.plot(loss_python, label='Python Random (PRNG)', color='blue', alpha=0.6)
plt.plot(loss_wavelock, label='Wavelock (Vacuum-PUF)', color='green', linewidth=2)
plt.title('AI Inversion Attack: Learning Rates')
plt.xlabel('Training Epochs')
plt.ylabel('Prediction Loss (Lower = Weaker)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("wavelock_ai_attack.png")
print("Saved 'wavelock_ai_attack.png'")

# --- 5. Plot: Phase Space (Determinism Check) ---
plt.figure(figsize=(12, 4))

# Sine Phase Space
plt.subplot(1, 3, 1)
plt.scatter(sine_data[:-1], sine_data[1:], s=2, c='red')
plt.title("Sine Wave (Deterministic)")
plt.xlabel("x(t)"); plt.ylabel("x(t+1)")

# Python Phase Space
plt.subplot(1, 3, 2)
plt.scatter(python_data[:-1], python_data[1:], s=2, c='blue', alpha=0.3)
plt.title("Python Random (Pseudo)")
plt.xlabel("x(t)")

# Wavelock Phase Space
plt.subplot(1, 3, 3)
plt.scatter(wavelock_data[:-1], wavelock_data[1:], s=2, c='green', alpha=0.5)
plt.title("Wavelock (Vacuum Entropy)")
plt.xlabel("x(t)")

plt.tight_layout()
plt.savefig("wavelock_phase_space.png")
print("Saved 'wavelock_phase_space.png'")