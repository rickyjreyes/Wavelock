import numpy as np
from pathlib import Path
# Parameters from the WaveLock paper
alpha = 1.50
beta = 2.6e-3
theta = 1.0e-5
eps = 1.0e-12
delta = 1.0e-12
mu = 2.0e-5
dt = 0.1
T = 50
N = 32
seed = 12  # assumption used to create "n12" data
def laplacian_periodic(u: np.ndarray) -> np.ndarray:
    return (
        np.roll(u, 1, axis=0)
        + np.roll(u, -1, axis=0)
        + np.roll(u, 1, axis=1)
        + np.roll(u, -1, axis=1)
        - 4.0 * u
    )
# Assumption: InitField(s, N) uses a deterministic FP64 seeded standard-normal field.
# The paper specifies deterministic seeded initialization, but not the exact distribution.
rng = np.random.default_rng(seed)
psi = rng.standard_normal((N, N), dtype=np.float64)
for _ in range(T):
    L = laplacian_periodic(psi)
    fb = alpha * L / (psi + eps * np.exp(-beta * psi**2))
    ent = theta * psi * laplacian_periodic(np.log(psi**2 + delta))
    psi = psi + dt * (fb - ent) - mu * psi
outdir = Path("./data/wavelock_data")
outdir.mkdir(parents=True, exist_ok=True)
flat_path = outdir / "soliton_n12.csv"
matrix_path = outdir / "soliton_n12_matrix_32x32.csv"
meta_path = outdir / "soliton_n12_metadata.txt"
np.savetxt(flat_path, psi.ravel(), delimiter=",")
np.savetxt(matrix_path, psi, delimiter=",")
meta = f"""WaveLock PDE-generated terminal field
Assumption set:
- seed s = {seed}
- grid N = {N}
- iterations T = {T}
- dt = {dt}
- alpha = {alpha}
- beta = {beta}
- theta = {theta}
- eps = {eps}
- delta = {delta}
- mu = {mu}
Important:
The WaveLock paper specifies deterministic seeded initialization but does not specify
the exact InitField(s, N) distribution in the excerpt used here. This file uses:
InitField(s, N) = standard_normal((N, N), FP64) with NumPy default_rng(seed).
Files:
- soliton_n12.csv: flattened 1024-value terminal field for direct use in the attack script
- soliton_n12_matrix_32x32.csv: 32x32 terminal field
"""
meta_path.write_text(meta, encoding="utf-8")
print(f"Created: {flat_path}")
print(f"Created: {matrix_path}")
print(f"Created: {meta_path}")
print(f"Flattened length: {psi.size}")
print(f"min={psi.min():.6e}, max={psi.max():.6e}, mean={psi.mean():.6e}, std={psi.std():.6e}")
