import numpy as np
import cupy as cp
import hashlib
import sys
import os
from scipy.fftpack import fft2, ifft2
from scipy.linalg import svd

# =====================================================================
# IMPORTS
# =====================================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wavelock.chain.WaveLock import (
    CurvatureKeyPair,
    laplacian,
    _serialize_commitment_v2,
)

# =====================================================================
# COMMITMENT HELPER
# =====================================================================

def commit(psi):
    """Return WaveLock commitment for any ψ-field."""
    psi = cp.asarray(psi, dtype=cp.float32)
    raw = _serialize_commitment_v2(psi)
    return hashlib.sha256(raw).hexdigest()

# =====================================================================
# 1. ROTATIONAL DIFFERENTIAL ATTACK
# =====================================================================

def rotational_attack(n=6):
    """
    Test if commitment is invariant under rotations.
    If ψ and rotate(ψ) produce same hash, collision found.
    """
    kp = CurvatureKeyPair(n=n)
    psi = cp.asnumpy(kp.psi_star)
    h0 = commit(psi)
    
    side = psi.shape[0]
    
    results = []
    for angle in range(1, 4):  # 90, 180, 270 degree rotations
        rotated = np.rot90(psi, k=angle)
        h_rot = commit(rotated)
        match = (h_rot == h0)
        results.append({
            "angle_90k": angle,
            "matched": match,
            "hash": h_rot[:16]
        })
        if match:
            return {"broken": True, "results": results}
    
    return {"broken": False, "results": results}

# =====================================================================
# 2. MEET-IN-THE-MIDDLE ATTACK
# =====================================================================

def evolve_step_torch(psi, dt=0.01, alpha=1.0, beta=1.0, gamma=0.3, eps=1e-12):
    """Single PDE step (torch-free, numpy version)."""
    L = (
        -4*psi
        + np.roll(psi, 1, axis=0)
        + np.roll(psi, -1, axis=0)
        + np.roll(psi, 1, axis=1)
        + np.roll(psi, -1, axis=1)
    )
    F = alpha*L - beta*(psi**3 - psi) - gamma*np.log(psi*psi + eps)
    return psi + dt * F

def meet_in_middle_attack(n=6, forward_steps=25, samples=5000):
    """
    Split PDE evolution at midpoint.
    Forward: ψ0 → ψ_mid
    Backward: ψ* → ψ_mid
    Look for collisions.
    """
    kp = CurvatureKeyPair(n=n)
    target = cp.asnumpy(kp.psi_star)
    target_hash = commit(target)
    
    side = target.shape[0]
    
    # Forward table: random ψ0 evolved halfway
    forward_table = {}
    for i in range(samples):
        psi0 = np.random.randn(side, side).astype(np.float64)
        psi_mid = psi0.copy()
        
        for _ in range(forward_steps):
            psi_mid = evolve_step_torch(psi_mid)
        
        # Hash the midpoint state
        mid_hash = commit(psi_mid)
        forward_table[mid_hash] = (i, psi_mid)
        
        if i % 1000 == 0:
            print(f"  [MITM Forward] Sample {i}/{samples}")
    
    # Backward: evolve target backward, look for match in forward table
    psi_back = target.copy()
    for step in range(forward_steps):
        psi_back = evolve_step_torch(psi_back)
        back_hash = commit(psi_back)
        
        if back_hash in forward_table:
            fwd_idx, fwd_mid = forward_table[back_hash]
            print(f"  [MITM] Collision found at step {step}, forward sample {fwd_idx}")
            return {
                "matched": True,
                "step": step,
                "forward_sample": fwd_idx
            }
    
    return {"matched": False, "samples": samples}

# =====================================================================
# 3. LINEAR APPROXIMATION ATTACK
# =====================================================================

def linear_approximation_attack(n=6, iterations=500, lr=1e-4):
    """
    Linearize the PDE (ignore cubic, log terms).
    Solve linear system: ψ_{t+1} = ψ_t + dt*α*Δψ
    Use solution as starting point for refinement.
    """
    kp = CurvatureKeyPair(n=n)
    target = cp.asnumpy(kp.psi_star)
    target_hash = commit(target)
    
    side = target.shape[0]
    
    # Linearized evolution: ψ_next = (I + dt*α*Δ)*ψ
    # Build the linear operator
    dt, alpha = 0.01, 1.0
    I = np.eye(side*side)
    
    # Laplacian as matrix
    L_mat = np.zeros((side*side, side*side))
    for i in range(side):
        for j in range(side):
            idx = i*side + j
            L_mat[idx, idx] = -4
            L_mat[idx, (i+1)%side*side + j] = 1
            L_mat[idx, (i-1)%side*side + j] = 1
            L_mat[idx, i*side + (j+1)%side] = 1
            L_mat[idx, i*side + (j-1)%side] = 1
    
    A = I + dt*alpha*L_mat
    
    # Evolve from random start using linear dynamics
    psi = np.random.randn(side, side).astype(np.float64)
    psi_flat = psi.flatten()
    
    for step in range(iterations):
        psi_flat = A @ psi_flat
        psi = psi_flat.reshape((side, side))
        
        if commit(psi) == target_hash:
            return {"matched": True, "iteration": step}
    
    return {"matched": False}

# =====================================================================
# 4. LOW-RANK APPROXIMATION ATTACK (SVD-based)
# =====================================================================

def lowrank_attack(n=6, max_rank=10):
    """
    Decompose target via SVD.
    Reconstruct at progressively higher ranks.
    Check if low-rank version collides.
    """
    kp = CurvatureKeyPair(n=n)
    target = cp.asnumpy(kp.psi_star)
    target_hash = commit(target)
    
    U, S, Vt = svd(target, full_matrices=False)
    
    results = []
    for rank in range(1, min(max_rank, len(S)) + 1):
        U_r = U[:, :rank]
        S_r = S[:rank]
        Vt_r = Vt[:rank, :]
        
        reconstructed = U_r @ np.diag(S_r) @ Vt_r
        h_rank = commit(reconstructed)
        match = (h_rank == target_hash)
        
        results.append({
            "rank": rank,
            "matched": match,
            "hash": h_rank[:16]
        })
        
        if match:
            return {"broken": True, "results": results}
    
    return {"broken": False, "results": results}

# =====================================================================
# 5. SYMMETRY EXPLOITATION (reflections)
# =====================================================================

def symmetry_attack(n=6):
    """
    Test various symmetries: rotation, reflection, transpose.
    If any symmetry preserves commitment, collision found.
    """
    kp = CurvatureKeyPair(n=n)
    psi = cp.asnumpy(kp.psi_star)
    h0 = commit(psi)
    
    results = []
    
    # Rotations (0, 90, 180, 270)
    for k in range(1, 4):
        sym_psi = np.rot90(psi, k=k)
        h_sym = commit(sym_psi)
        match = (h_sym == h0)
        results.append({"symmetry": f"rot90_k{k}", "matched": match})
        if match:
            return {"broken": True, "results": results}
    
    # Horizontal flip
    sym_psi = np.fliplr(psi)
    h_sym = commit(sym_psi)
    if h_sym == h0:
        return {"broken": True, "results": results + [{"symmetry": "fliplr", "matched": True}]}
    results.append({"symmetry": "fliplr", "matched": False})
    
    # Vertical flip
    sym_psi = np.flipud(psi)
    h_sym = commit(sym_psi)
    if h_sym == h0:
        return {"broken": True, "results": results + [{"symmetry": "flipud", "matched": True}]}
    results.append({"symmetry": "flipud", "matched": False})
    
    # Transpose
    sym_psi = psi.T
    h_sym = commit(sym_psi)
    if h_sym == h0:
        return {"broken": True, "results": results + [{"symmetry": "transpose", "matched": True}]}
    results.append({"symmetry": "transpose", "matched": False})
    
    return {"broken": False, "results": results}

# =====================================================================
# 6. DIFFERENTIAL CRYPTANALYSIS (simplified)
# =====================================================================

def differential_attack(n=6, trials=1000, delta=1e-6):
    """
    Track how small input differences propagate through evolution.
    Look for differential characteristics that collapse.
    """
    kp = CurvatureKeyPair(n=n)
    target = cp.asnumpy(kp.psi_star)
    target_hash = commit(target)
    
    side = target.shape[0]
    
    collisions = 0
    
    for trial in range(trials):
        psi0 = np.random.randn(side, side).astype(np.float64)
        
        # Perturbed version
        psi0_delta = psi0.copy()
        i, j = np.random.randint(0, side, 2)
        psi0_delta[i, j] += delta
        
        # Evolve both
        psi = psi0.copy()
        psi_delta = psi0_delta.copy()
        
        for _ in range(50):
            psi = evolve_step_torch(psi)
            psi_delta = evolve_step_torch(psi_delta)
        
        # Check if difference collapsed to exact collision
        if commit(psi) == commit(psi_delta):
            collisions += 1
        
        if trial % 100 == 0:
            print(f"  [Differential] Trial {trial}/{trials}")
    
    return {
        "collisions": collisions,
        "rate": collisions / trials,
        "matched": collisions > 0
    }





# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("\n=== WAVELOCK NASTIEST ATTACK SUITE ===\n")
    
    print("1) Rotational Differential Attack:")
    result = rotational_attack()
    print(result)
    
    print("\n2) Symmetry Exploitation Attack:")
    result = symmetry_attack()
    print(result)
    
    print("\n3) Low-Rank Approximation Attack:")
    result = lowrank_attack()
    print(result)
    
    print("\n4) Linear Approximation Attack:")
    result = linear_approximation_attack()
    print(result)
    
    print("\n5) Differential Cryptanalysis Attack:")
    result = differential_attack()
    print(result)
    
    print("\n6) Meet-in-the-Middle Attack (this may take a while):")
    print("    [Skipping for now due to computation time. Uncomment to run.]")
    # result = meet_in_middle_attack()
    # print(result)
    
    print("\n=== DONE ===\n")