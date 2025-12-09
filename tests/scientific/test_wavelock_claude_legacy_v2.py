import numpy as np
import cupy as cp
import hashlib
import sys
import os
from scipy.linalg import eigh

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

def commit_raw_bytes(psi):
    """Return raw serialized bytes (before SHA256)."""
    psi = cp.asarray(psi, dtype=cp.float32)
    return _serialize_commitment_v2(psi)

# =====================================================================
# 1. SERIALIZATION COLLISION ATTACK
# =====================================================================

def serialization_collision_attack(n=6, trials=10000):
    """
    Find two different ψ-fields that serialize to identical bytes.
    This would explain the rank-8 match.
    """
    kp = CurvatureKeyPair(n=n)
    target = cp.asnumpy(kp.psi_star)
    target_bytes = commit_raw_bytes(target)
    target_hash = hashlib.sha256(target_bytes).hexdigest()
    
    side = target.shape[0]
    
    print("  [Serialization] Searching for byte-level collisions...")
    
    for trial in range(trials):
        # Generate random ψ
        psi_candidate = np.random.randn(side, side).astype(np.float32)
        candidate_bytes = commit_raw_bytes(psi_candidate)
        
        # Check if bytes match (hash collision would follow)
        if candidate_bytes == target_bytes:
            print(f"    BYTES MATCH at trial {trial}")
            return {
                "matched": True,
                "trial": trial,
                "collision_type": "serialization_bytes"
            }
        
        # Also check hash
        candidate_hash = hashlib.sha256(candidate_bytes).hexdigest()
        if candidate_hash == target_hash:
            print(f"    HASH MATCH at trial {trial}")
            # Check if bytes are different
            if candidate_bytes != target_bytes:
                print(f"    -> SHA256 collision (different bytes, same hash)")
                return {
                    "matched": True,
                    "trial": trial,
                    "collision_type": "sha256_collision"
                }
            else:
                return {
                    "matched": True,
                    "trial": trial,
                    "collision_type": "serialization_bytes"
                }
        
        if trial % 1000 == 0:
            print(f"    Trial {trial}/{trials}")
    
    return {
        "matched": False,
        "trials": trials
    }

# =====================================================================
# 2. PDE INVERSE ATTACK (Laplacian inversion)
# =====================================================================

def pde_inverse_attack(n=6, iterations=500, lr=1e-3):
    """
    Try to invert the PDE evolution.
    Given ψ_final, solve backward for ψ0.
    Uses gradient descent on Laplacian-based loss.
    """
    kp = CurvatureKeyPair(n=n)
    target = cp.asnumpy(kp.psi_star)
    target_hash = commit(target)
    
    side = target.shape[0]
    
    # Build discrete Laplacian matrix
    def laplacian_matrix(side):
        L = np.zeros((side*side, side*side))
        for i in range(side):
            for j in range(side):
                idx = i*side + j
                L[idx, idx] = -4
                L[idx, ((i+1)%side)*side + j] = 1
                L[idx, ((i-1)%side)*side + j] = 1
                L[idx, i*side + ((j+1)%side)] = 1
                L[idx, i*side + ((j-1)%side)] = 1
        return L
    
    L_mat = laplacian_matrix(side)
    
    # Try to solve: L * ψ0 ≈ target (inverse of evolution)
    try:
        psi0_guess = np.linalg.lstsq(L_mat, target.flatten(), rcond=None)[0].reshape((side, side))
    except:
        psi0_guess = np.random.randn(side, side)
    
    print("  [PDE Inverse] Optimizing ψ0 to invert evolution...")
    
    for iteration in range(iterations):
        # Forward evolution
        L_psi0 = L_mat @ psi0_guess.flatten()
        psi_evolved = L_psi0.reshape((side, side))
        
        # Loss: how far from target
        loss = np.mean((psi_evolved - target)**2)
        
        # Gradient: derivative of loss w.r.t. ψ0
        grad = 2 * L_mat.T @ (L_psi0 - target.flatten())
        
        # Update
        psi0_guess = psi0_guess.flatten() - lr * grad
        psi0_guess = psi0_guess.reshape((side, side))
        
        if iteration % 50 == 0:
            print(f"    Iteration {iteration}, loss={loss:.6f}")
        
        # Check if evolution of this ψ0 matches target
        if commit(psi_evolved) == target_hash:
            return {
                "matched": True,
                "iteration": iteration,
                "loss": float(loss)
            }
    
    return {
        "matched": False,
        "final_loss": float(loss)
    }

# =====================================================================
# 3. LAPLACIAN EIGENDECOMPOSITION ATTACK
# =====================================================================

def eigendecomposition_attack(n=6, max_eigenvecs=15):
    """
    Decompose target using Laplacian eigendecomposition.
    If ψ* has low-rank structure in eigenspace of Δ, reconstruct it.
    Similar to SVD attack but uses problem-specific structure.
    """
    kp = CurvatureKeyPair(n=n)
    target = cp.asnumpy(kp.psi_star)
    target_hash = commit(target)
    
    side = target.shape[0]
    
    # Build discrete Laplacian matrix
    L = np.zeros((side*side, side*side))
    for i in range(side):
        for j in range(side):
            idx = i*side + j
            L[idx, idx] = -4
            L[idx, ((i+1)%side)*side + j] = 1
            L[idx, ((i-1)%side)*side + j] = 1
            L[idx, i*side + ((j+1)%side)] = 1
            L[idx, i*side + ((j-1)%side)] = 1
    
    print("  [Eigendecomposition] Computing Laplacian eigenvectors...")
    
    # Compute eigendecomposition
    try:
        eigenvalues, eigenvectors = eigh(L)
        # Sort by magnitude
        idx = np.argsort(np.abs(eigenvalues))[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
    except Exception as e:
        print(f"    Eigendecomposition failed: {e}")
        return {"matched": False, "error": str(e)}
    
    target_flat = target.flatten()
    
    results = []
    
    # Project onto increasing numbers of eigenvectors
    for n_eigs in range(1, min(max_eigenvecs+1, len(eigenvalues))):
        V = eigenvectors[:, :n_eigs]
        coeffs = V.T @ target_flat
        reconstructed_flat = V @ coeffs
        reconstructed = reconstructed_flat.reshape((side, side))
        
        h_recon = commit(reconstructed)
        match = (h_recon == target_hash)
        
        results.append({
            "n_eigenvectors": n_eigs,
            "matched": match,
            "hash": h_recon[:16]
        })
        
        print(f"    n_eigs={n_eigs}: match={match}")
        
        if match:
            return {
                "broken": True,
                "results": results
            }
    
    return {
        "broken": False,
        "results": results
    }

# =====================================================================
# 4. BONUS: PRECISION BOUNDARY ATTACK (refined)
# =====================================================================

def precision_cascade_attack(n=6):
    """
    Test serialization at different precisions.
    Look for precision boundaries where commitment collapses.
    """
    kp = CurvatureKeyPair(n=n)
    target = cp.asnumpy(kp.psi_star)
    h_target = commit(target)
    
    results = []
    
    # Convert to different precisions and back
    precisions = [np.float64, np.float32, np.float16]
    
    for p in precisions:
        try:
            converted = np.asarray(target, dtype=p)
            converted_back = np.asarray(converted, dtype=np.float64)
            h_conv = commit(converted_back)
            match = (h_conv == h_target)
            
            results.append({
                "precision": str(p),
                "matched": match,
                "hash": h_conv[:16]
            })
        except Exception as e:
            results.append({
                "precision": str(p),
                "error": str(e)
            })
    
    # Also test quantization at different bit depths
    for bits in [8, 16, 24, 32]:
        try:
            # Quantize to N bits
            min_val, max_val = target.min(), target.max()
            quantized = np.round((target - min_val) / (max_val - min_val) * (2**bits - 1))
            dequantized = quantized / (2**bits - 1) * (max_val - min_val) + min_val
            
            h_quant = commit(dequantized)
            match = (h_quant == h_target)
            
            results.append({
                "quantization_bits": bits,
                "matched": match,
                "hash": h_quant[:16]
            })
        except Exception as e:
            results.append({
                "quantization_bits": bits,
                "error": str(e)
            })
    
    return {"results": results}




# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    print("\n=== WAVELOCK DEEP VULNERABILITY PROBES ===\n")
    
    print("1) Serialization Collision Attack:")
    result = serialization_collision_attack()
    print(result)
    
    print("\n2) PDE Inverse Attack:")
    result = pde_inverse_attack()
    print(result)
    
    print("\n3) Laplacian Eigendecomposition Attack:")
    result = eigendecomposition_attack()
    print(result)
    
    print("\n4) Precision Cascade Attack:")
    result = precision_cascade_attack()
    print(result)
    
    print("\n=== DONE ===\n")