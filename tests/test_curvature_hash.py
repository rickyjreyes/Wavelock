import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = np
from wavelock.chain.WaveLock import CurvatureKeyPair

# def test_curvature_hash_sensitivity():
#     print("🔬 Testing curvature hash sensitivity...")

#     keypair = CurvatureKeyPair(n=4, seed=123)
#     original_hash = keypair._curvature_hash(keypair.psi_star)
#     print(f"Original curvature hash: {original_hash}")

#     # Perturb psi_star slightly
#     perturbed_psi = keypair.psi_star.copy()
#     perturbed_psi += cp.random.normal(0, 1e-4, size=perturbed_psi.shape)

#     perturbed_hash = keypair._curvature_hash(perturbed_psi)
#     print(f"Perturbed curvature hash: {perturbed_hash}")

#     if original_hash != perturbed_hash:
#         print("✅ Hash sensitivity confirmed. Curvature fingerprint changed.")
#     else:
#         print("❌ WARNING: Hash did not change. Possible robustness issue.")


def test_curvature_hash_sensitivity():
    """
    Curvature hash must be sensitive to infinitesimal perturbations in ψ★.
    This test explicitly opts into ψ★ access via test_mode.
    """

    keypair = CurvatureKeyPair(n=4, seed=123, test_mode=True)

    original_psi = keypair.psi_star
    original_hash = keypair._curvature_hash(original_psi)

    # Perturb ψ★ slightly
    perturbed_psi = original_psi.copy()
    perturbed_psi += cp.random.normal(0.0, 1e-4, size=perturbed_psi.shape)

    perturbed_hash = keypair._curvature_hash(perturbed_psi)

    # ASSERT — do not print
    assert original_hash != perturbed_hash, (
        "Curvature hash is insensitive to ψ★ perturbation"
    )
    
if __name__ == "__main__":
    test_curvature_hash_sensitivity()
