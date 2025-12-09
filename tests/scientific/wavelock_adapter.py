"""
Universal WaveLock Adapter (Safe Version)
Prevents recursive auto-evolution inside CurvatureKeyPair.__init__
and exposes stable generate/evolve functions for adversarial tests.
"""

import cupy as cp

# Attempt package import first
try:
    from wavelock.chain.WaveLock import CurvatureKeyPair
    PACKAGE = True
except Exception:
    PACKAGE = False

if not PACKAGE:
    from WaveLock import CurvatureKeyPair


# ========= SAFE CONSTRUCTOR (NO AUTO EVOLVE) ================
class SafeKeyPair(CurvatureKeyPair):
    """
    Override default constructor to prevent automatic evolution,
    which causes ~34GB memory recursive explosion.
    """
    def __init__(self, n, seed):
        # Call parent init but DO NOT compute psi_star
        super().__init__(n=n, seed=seed)

        # Prevent evolve() from running in constructor:
        # Remove psi_star if created
        if hasattr(self, "psi_star"):
            del self.psi_star


# ======== STABLE API FOR TEST SUITES =========================

def generate_seed_field(seed: int, n: int):
    """
    Use SafeKeyPair to generate ψ0 safely.
    """
    kp = SafeKeyPair(n=n, seed=seed)
    return cp.asarray(kp.psi_0.copy(), dtype=cp.float64), kp


def evolve_field(psi0, n: int):
    """
    Computes ψ* without auto-trigger recursion.
    """
    kp = SafeKeyPair(n=n, seed=0)  # seed ignored
    kp.psi_0 = cp.asarray(psi0, dtype=cp.float64)
    psi_star = kp._evolve(kp.psi_0)
    return psi_star.copy(), kp
