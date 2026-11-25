"""WaveLock Curvature-Capacity Core CC-Core-v1 / Candidate B -- experimental.

Linear-injection revision of Candidate A (wavelock.curvature_capacity). Removes
Candidate A's proved generic 2-to-1 injection (ETA*u^2) by setting ETA=ZETA=0:

    j_B(u, v) = u + GAMMA*u*v = u*(1 + GAMMA*v)   (mod p)

This introduces a singular hyperplane at v = V_STAR = -GAMMA^{-1} mod p, analyzed
adversarially in docs/CC_CORE_V1_ALGEBRA.md. EXPERIMENTAL. No security claim. The
frozen Design A primitive (wavelock.pde_hash) and Candidate A are untouched.
"""

from __future__ import annotations

from . import spec
from .optimized import cc_hash, trajectory_digest

__all__ = ["spec", "cc_hash", "trajectory_digest"]
