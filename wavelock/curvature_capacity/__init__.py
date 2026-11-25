"""WaveLock Curvature-Capacity Core (CC-Core-v0) -- experimental candidate.

A PDE-native, hash-free candidate that co-evolves the Design A wave field with a
path-binding accumulator field so the digest commits to the ordered wave
*trajectory* (the curvature signature / wake), not only the terminal state.

EXPERIMENTAL. No security claim. See docs/WAVELOCK_CURVATURE_CAPACITY_SPEC.md
and docs/WAVELOCK_CURVATURE_CAPACITY_RESULTS.md. The historical Design A
primitive (wavelock.pde_hash) is untouched and remains the frozen baseline.
"""

from __future__ import annotations

from . import spec
from .optimized import cc_hash, trajectory_digest

__all__ = ["spec", "cc_hash", "trajectory_digest"]
