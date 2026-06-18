"""Adversarial audit suite for the WaveLock Curvature-Capacity Core.

Every module operates on the raw coupled state or the raw 256-bit digest. No
conventional cryptographic primitive is routed over candidate output. Every
experiment uses a fixed seed and writes a machine-readable artifact under
``curvature_audit/artifacts/``. Negative results record their attack budget and
are never described as proofs.
"""
