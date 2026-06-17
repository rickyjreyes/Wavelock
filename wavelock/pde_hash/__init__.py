"""wavelock.pde_hash — WaveLock-PDE-256-v0 experimental PDE-native digest.

A hash-free candidate one-way compression function: a finite-field Allen-Cahn
reaction-diffusion PDE over F_p (p = 2**31 - 1) run as a sponge. NO conventional
cryptographic primitive (SHA/SHAKE/BLAKE/MD5/RIPEMD/HMAC/HKDF/AES/ChaCha/
SipHash/Argon2/scrypt/PBKDF) is imported or used anywhere in this package; a
forbidden-import test enforces this (see pde_audit/).

This is an EXPERIMENTAL research candidate with no formal security proof and no
claim of production suitability. See docs/PDE_HASH_SPEC.md and
docs/PDE_HASH_THREAT_MODEL.md.

The canonical ``pde_hash`` is the pure-Python reference. ``optimized.pde_hash``
is an independent NumPy implementation that must agree byte-for-byte.
"""

from __future__ import annotations

from . import spec
from .state import PDEState, initial_state
from .evolve import evolve, permute
from .absorb import absorb
from .squeeze import squeeze
from .reference import pde_hash

__all__ = [
    "spec",
    "PDEState",
    "initial_state",
    "evolve",
    "permute",
    "absorb",
    "squeeze",
    "pde_hash",
]
