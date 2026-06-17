"""wavelock.pde_hash — WaveLock-PDE-256-v0 experimental PDE-native digest.

A hash-free candidate one-way compression function: a finite-field polynomial
dynamical system derived from the algebraic form of the Allen-Cahn reaction-
diffusion equation over F_p (p = 2**31 - 1), iterated in a sponge-like
absorb/evolve/squeeze construction. The T-round state transformation is NOT
known to be bijective. NO conventional cryptographic primitive (SHA/SHAKE/
BLAKE/MD5/RIPEMD/HMAC/HKDF/AES/ChaCha/SipHash/Argon2/scrypt/PBKDF) is imported
or used anywhere in this package; a forbidden-import test enforces this (see
pde_audit/).

This is an EXPERIMENTAL research candidate with no formal security proof and no
claim of production suitability. See docs/PDE_HASH_SPEC.md and
docs/PDE_HASH_THREAT_MODEL.md.

The canonical ``pde_hash`` is the pure-Python reference. ``optimized.pde_hash``
is an independent NumPy implementation that must agree byte-for-byte.
"""

from __future__ import annotations

from . import spec
from .state import PDEState, initial_state
from .evolve import evolve, evolve_T
from .absorb import absorb
from .squeeze import squeeze
from .reference import pde_hash

__all__ = [
    "spec",
    "PDEState",
    "initial_state",
    "evolve",
    "evolve_T",
    "absorb",
    "squeeze",
    "pde_hash",
]
