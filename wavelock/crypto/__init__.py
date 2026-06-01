"""WaveLock cryptographic constructions.

This package hosts the *asymmetric* WaveLock signature work. The flagship
construction is WaveLock-OTS (a Lamport/WOTS-style one-time signature whose
secret material is bound to a WaveLock ψ-state), implemented in
:mod:`wavelock.crypto.wavelock_ots`.

Unlike the legacy SIGv2 scheme in :mod:`wavelock.chain.WaveLock`, nothing in
this package ever requires the verifier to possess ψ★ in order to verify a
signature. See ``docs/WAVELOCK_OTS_DESIGN.md``.
"""

from .wavelock_ots import (  # noqa: F401
    WaveLockOTSError,
    OTSKeyReuseError,
    SCHEME,
    HASH_ALG,
    generate_ots_keypair,
    sign_ots,
    verify_ots,
    export_public_key,
    export_secret_key,
    load_public_key,
    load_secret_key,
)

__all__ = [
    "WaveLockOTSError",
    "OTSKeyReuseError",
    "SCHEME",
    "HASH_ALG",
    "generate_ots_keypair",
    "sign_ots",
    "verify_ots",
    "export_public_key",
    "export_secret_key",
    "load_public_key",
    "load_secret_key",
]
