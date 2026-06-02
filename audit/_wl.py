"""
WaveLock audit harness — thin wrappers over the consensus reference
implementation (wavelock/chain/Wavelock_numpy.py) and the seed→ψ0 derivation
(wavelock/chain/xof_init.py).

Nothing here modifies repo code. We import the exact functions the project
uses to produce a commitment, so our evidence binds to real behavior.

WaveLock(seed) pipeline (consensus / WLv3.1 path, use_xof_init=True):
    seed --SHAKE256(WL-PSI-INIT-v1)--> psi0 in [0,1)^(side x side)
    psi0 --50-step nonlinear PDE--> psi_star
    serialized = magic + json(header) + psi_star.tobytes('C') + pack('<4d', E)
    C = SHA256(serialized)        (primary leg of the dual hash)
"""
from __future__ import annotations

import hashlib
import os
import sys

import numpy as np

# Make the repo importable regardless of CWD.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from wavelock.chain import Wavelock_numpy as wl  # noqa: E402
from wavelock.chain.xof_init import derive_psi_zero  # noqa: E402

# Re-export the canonical parameters so scripts can vary them explicitly.
ALPHA = wl.alpha
BETA = wl.beta
THETA = wl.theta
EPSILON = wl.epsilon
DELTA = wl.delta
DT = wl._dt
STEPS = wl._steps
DAMPING = wl._damping


def side_for_n(n: int) -> int:
    return 2 ** max(1, n // 2)


def psi0_xof(seed, n: int) -> np.ndarray:
    """Consensus ψ0 derivation (SHAKE-256). seed: int|str|bytes."""
    s = side_for_n(n)
    return derive_psi_zero(seed, (s, s))


def psi0_legacy(seed: int, n: int) -> np.ndarray:
    """Legacy Mersenne-Twister ψ0 derivation (np.random.seed)."""
    s = side_for_n(n)
    np.random.seed(seed)
    return np.random.rand(s, s)


def evolve(
    psi0: np.ndarray,
    *,
    steps: int = STEPS,
    dt: float = DT,
    alpha: float = ALPHA,
    beta: float = BETA,
    theta: float = THETA,
    epsilon: float = EPSILON,
    delta: float = DELTA,
    damping: float = DAMPING,
    capture: bool = False,
):
    """Exact replica of Wavelock_numpy._evolve, with parameters exposed.

    Verified against wl._evolve in audit/_selfcheck (see assert_matches_repo).
    """
    psi = np.asarray(psi0, dtype=np.float64).copy()
    hist = [psi.copy()] if capture else None
    for _ in range(steps):
        lap = wl.laplacian(psi)
        fb = alpha * lap / (psi + epsilon * np.exp(-beta * psi ** 2))
        ent = theta * (psi * wl.laplacian(np.log(psi ** 2 + delta)))
        dpsi = dt * (fb - ent) - damping * psi
        psi = psi + dpsi
        if capture:
            hist.append(psi.copy())
    return (psi, hist) if capture else psi


def serialize(psi_star: np.ndarray, schema: str = wl.SCHEMA_V2) -> bytes:
    """Exact repo serialization (WLv2 body is identical across schemas here)."""
    return wl._serialize_commitment(psi_star, schema)


def commit_bytes(serialized: bytes) -> str:
    return hashlib.sha256(serialized).hexdigest()


def wavelock(seed, n: int = 4, *, legacy: bool = False, schema=None):
    """Full pipeline. Returns dict with psi0, psi_star, serialized, C, etc.

    Uses the repo's own CurvatureKeyPairV3 so that `C` is exactly what the
    project would publish (sanity-bound in assert_matches_repo).
    """
    schema = schema or (wl.SCHEMA_V2 if legacy else wl.SCHEMA_V3_SHAKE)
    psi0 = psi0_legacy(seed, n) if legacy else psi0_xof(seed, n)
    psi_star = evolve(psi0)
    serialized = serialize(psi_star, schema)
    C = commit_bytes(serialized)
    return {
        "seed": seed,
        "n": n,
        "legacy": legacy,
        "schema": schema,
        "psi0": psi0,
        "psi_star": psi_star,
        "serialized": serialized,
        "C": C,
    }


def assert_matches_repo(n: int = 4, seed: int = 42) -> None:
    """Prove our harness reproduces the project's published commitment byte-for-byte."""
    kp = wl.CurvatureKeyPairV3(n=n, seed=seed, use_xof_init=True)
    ours = wavelock(seed, n=n, legacy=False)
    # psi_star must match exactly
    assert np.array_equal(kp.psi_star, ours["psi_star"]), "psi_star mismatch vs repo"
    assert kp._serialized == ours["serialized"], "serialized bytes mismatch vs repo"
    repo_primary = kp.commitment.split(":")[1]
    assert repo_primary == ours["C"], "commitment mismatch vs repo"
    # And our standalone evolve() matches the repo's _evolve()
    assert np.array_equal(wl._serialize_commitment(kp._evolve(ours["psi0"], n), kp.schema),
                          ours["serialized"]), "evolve() mismatch vs repo _evolve"


if __name__ == "__main__":
    assert_matches_repo()
    print("audit/_wl.py: harness reproduces repo commitment byte-for-byte (n=4, seed=42)")
    d = wavelock(42, n=4)
    ps = d["psi_star"]
    print("psi_star min/max/mean/std:",
          float(ps.min()), float(ps.max()), float(ps.mean()), float(ps.std()))
    print("C =", d["C"])
