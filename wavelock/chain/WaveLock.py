from __future__ import annotations
from typing import Optional, Tuple
import hashlib, json, struct
import cupy as cp
from dataclasses import dataclass

# ===========================================================
#  WaveLock v4 — Unified Curvature System
#  DEFAULT: WLv2 (fast, stable, test-compatible)
# ===========================================================

# --------------------
# Physics parameters
# --------------------
alpha   = 1.50
beta    = 0.0026
theta   = 1.0e-5
epsilon = 1.0e-12
delta   = 1.0e-12

_dt      = 0.1
_steps   = 50
_damping = 0.00002

SCHEMA_V1 = "WLv1"
SCHEMA_V2 = "WLv2"
SCHEMA_V3 = "WLv3"
SCHEMA_V4 = "WLv4"

# ---------------------------
# Invariant Selector (WLv2 default)
# ---------------------------
DEFAULT_INVARIANTS = {
    "WCT_core": True,     # always on
    "Lyapunov": False,    # optional
    "Ricci":    False,
    "ScalarR":  False,
    "Holonomy": False
}

def _canonical_json(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")

# ---------------------------
# Kernel identity
# ---------------------------
KERNEL_VERSION = "WL-psi-001"

def _kernel_descriptor() -> dict:
    return {
        "kernel_version": KERNEL_VERSION,
        "alpha": float(alpha),
        "beta":  float(beta),
        "theta": float(theta),
        "epsilon": float(epsilon),
        "delta": float(delta),
    }

def _kernel_hash() -> str:
    desc_bytes = _canonical_json(_kernel_descriptor())
    return hashlib.sha256(desc_bytes).hexdigest()

# ===========================================================
#                    WCC Rails (restored)
# ===========================================================

@dataclass(frozen=True)
class WCCBounds:
    name: str
    max_steps: int
    max_cells: int
    max_avg_c_theta: float

PWCC_BOUNDS = WCCBounds("PWCC", 10_4, 256*256, 10.0)
NPWCC_BOUNDS = WCCBounds("NPWCC", 10_6, 1024*1024, 1.0e3)
QUANTUM_CLASSICAL_BOUND = 1.0e2

MAX_WCC_SPATIAL_DIM = 2
REQUIRE_POWER_OF_TWO_SIDE = True


def _float64_bytes(x_cp) -> bytes:
    arr = cp.asnumpy(cp.asarray(x_cp, dtype=cp.float64))
    return arr.ravel(order="C").tobytes()


def laplacian(x):
    return (-4.0 * x
            + cp.roll(x, +1, 0) + cp.roll(x, -1, 0)
            + cp.roll(x, +1, 1) + cp.roll(x, -1, 1))


def enforce_dimensional_lock(psi):
    psi = cp.asarray(psi)
    if psi.ndim != 2:
        raise ValueError(f"ψ must be 2D, got ndim={psi.ndim}.")
    nx, ny = psi.shape
    if nx != ny:
        raise ValueError("ψ must be square.")
    if REQUIRE_POWER_OF_TWO_SIDE:
        if nx <= 0 or (nx & (nx - 1)) != 0:
            raise ValueError("ψ side length must be power-of-two.")

# ===========================================================
#      Commitment Serialization (WLv1 / WLv2)
# ===========================================================

def _commit_header(psi):
    return {
        "schema": SCHEMA_V2,
        "dtype": "float64",
        "order": "C",
        "bc": "periodic",
        "shape": [int(x) for x in psi.shape],
        "alpha": float(alpha),
        "beta":  float(beta),
        "theta": float(theta),
        "epsilon": float(epsilon),
        "delta": float(delta),
        "kernel_version": KERNEL_VERSION,
        "kernel_hash": _kernel_hash(),
    }

# core WCT curvature functional
def _curvature_functional(psi) -> Tuple[float, float, float, float]:
    gx, gy = cp.gradient(psi)
    E_grad = float(cp.sum(gx*gx) + cp.sum(gy*gy))
    lap = laplacian(psi)
    feedback = alpha * lap / (psi + epsilon * cp.exp(-beta * psi**2))
    entropy_term = theta * (psi * laplacian(cp.log(psi**2 + delta)))
    E_fb  = float(cp.sum(feedback * feedback))
    E_ent = float(cp.sum(entropy_term * entropy_term))
    E_tot = float(E_grad + E_fb + E_ent)
    return E_grad, E_fb, E_ent, E_tot

def _serialize_commitment_v2(psi) -> bytes:
    H = _canonical_json(_commit_header(psi))
    E_grad, E_fb, E_ent, E_tot = _curvature_functional(psi)
    packed_E = struct.pack("<4d", E_grad, E_fb, E_ent, E_tot)
    return b"WLv2\0" + H + _float64_bytes(psi) + packed_E

def _serialize_commitment_v1(psi) -> bytes:
    gx, gy = cp.gradient(psi)
    curv = float(cp.sum(gx*gx) + cp.sum(gy*gy))
    return _float64_bytes(psi) + f"{curv:.17g}".encode("utf-8")

# ===========================================================
#          WCC curvature budget & classification
# ===========================================================

def wcc_curvature_budget(psi) -> float:
    E_grad, E_fb, _, _ = _curvature_functional(psi)
    return float(E_grad + E_fb)

def wcc_avg_curvature_budget(psi) -> float:
    psi = cp.asarray(psi, dtype=cp.float64)
    C = wcc_curvature_budget(psi)
    return C / float(psi.size)

def classify_wcc_run(steps: int, psi) -> str:
    psi = cp.asarray(psi)
    n_cells = int(psi.size)
    avg_c = wcc_avg_curvature_budget(psi)

    def within(b: WCCBounds) -> bool:
        return (
            steps <= b.max_steps
            and n_cells <= b.max_cells
            and avg_c <= b.max_avg_c_theta
        )

    if within(PWCC_BOUNDS): return "PWCC"
    if within(NPWCC_BOUNDS): return "NPWCC"
    return "OUT_OF_WCC"

def check_quantum_classical_bound(psi) -> bool:
    return wcc_avg_curvature_budget(psi) <= QUANTUM_CLASSICAL_BOUND

# ===========================================================
#               CurvatureKeyPair (WLv2 default)
# ===========================================================

class CurvatureKeyPair:
    def __init__(self, n: int, seed: Optional[int] = None,
                 invariants=None, use_v3=False, use_v4=False):
        self.n = n
        self.invariants = invariants or DEFAULT_INVARIANTS
        self.use_v3 = False   # A-mode: OFF
        self.use_v4 = False   # A-mode: OFF

        cp.random.seed(seed)
        side = 2 ** max(1, n // 2)
        self.psi_0 = cp.random.rand(side, side)
        enforce_dimensional_lock(self.psi_0)

        self.psi_star = self.evolve(self.psi_0, n)
        enforce_dimensional_lock(self.psi_star)

        v2_hex = hashlib.sha256(_serialize_commitment_v2(self.psi_star)).hexdigest()
        self.commitment = f"{SCHEMA_V2}:{v2_hex}"

    # evolution kernel
    def evolve(self, psi0, n: int):
        psi = cp.asarray(psi0, dtype=cp.float64).copy()
        for _ in range(_steps):
            lap = laplacian(psi)
            fb  = alpha * lap / (psi + epsilon * cp.exp(-beta * psi**2))
            ent = theta * (psi * laplacian(cp.log(psi**2 + delta)))
            dpsi = _dt * (fb - ent) - _damping * psi
            psi  = psi + dpsi
        return psi

    # signature payload (WLv2)
    def _sig_payload_v2(self, message: str) -> bytes:
        H = _canonical_json(_commit_header(self.psi_star))
        return (
            b"SIGv2\0" +
            message.encode("utf-8") +
            b"\0" +
            H +
            b"\0" +
            _float64_bytes(self.psi_star)
        )

    # signing
    def sign(self, message: str) -> str:
        c = str(self.commitment)
        if c.startswith("WLv1:") or (len(c) == 64):
            psi_bytes = cp.asnumpy(self.psi_star.ravel()).tobytes()
            return hashlib.sha256(message.encode("utf-8") + psi_bytes).hexdigest()
        return hashlib.sha256(self._sig_payload_v2(message)).hexdigest()

    # verifying
    def _curvature_hash(self, psi) -> str:
        c = str(self.commitment)
        is_v1 = (
            c.startswith("WLv1:") or
            (len(c) == 64 and all(ch in "0123456789abcdef" for ch in c.lower()))
        )
        payload = _serialize_commitment_v1(psi) if is_v1 else _serialize_commitment_v2(psi)
        return hashlib.sha256(payload).hexdigest()

    def verify(self, message: str, signature: str) -> bool:
        c = str(self.commitment)
        stored = c.split(":", 1)[1] if ":" in c else c

        # check commitment
        recomputed = self._curvature_hash(self.psi_star)
        if recomputed != stored:
            return False

        # check signature
        if c.startswith("WLv1:") or (len(c) == 64):
            psi_bytes = cp.asnumpy(self.psi_star.ravel()).tobytes()
            expected = hashlib.sha256(message.encode("utf-8") + psi_bytes).hexdigest()
        else:
            expected = hashlib.sha256(self._sig_payload_v2(message)).hexdigest()

        return expected == signature

# ===========================================================
# Load keys
# ===========================================================

def load_quantum_keys(path: str = "psi_keypair.json") -> CurvatureKeyPair:
    import json
    with open(path, "r") as f:
        data = json.load(f)

    n = int(data.get("n", 4))
    kp = CurvatureKeyPair(n=n, seed=None)

    kp.psi_0 = cp.asarray(data["psi_0"], dtype=cp.float64)
    kp.psi_star = cp.asarray(data["psi_star"], dtype=cp.float64)

    stored = data.get("commitment")
    if stored:
        kp.commitment = stored

    return kp

# ===========================================================
# symbolic_verifier (WLv2 default)
# ===========================================================

def symbolic_verifier(psi_candidate, reference_psi, keypair=None) -> bool:
    """
    WLv2 symbolic verification:
    True iff commitments match exactly.
    """
    h1 = hashlib.sha256(
        _serialize_commitment_v2(cp.asarray(psi_candidate, dtype=cp.float64))
    ).hexdigest()
    h2 = hashlib.sha256(
        _serialize_commitment_v2(cp.asarray(reference_psi, dtype=cp.float64))
    ).hexdigest()
    return h1 == h2

# ===========================================================
# Public API
# ===========================================================
__all__ = [
    "CurvatureKeyPair",
    "symbolic_verifier",
    "classify_wcc_run",
    "wcc_avg_curvature_budget",
    "wcc_curvature_budget",
    "check_quantum_classical_bound",
    "_steps",
]

# ===========================================================
#  Legacy API required by psi-core (compatibility only)
# ===========================================================

def generate_quantum_keys(n: int = 4, seed: int | None = None, path: str = "psi_keypair.json"):
    """
    Compatibility: older ψ-Core Stage-3 agents expect on-disk ψ-keypairs.

    Publicly harmless:
    • Just saves the ψ_0, ψ_star, and commitment already generated.
    • No new cryptography.
    • No privileged data.
    """
    kp = CurvatureKeyPair(n=n, seed=seed)

    data = {
        "n": n,
        "psi_0": cp.asnumpy(kp.psi_0).tolist(),
        "psi_star": cp.asnumpy(kp.psi_star).tolist(),
        "commitment": kp.commitment,
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return data


def save_quantum_keys(path: str, data: dict):
    """Compatibility shim."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_quantum_keys(path: str = "psi_keypair.json") -> CurvatureKeyPair:
    """
    Compatibility loader:
    Reconstruct CurvatureKeyPair from a saved JSON file.
    """
    with open(path, "r") as f:
        data = json.load(f)

    n = int(data.get("n", 4))

    kp = CurvatureKeyPair(n=n, seed=None)
    kp.psi_0 = cp.asarray(data["psi_0"], dtype=cp.float64)
    kp.psi_star = cp.asarray(data["psi_star"], dtype=cp.float64)

    if "commitment" in data:
        kp.commitment = data["commitment"]

    return kp
