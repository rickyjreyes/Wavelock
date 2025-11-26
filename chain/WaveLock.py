from __future__ import annotations
from typing import Optional, Tuple
import hashlib, json, struct
import cupy as cp
from dataclasses import dataclass

# --- physics / evolution params (must match between sign & verify) ---
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


def _canonical_json(obj) -> bytes:
    return json.dumps(obj, separators=(",", ":"), sort_keys=True).encode("utf-8")


# --- kernel identity (versioned & hashed) ---

# Bump this whenever you change the evolution rule or physics invariants
KERNEL_VERSION = "WL-psi-001"


def _kernel_descriptor() -> dict:
    """
    Stable descriptor of the WaveLock kernel.

    Only include fields that are intended to be identical for a given kernel
    version across machines. Local environment details (python version, CUDA,
    etc.) are *not* included here.
    """
    return {
        "kernel_version": KERNEL_VERSION,
        "alpha": float(alpha),
        "beta": float(beta),
        "theta": float(theta),
        "epsilon": float(epsilon),
        "delta": float(delta),
    }


def _kernel_hash() -> str:
    """
    Canonical hash of the kernel descriptor. This is what gets embedded in
    the WaveLock header and in KERNEL_DECL blocks on-chain.
    """
    desc_bytes = _canonical_json(_kernel_descriptor())
    return hashlib.sha256(desc_bytes).hexdigest()


# --- WCC / resource bounds (model-relative rails) ---

@dataclass(frozen=True)
class WCCBounds:
    """
    Model-relative resource bounds for Wave Curvature Computation (WCC).

    - max_steps:       maximum evolution steps allowed
    - max_cells:       maximum lattice size (number of sites)
    - max_avg_c_theta: maximum average curvature budget per site
                       (C_Theta / N_cells, using E_grad+E_fb)
    """
    name: str
    max_steps: int
    max_cells: int
    max_avg_c_theta: float


# "Polynomial" WCC rail (PWCC): strict, conservative bounds
PWCC_BOUNDS = WCCBounds(
    name="PWCC",
    max_steps=10_4,            # can be tuned; should be poly in input size
    max_cells=256 * 256,       # moderate lattice
    max_avg_c_theta=10.0,      # average curvature budget per site
)

# "NP" WCC rail (NPWCC): looser, allows more curvature/time
NPWCC_BOUNDS = WCCBounds(
    name="NPWCC",
    max_steps=10_6,
    max_cells=1024 * 1024,
    max_avg_c_theta=1.0e3,
)

# --- quantum–classical crossover bound (model-relative) ---
# Above this average curvature budget, we treat the configuration as
# effectively "classical" rather than coherently quantum.
QUANTUM_CLASSICAL_BOUND = 1.0e2  # tune per your theory

# --- dimensional locking (n <= 3; here explicit 2D lattice rail) ---
MAX_WCC_SPATIAL_DIM = 2            # we only allow 2D ψ-lattices here
REQUIRE_POWER_OF_TWO_SIDE = True   # enforce 2^k x 2^k to mimic spectral banding


def _float64_bytes(x_cp) -> bytes:
    arr = cp.asnumpy(cp.asarray(x_cp, dtype=cp.float64))
    return arr.ravel(order="C").tobytes()


def laplacian(x):
    return (-4.0 * x
            + cp.roll(x, +1, 0) + cp.roll(x, -1, 0)
            + cp.roll(x, +1, 1) + cp.roll(x, -1, 1))


def enforce_dimensional_lock(psi):
    """
    Enforce the WCT dimensional-locking rail for this kernel:

    - ψ must be a 2D array (spatial dimension <= 2 here, with time external).
    - lattice must be square.
    - optionally, side length must be a power of two (2^k).
    """
    psi = cp.asarray(psi)
    if psi.ndim != MAX_WCC_SPATIAL_DIM:
        raise ValueError(f"Dimensional lock violation: ψ has ndim={psi.ndim}, expected {MAX_WCC_SPATIAL_DIM}.")
    nx, ny = psi.shape
    if nx != ny:
        raise ValueError(f"Dimensional lock violation: ψ shape={psi.shape}, expected square.")
    if REQUIRE_POWER_OF_TWO_SIDE:
        side = int(nx)
        # power-of-two check: side & (side - 1) == 0 for side>0
        if side <= 0 or (side & (side - 1)) != 0:
            raise ValueError(
                f"Dimensional lock violation: side={side} is not a power of two."
            )


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
        # explicit kernel identity
        "kernel_version": KERNEL_VERSION,
        "kernel_hash": _kernel_hash(),
    }


def _curvature_functional(psi) -> Tuple[float, float, float, float]:
    gx, gy = cp.gradient(psi)
    E_grad = float(cp.sum(gx * gx) + cp.sum(gy * gy))
    lap = laplacian(psi)
    feedback = alpha * lap / (psi + epsilon * cp.exp(-beta * psi**2))
    entropy_term = theta * (psi * laplacian(cp.log(psi**2 + delta)))
    E_fb   = float(cp.sum(feedback * feedback))
    E_ent  = float(cp.sum(entropy_term * entropy_term))
    E_tot  = float(E_grad + E_fb + E_ent)
    return E_grad, E_fb, E_ent, E_tot


def wcc_curvature_budget(psi) -> float:
    """
    Compute the global curvature budget C_Theta for a given ψ configuration,
    using the discrete analogue of
        C_Theta ~ ∑ (|∇ψ|^2 + |Θ[ψ]|^2) ΔV
    here approximated as E_grad + E_fb.
    """
    psi = cp.asarray(psi, dtype=cp.float64)
    E_grad, E_fb, _, _ = _curvature_functional(psi)
    return float(E_grad + E_fb)


def wcc_avg_curvature_budget(psi) -> float:
    """
    Average curvature budget per site: C_Theta / N_cells.
    """
    psi = cp.asarray(psi, dtype=cp.float64)
    C_theta = wcc_curvature_budget(psi)
    return C_theta / float(psi.size)


def classify_wcc_run(steps: int, psi) -> str:
    """
    Classify an evolution run into PWCC / NPWCC / OUT_OF_WCC based on:

    - number of evolution steps used
    - lattice size
    - average curvature budget per site

    This doesn't change behavior; it just makes the rails explicit.
    """
    psi = cp.asarray(psi, dtype=cp.float64)
    n_cells = int(psi.size)
    avg_c = wcc_avg_curvature_budget(psi)

    def within(bounds: WCCBounds) -> bool:
        return (
            steps <= bounds.max_steps
            and n_cells <= bounds.max_cells
            and avg_c <= bounds.max_avg_c_theta
        )

    if within(PWCC_BOUNDS):
        return "PWCC"
    if within(NPWCC_BOUNDS):
        return "NPWCC"
    return "OUT_OF_WCC"


def check_quantum_classical_bound(psi) -> bool:
    """
    Return True iff the average curvature budget per site is below the
    quantum–classical crossover bound. Above this, we treat the configuration
    as effectively classical for this kernel.
    """
    avg_c = wcc_avg_curvature_budget(psi)
    return avg_c <= QUANTUM_CLASSICAL_BOUND


def _serialize_commitment_v2(psi) -> bytes:
    H = _canonical_json(_commit_header(psi))
    E_grad, E_fb, E_ent, E_tot = _curvature_functional(psi)
    packed_E = struct.pack("<4d", E_grad, E_fb, E_ent, E_tot)
    return b"WLv2\0" + H + _float64_bytes(psi) + packed_E


def _serialize_commitment_v1(psi) -> bytes:
    gx, gy = cp.gradient(psi)
    curv = float(cp.sum(gx * gx) + cp.sum(gy * gy))
    return _float64_bytes(psi) + f"{curv:.17g}".encode("utf-8")


class CurvatureKeyPair:
    def __init__(self, n: int, seed: Optional[int] = None):
        self.n = n
        cp.random.seed(seed)
        side = 2 ** max(1, n // 2)
        self.psi_0 = cp.random.rand(side, side)
        enforce_dimensional_lock(self.psi_0)      # explicit dim lock
        self.psi_star = self.evolve(self.psi_0, n)
        enforce_dimensional_lock(self.psi_star)   # sanity after evolution
        v2_hex = hashlib.sha256(_serialize_commitment_v2(self.psi_star)).hexdigest()
        self.commitment = f"{SCHEMA_V2}:{v2_hex}"

    def evolve(self, psi0, n: int):
        psi = cp.asarray(psi0, dtype=cp.float64).copy()
        for _ in range(_steps):
            lap = laplacian(psi)
            fb  = alpha * lap / (psi + epsilon * cp.exp(-beta * psi**2))
            ent = theta * (psi * laplacian(cp.log(psi**2 + delta)))
            dpsi = _dt * (fb - ent) - _damping * psi
            psi  = psi + dpsi
        return psi

    def _curvature_hash(self, psi) -> str:
        c = str(getattr(self, "commitment", "")) or ""
        is_v1 = (
            c.startswith("WLv1:")
            or (len(c) == 64 and all(ch in "0123456789abcdef" for ch in c.lower()))
        )
        payload = _serialize_commitment_v1(psi) if is_v1 else _serialize_commitment_v2(psi)
        return hashlib.sha256(payload).hexdigest()

    def _sig_payload_v2(self, message: str) -> bytes:
        H = _canonical_json(_commit_header(self.psi_star))
        return (
            b"SIGv2\0"
            + message.encode("utf-8")
            + b"\0"
            + H
            + b"\0"
            + _float64_bytes(self.psi_star)
        )

    def sign(self, message: str) -> str:
        c = str(self.commitment)
        if c.startswith("WLv1:") or (len(c) == 64):
            psi_bytes = cp.asnumpy(self.psi_star.ravel()).tobytes()
            return hashlib.sha256(message.encode("utf-8") + psi_bytes).hexdigest()
        return hashlib.sha256(self._sig_payload_v2(message)).hexdigest()

    def verify(self, message: str, signature: str) -> bool:
        # 1) commitment check
        recomputed = self._curvature_hash(self.psi_star)
        stored = str(self.commitment)
        if ":" in stored:
            stored = stored.split(":", 1)[1]
        if recomputed != stored:
            return False
        # 2) signature check (mirror sign())
        c = str(self.commitment)
        if c.startswith("WLv1:") or (len(c) == 64):
            psi_bytes = cp.asnumpy(self.psi_star.ravel()).tobytes()
            expected = hashlib.sha256(message.encode("utf-8") + psi_bytes).hexdigest()
        else:
            expected = hashlib.sha256(self._sig_payload_v2(message)).hexdigest()
        return expected == signature


def load_quantum_keys(path: str = "psi_keypair.json") -> CurvatureKeyPair:
    import json
    with open(path, "r") as f:
        data = json.load(f)
    n = int(data.get("n", 4))
    kp = CurvatureKeyPair(n=n, seed=None)
    kp.psi_0 = cp.asarray(data["psi_0"], dtype=cp.float64)
    kp.psi_star = cp.asarray(data["psi_star"], dtype=cp.float64)
    # honor stored commitment if present; else recompute v2
    stored = data.get("commitment")
    if stored:
        kp.commitment = str(stored)
    else:
        v2_hex = hashlib.sha256(_serialize_commitment_v2(kp.psi_star)).hexdigest()
        kp.commitment = f"{SCHEMA_V2}:{v2_hex}"
    return kp


def symbolic_verifier(psi_candidate, reference_psi) -> bool:
    # "Return True iff the two ψ* arrays yield identical WLv2 commitments."
    h1 = hashlib.sha256(
        _serialize_commitment_v2(cp.asarray(psi_candidate, dtype=cp.float64))
    ).hexdigest()
    h2 = hashlib.sha256(
        _serialize_commitment_v2(cp.asarray(reference_psi, dtype=cp.float64))
    ).hexdigest()
    return h1 == h2
