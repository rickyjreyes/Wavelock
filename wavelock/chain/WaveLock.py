from __future__ import annotations
from typing import Optional, Tuple, List
import hashlib
import json
import struct
import cupy as cp
import numpy as np
from dataclasses import dataclass

import hmac


def _to_numpy(x):
    """Convert any input (NumPy, CuPy, list) -> NumPy float64 safely."""
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x, dtype=np.float64)


def _secure_compare(a: str, b: str) -> bool:
        return hmac.compare_digest(
        a.encode("utf-8"),
        b.encode("utf-8"),
    )

# === WLv3 Survivability Layer ===
from .hash_families import (
    HashFamily,
    DualHash,
    hash_hex,
    format_commitment_v3,
    parse_commitment,
    DEFAULT_PRIMARY_FAMILY,
    DEFAULT_SECONDARY_FAMILY,
)

# SHAKE-256 ψ₀ derivation (Claim 9 / Best Mode). Imported here so the GPU
# keypair can produce a byte-identical ψ₀ to the NumPy reference path when
# operating under the consensus regime.
from .xof_init import derive_psi_zero


# ===========================================================
#  WaveLock v1–v7 Unified Curvature System
#  Default: WLv2 (fast, stable, test-compatible)
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

# Schemas
SCHEMA_V1 = "WLv1"
SCHEMA_V2 = "WLv2"
SCHEMA_V3 = "WLv3"
SCHEMA_V4 = "WLv4"
SCHEMA_V5 = "WLv5"
SCHEMA_V6 = "WLv6"
SCHEMA_V7 = "WLv7"

# ---------------------------
# Invariant Selector (WLv2 default)
# ---------------------------
DEFAULT_INVARIANTS = {
    "WCT_core": True,
    "Lyapunov": False,
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
#                    WCC Rails
# ===========================================================

@dataclass(frozen=True)
class WCCBounds:
    name: str
    max_steps: int
    max_cells: int
    max_avg_c_theta: float

PWCC_BOUNDS = WCCBounds("PWCC", 10_4, 256 * 256, 10.0)
NPWCC_BOUNDS = WCCBounds("NPWCC", 10_6, 1024 * 1024, 1.0e3)
QUANTUM_CLASSICAL_BOUND = 1.0e2

MAX_WCC_SPATIAL_DIM = 2
REQUIRE_POWER_OF_TWO_SIDE = True

def _float64_bytes(x_cp) -> bytes:
    arr = cp.asnumpy(cp.asarray(x_cp, dtype=cp.float64))
    return arr.ravel(order="C").tobytes()

def laplacian(x):
    return (
        -4.0 * x
        + cp.roll(x, +1, 0) + cp.roll(x, -1, 0)
        + cp.roll(x, +1, 1) + cp.roll(x, -1, 1)
    )

def enforce_dimensional_lock(psi):
    psi = cp.asarray(psi)
    if psi.ndim != 2:
        raise ValueError("ψ must be 2D.")
    nx, ny = psi.shape
    if nx != ny:
        raise ValueError("ψ must be square.")
    if REQUIRE_POWER_OF_TWO_SIDE:
        if nx <= 0 or (nx & (nx - 1)) != 0:
            raise ValueError("ψ side length must be power-of-two.")

# ===========================================================
#      Curvature functional
# ===========================================================

def _curvature_functional(psi) -> Tuple[float, float, float, float]:
    gx, gy = cp.gradient(psi)
    E_grad = float(cp.sum(gx * gx) + cp.sum(gy * gy))

    lap = laplacian(psi)
    feedback = alpha * lap / (psi + epsilon * cp.exp(-beta * psi ** 2))

    entropy_term = theta * (psi * laplacian(cp.log(psi ** 2 + delta)))

    E_fb  = float(cp.sum(feedback * feedback))
    E_ent = float(cp.sum(entropy_term * entropy_term))

    E_tot = float(E_grad + E_fb + E_ent)
    return E_grad, E_fb, E_ent, E_tot

# ===========================================================
#      WCC curvature budget & classification
# ===========================================================

def wcc_curvature_budget(psi) -> float:
    E_grad, E_fb, _, _ = _curvature_functional(psi)
    return float(E_grad + E_fb)

def wcc_avg_curvature_budget(psi) -> float:
    psi = cp.asarray(psi, dtype=cp.float64)
    return wcc_curvature_budget(psi) / float(psi.size)

def classify_wcc_run(steps: int, psi) -> str:
    psi = cp.asarray(psi)
    n_cells = int(psi.size)
    avg_c = wcc_avg_curvature_budget(psi)

    def within(b: WCCBounds):
        return (
            steps   <= b.max_steps and
            n_cells <= b.max_cells and
            avg_c   <= b.max_avg_c_theta
        )

    if within(PWCC_BOUNDS):
        return "PWCC"
    if within(NPWCC_BOUNDS):
        return "NPWCC"
    return "OUT_OF_WCC"

def check_quantum_classical_bound(psi) -> bool:
    return wcc_avg_curvature_budget(psi) <= QUANTUM_CLASSICAL_BOUND

# ===========================================================
#   Commitment Serialization (v1–v7)
# ===========================================================

# ------------------
# v1: legacy curvature
# ------------------
def _serialize_commitment_v1(psi) -> bytes:
    gx, gy = cp.gradient(psi)
    curv = float(cp.sum(gx * gx) + cp.sum(gy * gy))
    return _float64_bytes(psi) + f"{curv:.17g}".encode("utf-8")

# ------------------
# v2: header + ψ + curvature
# ------------------
def _serialize_commitment_v2(psi) -> bytes:
    header_bytes = _canonical_json({
        "schema": SCHEMA_V2,
        "dtype": "float64",
        "ord": "C",
        "shape": [int(x) for x in psi.shape],
        "alpha": float(alpha),
        "beta":  float(beta),
        "theta": float(theta),
        "epsilon": float(epsilon),
        "delta": float(delta),
        "kernel_version": KERNEL_VERSION,
        "kernel_hash": _kernel_hash(),
    })

    E_grad, E_fb, E_ent, E_tot = _curvature_functional(psi)
    packed_E = struct.pack("<4d", E_grad, E_fb, E_ent, E_tot)

    return b"WLv2\0" + header_bytes + _float64_bytes(psi) + packed_E

# ------------------
# v3: float64 canonical ψ
# ------------------
def _serialize_commitment_v3(psi):
    psi_arr = cp.asnumpy(cp.asarray(psi, dtype=cp.float64))
    psi_be = psi_arr.astype(">f8", copy=False)
    return psi_be.tobytes(order="C")

# ------------------
# v4: curvature-invariants + asymmetry + low-freq FFT
# ------------------
def _serialize_commitment_v4(psi):
    psi_cp = cp.asarray(psi, dtype=cp.float64)
    psi_np = cp.asnumpy(psi_cp)

    # curvature invariants
    E_grad, E_fb, E_ent, E_tot = _curvature_functional(psi_cp)
    eps = 1e-9
    scale = np.sqrt(abs(E_tot) + eps)

    I_grad = E_grad / (scale + eps)
    I_fb   = E_fb   / (scale + eps)
    I_ent  = E_ent  / (scale + eps)
    I_tot  = np.tanh(E_tot * 1e-3)

    invariants = np.array([I_grad, I_fb, I_ent, I_tot], dtype=np.float64)

    # anti-symmetric projections
    h, w = psi_np.shape
    xs = (np.arange(w) - w / 2.0).reshape(1, -1)
    ys = (np.arange(h) - h / 2.0).reshape(-1, 1)

    Mx  = np.sum(psi_np * xs)
    My  = np.sum(psi_np * ys)
    Mxy = np.sum(psi_np * (xs * ys))

    norm = np.sqrt(np.sum(psi_np ** 2)) + eps
    asym = np.array([Mx / norm, My / norm, Mxy / norm], dtype=np.float64)

    # FFT orientation features
    F = np.fft.rfft2(psi_np)

    def _pair(z):
        return float(np.real(z)), float(np.imag(z))

    f10r, f10i = _pair(F[1, 0])
    f01r, f01i = _pair(F[0, 1])
    f11r, f11i = _pair(F[1, 1])

    fft_feats = np.array([f10r, f10i, f01r, f01i, f11r, f11i], dtype=np.float64)

    all_feats = np.concatenate([invariants, asym, fft_feats])
    return all_feats.astype(">f8").tobytes(order="C")

# ------------------
# v5: curvature + Laplacian spectral + winding + wavelets
# ------------------
def _serialize_commitment_v5(psi):
    psi_cp = cp.asarray(psi, dtype=cp.float64)
    psi_np = cp.asnumpy(psi_cp)

    E_grad, E_fb, E_ent, E_tot = _curvature_functional(psi_cp)

    eps = 1e-9
    scale = np.sqrt(abs(E_tot) + eps)

    base = np.array(
        [
            E_grad / scale,
            E_fb   / scale,
            E_ent  / scale,
            np.tanh(E_tot * 1.0e-3),
        ],
        dtype=np.float64,
    )

    # Laplacian spectral
    L = (
        -4 * psi_np
        + np.roll(psi_np, 1, 0) + np.roll(psi_np, -1, 0)
        + np.roll(psi_np, 1, 1) + np.roll(psi_np, -1, 1)
    )
    F = np.fft.rfft2(L)
    eig_feats = np.sort(np.abs(F.flatten()))[:6].astype(np.float64)

    # winding
    ph = np.angle(F)
    winding = np.array(
        [
            float(np.sum(ph)),
            float(np.sum(np.sin(ph))),
            float(np.sum(np.cos(ph))),
        ],
        dtype=np.float64,
    )

    # wavelet features
    import pywt
    coeffs = pywt.wavedec2(psi_np, "haar", level=2)
    wv = np.concatenate([c.flatten()[:8] for c in coeffs[1:]]).astype(np.float64)

    combined = np.concatenate([base, eig_feats, winding, wv])
    return combined.astype(">f8").tobytes(order="C")

# ------------------
# v6: full evolution chain spectral sketch
# ------------------
def _serialize_commitment_v6(psi_history: List[cp.ndarray]) -> bytes:
    sketches: List[bytes] = []

    for psi in psi_history:
        psi_np = cp.asnumpy(cp.asarray(psi, dtype=cp.float64))
        F = np.fft.rfft2(psi_np)

        sel = np.concatenate(
            [
                np.real(F[0, 1:5]),
                np.imag(F[0, 1:5]),
                np.real(F[1, 1:5]),
                np.imag(F[1, 1:5]),
            ]
        ).astype(np.float64)

        sketches.append(sel[:4].astype(">f8").tobytes(order="C"))

    chain = b"".join(sketches)
    return chain

# ------------------
# v7: curvature + chaotic logistic salt
# ------------------
def _serialize_commitment_v7(psi) -> bytes:
    psi_cp = cp.asarray(psi, dtype=cp.float64)
    psi_np = cp.asnumpy(psi_cp)

    # base curvature
    E_grad, E_fb, E_ent, E_tot = _curvature_functional(psi_cp)
    base = np.array([E_grad, E_fb, E_ent, E_tot], dtype=np.float64)

    # chaotic salt
    C = np.abs(np.fft.rfft2(psi_np))
    seed = float(np.sum(C) + np.sum(np.sin(C)) + np.sum(np.cos(C)))
    x = seed % 1.0

    r = 3.99999
    logi = []
    for _ in range(256):
        x = r * x * (1.0 - x)
        logi.append(x)

    chaos = np.array(logi[:16], dtype=np.float64)
    combo = np.concatenate([base, chaos])
    return combo.astype(">f8").tobytes(order="C")

# ===========================================================
#               CurvatureKeyPair
# ===========================================================
class CurvatureKeyPair:
    def __init__(
        self,
        n: int,
        seed: Optional[int] = None,
        invariants=None,
        use_v3: bool = False,
        use_v4: bool = False,
        use_v5: bool = False,
        use_v6: bool = False,
        use_v7: bool = False,
        test_mode: bool = False,
        use_xof_init: Optional[bool] = None,
    ):
        # ---------------------------------------------------------
        # Auto-enable test_mode when running under pytest
        # (works even during import & test collection)
        # ---------------------------------------------------------
        # import sys
        # if "pytest" in sys.modules:
        #     test_mode = True

        self.test_mode = bool(test_mode)



        # version flags
        self.use_v3 = bool(use_v3)
        self.use_v4 = bool(use_v4)
        self.use_v5 = bool(use_v5)
        self.use_v6 = bool(use_v6)
        self.use_v7 = bool(use_v7)

        self.n = n
        self.invariants = invariants or DEFAULT_INVARIANTS

        # ---------------------------------------------------------
        # ψ₀ derivation regime
        # ---------------------------------------------------------
        # SHAKE-256 with the WL-PSI-INIT-v1 domain tag is the consensus
        # path described by Claim 9 / Best Mode. The legacy cupy.random
        # path is non-consensus (backend- and version-bound) and only
        # remains for pre-v3 callers that have never produced a binding
        # commitment.
        any_consensus_schema = (
            self.use_v3 or self.use_v4 or self.use_v5 or self.use_v6 or self.use_v7
        )
        if use_xof_init is None:
            use_xof_init = any_consensus_schema
        self.use_xof_init = bool(use_xof_init)

        # side length is always power-of-two
        side = 2 ** max(1, n // 2)

        # history for v6, and general introspection
        self.psi_history: List[cp.ndarray] = []

        # initial field
        if self.use_xof_init:
            if seed is None:
                raise ValueError(
                    "use_xof_init=True requires an explicit seed; SHAKE-256 "
                    "derivation is deterministic in the seed bytes alone."
                )
            psi0_host = derive_psi_zero(seed, (side, side))
            self._psi_0 = cp.asarray(psi0_host, dtype=cp.float64)
        else:
            cp.random.seed(seed)
            self._psi_0 = cp.random.rand(side, side)
        enforce_dimensional_lock(self._psi_0)
        # evolve while storing all frames (safe for all schemas)
        self.__psi_star = self._evolve_capture(self._psi_0, n)
        enforce_dimensional_lock(self.__psi_star)

        # ---------------------------------------------------------
        # CONSENSUS GUARD
        # ---------------------------------------------------------
        # GPU / CuPy backend is NON-CONSENSUS.
        # Commitments must be generated via the NumPy reference
        # implementation unless explicitly in test_mode.
        if (not self.test_mode) and (
            self.use_v3 or self.use_v4 or self.use_v5 or self.use_v6 or self.use_v7
        ):
            raise RuntimeError(
                "GPU backend is non-consensus. "
                "Only NumPy reference backend may emit consensus commitments."
            )



        # choose commitment version once
        if self.use_v7:
            raw = _serialize_commitment_v7(self.__psi_star)
            schema = SCHEMA_V7
        elif self.use_v6:
            raw = _serialize_commitment_v6(self.psi_history)
            schema = SCHEMA_V6
        elif self.use_v5:
            raw = _serialize_commitment_v5(self.__psi_star)
            schema = SCHEMA_V5
        elif self.use_v4:
            raw = _serialize_commitment_v4(self.__psi_star)
            schema = SCHEMA_V4
        elif self.use_v3:
            raw = _serialize_commitment_v3(self.__psi_star)
            schema = SCHEMA_V3
        else:
            raw = _serialize_commitment_v2(self.__psi_star)
            schema = SCHEMA_V2

        self.schema = schema
        # self.commitment = f"{schema}:{hashlib.sha256(raw).hexdigest()}"


        self.primary_family   = DEFAULT_PRIMARY_FAMILY
        self.secondary_family = DEFAULT_SECONDARY_FAMILY
        # self.dual_hash        = None
        # self.commitment_v2    = None  # backward-compatible


        # --- Compute raw primary hash (WLv2 format remains intact) ---
        primary_hex = hashlib.sha256(raw).hexdigest()
        self.commitment_v2 = f"{schema}:{primary_hex}"

        # --- WLv3 dual-hash commitment ---
        self.dual_hash = DualHash.from_data(raw,
                                            primary_family=self.primary_family,
                                            secondary_family=self.secondary_family)

        secondary_hex = self.dual_hash.secondary.hex()

        # Full WLv3 commitment string
        self.commitment = format_commitment_v3(schema, primary_hex, secondary_hex)
    
    # ============================
    # Protected Access to ψ★ / ψ₀
    # ============================
    @property
    def psi_star(self):
        if not self.test_mode:
            raise PermissionError("ψ★ access is prohibited. Enable test_mode=True explicitly.")
        return self.__psi_star
    
    @psi_star.setter
    def psi_star(self, value):
        """
        Allow ψ★ mutation ONLY in test_mode.
        Production mode remains immutable.
        """
        if not self.test_mode:
            raise PermissionError("Cannot modify ψ★ outside of test_mode.")
        self.__psi_star = cp.asarray(value, dtype=cp.float64)


    @property
    def psi_0(self):
        if not self.test_mode:
            raise PermissionError("ψ₀ access is prohibited. Enable test_mode=True explicitly.")
        return self._psi_0
    
    @psi_0.setter
    def psi_0(self, value):
        if not self.test_mode:
            raise PermissionError("Cannot modify ψ₀ outside of test_mode.")
        self._psi_0 = cp.asarray(value, dtype=cp.float64)


    @classmethod
    def from_loaded(cls, psi_star, commitment, primary, secondary, schema=None):
        obj = cls.__new__(cls)

        # schema restore
        if schema is None:
            try:
                parsed_schema, _, _ = parse_commitment(commitment)
                obj.schema = parsed_schema
            except Exception:
                obj.schema = SCHEMA_V2
        else:
            obj.schema = schema

        # IMPORTANT: bypass protection
        obj.__psi_star = cp.asarray(psi_star, dtype=cp.float64)

        # assignment
        obj.commitment = commitment

        # Convert strings → enums
        obj.primary_family = HashFamily(primary)
        obj.secondary_family = HashFamily(secondary)

        # Extract hashes
        parts = commitment.split(":")
        p_hex = parts[1]
        s_hex = parts[2] if len(parts) == 3 else None

        if s_hex is None:
            raw = obj._curvature_hash_raw()
            s_hex = hash_hex(raw, HashFamily.SHA3_256)

        obj.dual_hash = DualHash.from_hex(p_hex, s_hex)

        # ensure loaded keys are test-only
        obj.test_mode = True

        return obj



    # PDE evolution capturing full history (for v6)
    def _evolve_capture(self, psi0, n: int):
        psi = cp.asarray(psi0, dtype=cp.float64).copy()
        self.psi_history.append(psi.copy())

        for _ in range(_steps):
            lap = laplacian(psi)
            fb  = alpha * lap / (psi + epsilon * cp.exp(-beta * psi ** 2))
            ent = theta * (psi * laplacian(cp.log(psi ** 2 + delta)))
            dpsi = _dt * (fb - ent) - _damping * psi
            psi = psi + dpsi
            self.psi_history.append(psi.copy())

        return psi
    
    
    # PDE evolution without history (legacy API for tests / utils)
    def evolve(self, psi0, n: int):
        """
        Legacy evolution API:
        Deterministically evolve psi0 using the same PDE as _evolve_capture,
        but without recording intermediate frames.
        The parameter n is kept for backward compatibility with older code.
        """
        psi = cp.asarray(psi0, dtype=cp.float64).copy()
        for _ in range(_steps):
            lap = laplacian(psi)
            fb  = alpha * lap / (psi + epsilon * cp.exp(-beta * psi ** 2))
            ent = theta * (psi * laplacian(cp.log(psi ** 2 + delta)))
            dpsi = _dt * (fb - ent) - _damping * psi
            psi  = psi + dpsi
        return psi

    # v2 signature payload
    def _sig_payload_v2(self, message: str) -> bytes:
        header = _canonical_json({
            "schema": SCHEMA_V2,
            "dtype": "float64",
            "ord": "C",
            # "shape": [int(x) for x in self.psi_star.shape],
            "shape": [int(x) for x in self.__psi_star.shape],
            "alpha": float(alpha),
            "beta":  float(beta),
            "theta": float(theta),
            "epsilon": float(epsilon),
            "delta": float(delta),
            "kernel_version": KERNEL_VERSION,
            "kernel_hash": _kernel_hash(),
        })

        if isinstance(message, bytes):
            msg_bytes = message
        else:
            msg_bytes = message.encode()

        return (
            b"SIGv2\0"
            + msg_bytes + b"\0"
            + header + b"\0"
            + _float64_bytes(self.__psi_star)
        )

        # return (
        #     b"SIGv2\0"
        #     + message.encode() + b"\0"
        #     + header + b"\0"
        #     + _float64_bytes(self.psi_star)
        # )
    
    def _sig_payload(self, message: str) -> bytes:
        """
        Unified signature payload for WLv3.
        Currently identical to v2 payload.
        Future schemas can plug in their own header.
        """
        return self._sig_payload_v2(message)


    # ------------------------
    # SIGNING (all versions)
    # ------------------------
    # def sign(self, message: str) -> str:
    #     schema = self.schema

    #     if schema == SCHEMA_V7:
    #         raw = _serialize_commitment_v7(self.psi_star) + message.encode()
    #     elif schema == SCHEMA_V6:
    #         raw = _serialize_commitment_v6(self.psi_history) + message.encode()
    #     elif schema == SCHEMA_V5:
    #         raw = _serialize_commitment_v5(self.psi_star) + message.encode()
    #     elif schema == SCHEMA_V4:
    #         raw = _serialize_commitment_v4(self.psi_star) + message.encode()
    #     elif schema == SCHEMA_V3:
    #         raw = _serialize_commitment_v3(self.psi_star) + message.encode()
    #     elif schema == SCHEMA_V2:
    #         raw = self._sig_payload_v2(message)
    #     else:  # SCHEMA_V1
    #         psi_bytes = cp.asnumpy(self.psi_star.ravel()).tobytes()
    #         raw = message.encode() + psi_bytes

    #     return hashlib.sha256(raw).hexdigest()
    def sign(self, message: str) -> str:
        """
        WLv3-compatible signature:
        Uses the unified signature payload and the primary hash family.
        """
        payload = self._sig_payload(message)
        return hash_hex(payload, self.primary_family)

    def sign_v3(self, message: str, family: HashFamily) -> str:
        """
        Single-hash signature using chosen hash family (WLv3).
        """
        payload = self._sig_payload(message)
        return hash_hex(payload, family)

    def sign_dual(self, message: str) -> tuple[str, str]:
        """
        Produce (primary_sig, secondary_sig)
        """
        payload = self._sig_payload(message)
        sig_primary   = hash_hex(payload, self.primary_family)
        sig_secondary = hash_hex(payload, self.secondary_family)
        return sig_primary, sig_secondary

    # ------------------------
    # Commitment hashing (all versions)
    # ------------------------
    def _curvature_hash(self, psi) -> str:
        schema = self.schema

        if schema == SCHEMA_V7:
            raw = _serialize_commitment_v7(psi)
        elif schema == SCHEMA_V6:
            # by design, v6 is a chain-hash of the *history*
            raw = _serialize_commitment_v6(self.psi_history)
        elif schema == SCHEMA_V5:
            raw = _serialize_commitment_v5(psi)
        elif schema == SCHEMA_V4:
            raw = _serialize_commitment_v4(psi)
        elif schema == SCHEMA_V3:
            raw = _serialize_commitment_v3(psi)
        elif schema == SCHEMA_V2:
            raw = _serialize_commitment_v2(psi)
        else:
            raw = _serialize_commitment_v1(psi)

        return hashlib.sha256(raw).hexdigest()

    def _curvature_hash_raw(self) -> bytes:
        """
        Return the EXACT raw bytes used in hashing the commitment.
        Required for WLv3 dual-hash verification.
        """
        schema = self.schema

        if schema == SCHEMA_V7:
            return _serialize_commitment_v7(self.__psi_star)
        elif schema == SCHEMA_V6:
            return _serialize_commitment_v6(self.psi_history)
        elif schema == SCHEMA_V5:
            return _serialize_commitment_v5(self.__psi_star)
        elif schema == SCHEMA_V4:
            return _serialize_commitment_v4(self.__psi_star)
        elif schema == SCHEMA_V3:
            # return _serialize_commitment_v3(self.psi_star)
            return _serialize_commitment_v3(self.__psi_star)

        elif schema == SCHEMA_V2:
            return _serialize_commitment_v2(self.__psi_star)
            # return _serialize_commitment_v2(self.psi_star)
        else:
            return _serialize_commitment_v1(self.__psi_star)


    def verify_commitment(self) -> tuple[bool, bool]:
        """
        Verify the WLv3 dual-hash commitment.
        Returns (primary_ok, secondary_ok)
        """
        schema, primary_hex, secondary_hex = parse_commitment(self.commitment)

        # Raw bytes exactly as hashed for commitment
        raw = self._curvature_hash_raw()

        # Use DualHash verifier
        return self.dual_hash.verify(
            raw,
            primary_family=self.primary_family,
            secondary_family=self.secondary_family,
        )

    # ------------------------
    # VERIFY (all versions)
    # ------------------------
    # def verify(self, message: str, signature: str) -> bool:
    #     schema, stored_hash = self.commitment.split(":", 1)

    #     # 1. Verify commitment
    #     recomputed = self._curvature_hash(self.psi_star)
    #     if recomputed != stored_hash:
    #         return False

    #     # 2. Verify signature
    #     expected = self.sign(message)
    #     return expected == signature
    
    def verify(self, message: str, signature: str) -> bool:
        """
        Survivability mode:
        Accept signature if it matches EITHER hash family.
        Timing-safe comparison.
        """
        payload = self._sig_payload(message)

        primary_sig   = hash_hex(payload, self.primary_family)
        secondary_sig = hash_hex(payload, self.secondary_family)

        return (
            _secure_compare(signature, primary_sig)
            or _secure_compare(signature, secondary_sig)
        )


    def verify_strict(self, message: str, sig_primary: str, sig_secondary: str) -> bool:
        expected_p, expected_s = self.sign_dual(message)

        return (
            _secure_compare(sig_primary, expected_p)
            and _secure_compare(sig_secondary, expected_s)
        )


# ===========================================================
# Symbolic Verifier — All Versions
# ===========================================================

def symbolic_verifier(psi_candidate, reference_psi, keypair: CurvatureKeyPair | None = None) -> bool:
    """
    Version-independent symbolic verifier:
    True iff commitments match exactly.
    """
    if keypair is None:
        # Assume v2 for backward compatibility
        h1 = hashlib.sha256(
            _serialize_commitment_v2(cp.asarray(psi_candidate))
        ).hexdigest()
        h2 = hashlib.sha256(
            _serialize_commitment_v2(cp.asarray(reference_psi))
        ).hexdigest()
        return h1 == h2

    candidate_hash = keypair._curvature_hash(cp.asarray(psi_candidate))
    reference_hash = keypair._curvature_hash(cp.asarray(reference_psi))
    return candidate_hash == reference_hash

# ===========================================================
#  Legacy API for psi-core / chain_utils
# ===========================================================

def generate_quantum_keys(
    n: int = 4,
    seed: int | None = None,
    path: str = "psi_keypair.json",
):
    """
    Legacy helper: generate ψ₀, ψ★ and commitment, write to JSON on disk.
    """
    kp = CurvatureKeyPair(n=n, seed=seed, test_mode=True)
    
    data = {
        "n": n,
        "psi_0": cp.asnumpy(kp.psi_0).tolist(),
        # "psi_star": cp.asnumpy(kp.psi_star).tolist(),
        "psi_star": (kp.__psi_star.tolist() if kp.test_mode else None),
        "commitment": kp.commitment,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data




def _serialize_keypair(kp):
    return {
        "schema": kp.schema,
        "commitment": kp.commitment,
        "primary_family": kp.primary_family.value,
        "secondary_family": kp.secondary_family.value,

        "psi_star": kp.psi_star.tolist(),   # safe fp64
        "params": {
            "alpha": float(alpha),
            "beta": float(beta),
            "theta": float(theta),
            "epsilon": float(epsilon),
            "delta": float(delta),
        }
    }


def save_quantum_keys(path, data: dict):
    serializable = {}
    for k, v in data.items():
        if isinstance(v, CurvatureKeyPair):
            serializable[k] = _serialize_keypair(v)
        else:
            serializable[k] = v

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)

def load_quantum_keys(path):
    with open(path, "r") as f:
        raw = json.load(f)

    out = {}
    for k, v in raw.items():
        if isinstance(v, dict) and "psi_star" in v:
            # kp = CurvatureKeyPair.from_loaded(
            #     psi_star=np.array(v["psi_star"], dtype=np.float64),
            #     commitment=v["commitment"],
            #     primary=v["primary_family"],
            #     secondary=v["secondary_family"],
            # )
                    
            # kp = CurvatureKeyPair.from_loaded(
            #     psi_star=np.array(v["psi_star"], dtype=np.float64),
            #     commitment=v["commitment"],
            #     primary=v["primary_family"],
            #     secondary=v["secondary_family"],
            #     schema=v.get("schema", None),
            # )
            kp = CurvatureKeyPair.from_loaded(
                psi_star=np.array(v["psi_star"], dtype=np.float64),
                commitment=v["commitment"],
                primary=v["primary_family"],
                secondary=v["secondary_family"],
                schema=v.get("schema", None),
            )

            # enforce test-mode ONLY
            kp.test_mode = True

            out[k] = kp
        else:
            out[k] = v

    return out

# def save_quantum_keys(path: str, data: dict):
#     with open(path, "w") as f:
#         json.dump(data, f, indent=2)

# def load_quantum_keys(path: str = "psi_keypair.json") -> CurvatureKeyPair:
#     """
#     Load a previously saved ψ₀, ψ★, and commitment.
#     Note: this reconstructs a CurvatureKeyPair with the *stored* commitment.
#     """
#     with open(path, "r") as f:
#         data = json.load(f)

#     n = int(data.get("n", 4))
#     kp = CurvatureKeyPair(n=n)  # will generate a fresh pair, then overwrite fields

#     kp.psi_0    = cp.asarray(data["psi_0"], dtype=cp.float64)
#     kp.psi_star = cp.asarray(data["psi_star"], dtype=cp.float64)
#     kp.commitment = data.get("commitment", kp.commitment)

#     # fix schema field from stored commitment
#     try:
#         # schema, _ = kp.commitment.split(":", 1)
#         schema, _, _ = parse_commitment(kp.commitment)

#     except ValueError:
#         schema = SCHEMA_V2
#     kp.schema = schema

#     return kp
