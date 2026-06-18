"""Part III -- measurable curvature and signature functionals.

Every functional is defined on *lifted integer representatives* of the field
(see _common.lift_centered). These are DIAGNOSTIC quantities: they describe the
forward trajectory. None of them is asserted to be a cryptographic invariant or
a hardness measure, and the document explicitly tests their dependence on the
lifting convention (lifting_sensitivity) -- a large numerical curvature that
flips under a different, equally valid lifting is not a cryptographic quantity.

Definitions (psi is an (N,N) field; lift L(.) = lift_centered unless noted):

  spatial curvature   K2(psi)   = sum_x ( Laplacian(psi)(x) )^2          (lifted)
  gradient energy     G(psi)    = sum_{edges} ( psi(x) - psi(y) )^2      (lifted)
  temporal curvature  Ktau      = sum_x ( p_{t+1} - 2 p_t + p_{t-1} )^2  (lifted)
  trajectory sep      D_Gamma   = sum_t w_t * || psi_t(m) - psi_t(m') ||^2

  signature complexity candidates C_Gamma(m):
    - curvature_integral    : sum_t K2(psi_t)             (lifted, scale-dependent)
    - spectral_effrank      : effective rank of the per-round 2D power spectrum
                              (efficiently computable; lifting-robust up to scale)
    - distinct_modes        : # FFT magnitude bins above 1% of the max (per round,
                              summed) -- a crude bandwidth proxy
    - bitstring_entropy     : Shannon entropy (bits) of the digest bytes (always
                              <= 256; a property of the output, not the path)

All sums are computed in Python big integers (exact) to avoid float overflow,
then returned as floats/ints. Spectral measures use float FFT and are therefore
approximate and report that explicitly.
"""

from __future__ import annotations

import math
import time

import numpy as np

from wavelock.curvature_capacity import spec, optimized as opt
from . import _common as C

P = spec.P
N = spec.N


# --- core lattice operators on lifted integers --------------------------
def _lap_lifted(field_resid: np.ndarray) -> np.ndarray:
    """Integer Laplacian on the centered lift (exact small integers)."""
    a = C.lift_centered(field_resid)
    return (np.roll(a, -1, 0) + np.roll(a, 1, 0)
            + np.roll(a, -1, 1) + np.roll(a, 1, 1) - 4 * a)


def spatial_curvature(field_resid: np.ndarray) -> int:
    lap = _lap_lifted(field_resid).astype(object)
    return int(np.sum(lap * lap))


def gradient_energy(field_resid: np.ndarray) -> int:
    a = C.lift_centered(field_resid).astype(object)
    dx = a - np.roll(a, -1, 1)
    dy = a - np.roll(a, -1, 0)
    return int(np.sum(dx * dx) + np.sum(dy * dy))


def temporal_curvature(prev_r, cur_r, next_r) -> int:
    a = C.lift_centered(prev_r).astype(object)
    b = C.lift_centered(cur_r).astype(object)
    c = C.lift_centered(next_r).astype(object)
    d = c - 2 * b + a
    return int(np.sum(d * d))


def trajectory_separation(traj_a, traj_b, weights=None) -> int:
    T = len(traj_a)
    if weights is None:
        weights = [1] * T
    total = 0
    for t in range(T):
        da = C.lift_centered(traj_a[t]).astype(object)
        db = C.lift_centered(traj_b[t]).astype(object)
        diff = da - db
        total += int(weights[t]) * int(np.sum(diff * diff))
    return total


# --- spectral / entropy proxies (approximate, float) --------------------
def spectral_effrank(field_resid: np.ndarray) -> float:
    """Effective rank of the 2D power spectrum: exp(H(normalized power))."""
    a = C.lift_centered(field_resid).astype(np.float64)
    f = np.abs(np.fft.fft2(a)) ** 2
    s = f.sum()
    if s <= 0:
        return 0.0
    p = (f / s).ravel()
    p = p[p > 0]
    H = -np.sum(p * np.log(p))
    return float(math.exp(H))


def distinct_modes(field_resid: np.ndarray, frac: float = 0.01) -> int:
    a = C.lift_centered(field_resid).astype(np.float64)
    mag = np.abs(np.fft.fft2(a))
    return int(np.sum(mag > frac * mag.max()))


def byte_entropy(digest: bytes) -> float:
    counts = np.bincount(np.frombuffer(digest, dtype=np.uint8), minlength=256)
    p = counts[counts > 0] / counts.sum()
    return float(-np.sum(p * np.log2(p)))


# --- trajectory extraction ---------------------------------------------
def wave_trajectory(psi0: np.ndarray, rounds: int) -> list:
    """Return [psi_0, psi_1, ..., psi_rounds] as residue arrays (wave field)."""
    psi = (np.asarray(psi0, dtype=np.int64) % P).reshape(N, N).copy()
    traj = [psi.copy()]
    for _ in range(rounds):
        psi = opt._wave_round(psi)
        traj.append(psi.copy())
    return traj


def signature_complexity(psi0: np.ndarray, rounds: int) -> dict:
    traj = wave_trajectory(psi0, rounds)
    ci = sum(spatial_curvature(s) for s in traj)
    effr = float(np.mean([spectral_effrank(s) for s in traj]))
    dm = sum(distinct_modes(s) for s in traj)
    return {"curvature_integral": ci, "mean_spectral_effrank": effr,
            "distinct_modes_sum": dm}


# --- lifting sensitivity (the central honesty check) --------------------
def lifting_sensitivity(psi0: np.ndarray, rounds: int) -> dict:
    """Compare curvature under centered vs naive lifting.

    If the same field yields wildly different curvature under two equally valid
    integer representatives, the curvature *magnitude* is convention-dependent
    and cannot by itself be a cryptographic quantity. We report both and their
    ratio.
    """
    def K2_with(lifter, resid):
        a = lifter(resid)
        lap = (np.roll(a, -1, 0) + np.roll(a, 1, 0)
               + np.roll(a, -1, 1) + np.roll(a, 1, 1) - 4 * a).astype(object)
        return int(np.sum(lap * lap))
    traj = wave_trajectory(psi0, rounds)
    centered = [K2_with(C.lift_centered, s) for s in traj]
    naive = [K2_with(C.lift_naive, s) for s in traj]
    # a deliberately structured low-magnitude state: residue 1 everywhere except
    # a single cell at P-1 (lift_centered -> -1, lift_naive -> P-1).
    structured = np.ones((N, N), dtype=np.int64)
    structured[0, 0] = P - 1
    K2_struct_c = K2_with(C.lift_centered, structured)
    K2_struct_n = K2_with(C.lift_naive, structured)
    return {
        "well_mixed_state": {
            "K2_centered_mean": float(np.mean([float(x) for x in centered])),
            "K2_naive_mean": float(np.mean([float(x) for x in naive])),
            "ratio_naive_over_centered_log10":
                float(np.mean([math.log10(max(n, 1) / max(c, 1))
                               for n, c in zip(naive, centered)])),
            "note": "for well-mixed (post-evolution) fields the residues fill "
                    "[0,P) so both lifts give statistically similar magnitude "
                    "(ratio ~ 1).",
        },
        "structured_low_magnitude_state": {
            "K2_centered": K2_struct_c,
            "K2_naive": K2_struct_n,
            "ratio_naive_over_centered_log10":
                float(math.log10(max(K2_struct_n, 1) / max(K2_struct_c, 1))),
            "note": "for structured states with residues near 0 or near P the two "
                    "lifts diverge by ~18 orders of magnitude; curvature MAGNITUDE "
                    "is therefore convention-dependent and is a diagnostic, not a "
                    "cryptographic invariant. Cryptographic claims must not rest "
                    "on raw curvature magnitude.",
        },
    }


def main(seed: int = 90100) -> dict:
    t0 = time.perf_counter()
    g = C.rng(seed)
    psi0 = g.integers(0, P, size=(N, N), dtype=np.int64)
    rounds = spec.T
    demo = {
        "random_state": {
            "spatial_curvature_K2": spatial_curvature(psi0),
            "gradient_energy_G": gradient_energy(psi0),
            "spectral_effrank": spectral_effrank(psi0),
            "distinct_modes": distinct_modes(psi0),
            "signature_complexity": signature_complexity(psi0, rounds),
        },
        "lifting_sensitivity": lifting_sensitivity(psi0, rounds),
        "digest_byte_entropy_example": byte_entropy(opt.cc_hash(b"abc")),
    }
    # eigenmode states: curvature integral differs across the family even though
    # they share terminal wave state 0.
    eig = C.eigenmode_states()
    demo["eigenmode_curvature_integral"] = {
        k: signature_complexity(v, rounds)["curvature_integral"]
        for k, v in eig.items()
    }
    out = {
        "phase": "curvature_metrics_demo",
        "metadata": C.env_metadata(),
        "seed": seed,
        "rounds": rounds,
        "definitions": "see module docstring; functionals on lift_centered",
        "results": demo,
        "limitations": [
            "curvature magnitudes are convention-dependent (see lifting_sensitivity)",
            "spectral measures use float FFT and are approximate",
            "these are forward-trajectory diagnostics, NOT hardness measures",
        ],
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    C.save_artifact("curvature_metrics_demo.json", out)
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(main()["results"]["lifting_sensitivity"], indent=2))
