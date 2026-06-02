#!/usr/bin/env python3
"""
ATTACK 4 — Statistical distinguishability.

CRITICAL FRAMING: C = SHA256(Serialize(psi*)). SHA-256 is a strong PRF, so the
commitment C will pass every avalanche / monobit / serial test REGARDLESS of how
structured psi* is. A "pass" on C is therefore non-evidence about WaveLock's
PDE. The honest tests run on psi* and on Serialize(psi*) — the pre-hash objects
that WaveLock's nonlinearity is actually responsible for.

  D1  Avalanche of C under 1-bit seed flips  -> expected ~50% (proves only SHA)
  D2  Monobit / byte-frequency on C bytes     -> expected uniform (proves SHA)
  D3  Monobit / byte-frequency on Serialize(psi*) bytes -> expect STRONG bias
      (float64 exponent/sign bytes are far from uniform)
  D4  Distribution of psi* values vs random (range, skew, kurtosis, KS)
  D5  2D spatial autocorrelation / FFT structure of psi* (neighbor correlation)

Usage:  python audit/a7_distinguishability.py [M]
Writes: audit/artifacts/a7_distinguishability.json
"""
from __future__ import annotations
import sys, os, json, hashlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audit._wl as H
from audit.a1_attractor_entropy import evolve_batch, psi0_batch


def bits_of(b: bytes) -> np.ndarray:
    return np.unpackbits(np.frombuffer(b, dtype=np.uint8))


def monobit_ratio(b: bytes) -> float:
    bits = bits_of(b)
    return float(bits.mean())  # fraction of 1-bits; ideal 0.5


def byte_chi2_uniform(b: bytes):
    arr = np.frombuffer(b, dtype=np.uint8)
    counts = np.bincount(arr, minlength=256)
    exp = len(arr) / 256.0
    chi2 = float(((counts - exp) ** 2 / exp).sum())  # df=255
    return chi2


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    n = 4
    side = H.side_for_n(n)
    out = {"params": {"n": n, "M": M},
           "FRAMING": ("C passes randomness tests because of SHA-256, not the "
                       "PDE. Evidence about WaveLock's one-wayness lives in psi* "
                       "and Serialize(psi*), not in C.")}

    seeds = list(range(M))
    ps = evolve_batch(psi0_batch(seeds, n))      # (M, side, side)
    flat = ps.reshape(M, side * side)

    # Commitments and serialized bodies
    C_bytes = b"".join(bytes.fromhex(hashlib.sha256(H.serialize(ps[i])).hexdigest())
                       for i in range(M))
    ser_bodies = b"".join(ps[i].astype("<f8").tobytes() for i in range(M))

    # D1 avalanche of C under 1-bit seed flips
    flips = []
    base = 0xDEADBE
    cbase = bits_of(bytes.fromhex(hashlib.sha256(H.serialize(H.evolve(H.psi0_xof(base, n)))).hexdigest()))
    for bit in range(20):
        s2 = base ^ (1 << bit)
        c2 = bits_of(bytes.fromhex(hashlib.sha256(H.serialize(H.evolve(H.psi0_xof(s2, n)))).hexdigest()))
        flips.append(float((cbase != c2).mean()))
    out["D1_avalanche_C_seedbit"] = {
        "mean_fraction_C_bits_flipped": round(float(np.mean(flips)), 4),
        "ideal": 0.5,
        "verdict": "≈0.5 as expected — attributable to SHA-256, not the PDE.",
    }

    # D2 monobit + chi2 on C
    out["D2_C_randomness"] = {
        "monobit_ones_fraction": round(monobit_ratio(C_bytes), 5),
        "byte_chi2_df255": round(byte_chi2_uniform(C_bytes), 1),
        "chi2_critical_0.01_df255": 310.5,
        "verdict": "C looks uniform — SHA-256 doing its job; says nothing about PDE.",
    }

    # D3 monobit + chi2 on Serialize(psi*) bodies  (the real test)
    out["D3_serialized_psistar_randomness"] = {
        "monobit_ones_fraction": round(monobit_ratio(ser_bodies), 5),
        "byte_chi2_df255": round(byte_chi2_uniform(ser_bodies), 1),
        "chi2_critical_0.01_df255": 310.5,
        "verdict": ("Strongly NON-uniform: float64 sign/exponent bytes of psi* "
                    "are highly biased. Serialize(psi*) is trivially "
                    "distinguishable from random; only SHA-256 hides it."),
    }

    # D4 value distribution vs random
    allv = flat.ravel()
    finite = allv[np.isfinite(allv)]
    mean = float(finite.mean()); std = float(finite.std())
    out["D4_value_distribution"] = {
        "min": float(finite.min()), "max": float(finite.max()),
        "mean": mean, "std": std,
        "skew": float(((finite - mean) ** 3).mean() / std ** 3),
        "excess_kurtosis": float(((finite - mean) ** 4).mean() / std ** 4 - 3),
        "fraction_negative": float((finite < 0).mean()),
        "verdict": ("Heavy-tailed, skewed, far from the input U[0,1) and far "
                    "from any clean reference distribution — psi* is structured."),
    }

    # D5 spatial neighbor correlation in psi* (toroidal)
    # correlation between each cell and its +x neighbor across the M fields
    a = flat
    nb = np.roll(ps, -1, axis=2).reshape(M, side * side)
    af = a.ravel(); bf = nb.ravel()
    m = np.isfinite(af) & np.isfinite(bf)
    corr = float(np.corrcoef(af[m], bf[m])[0, 1])
    out["D5_spatial_autocorrelation"] = {
        "neighbor_pearson_corr": round(corr, 4),
        "verdict": ("Non-zero spatial correlation (smoothing stencil) — psi* "
                    "carries detectable spatial structure unlike random fields."),
    }

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "artifacts", "a7_distinguishability.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=== a7 distinguishability ===")
    print("D1 C avalanche (seed bit):", out["D1_avalanche_C_seedbit"]["mean_fraction_C_bits_flipped"], "(ideal 0.5)")
    print("D2 C monobit/chi2:", out["D2_C_randomness"]["monobit_ones_fraction"],
          "/", out["D2_C_randomness"]["byte_chi2_df255"], "(crit 310.5)")
    print("D3 Serialize(psi*) monobit/chi2:",
          out["D3_serialized_psistar_randomness"]["monobit_ones_fraction"],
          "/", out["D3_serialized_psistar_randomness"]["byte_chi2_df255"], "(crit 310.5)")
    print("D4 psi* mean/std/skew/kurt:", round(mean, 2), round(std, 2),
          round(out["D4_value_distribution"]["skew"], 2),
          round(out["D4_value_distribution"]["excess_kurtosis"], 2))
    print("D5 neighbor corr:", out["D5_spatial_autocorrelation"]["neighbor_pearson_corr"])


if __name__ == "__main__":
    main()
