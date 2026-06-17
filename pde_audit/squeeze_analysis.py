"""Phase 8C: independent analysis of the comparison-based 256-bit squeeze.

Over a large deterministic message sample, measures per-bit one-frequency and
confidence intervals, per-pair tie frequency, pairwise and serial output-bit
correlations, byte-frequency, output Hamming-weight distribution, dependence
between the four 64-bit squeeze rounds, and sensitivity to the cell-pair table
(tested as separately named variants WITHOUT changing the normative
SQUEEZE_PAIRS). Operates on raw 256-bit output / raw state only.

Run:
    python -m pde_audit.squeeze_analysis
"""

from __future__ import annotations

import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams
from wavelock.pde_hash import spec


def _squeeze_trace(v: PDEVariant, psi: np.ndarray):
    """Return (bits[256], ties[4,64]) capturing the 4 squeeze rounds."""
    pa, pb = v._pairs_a, v._pairs_b
    bits = np.zeros(256, dtype=np.uint8)
    ties = np.zeros((4, 64), dtype=np.int64)
    for r in range(4):
        flat = psi.reshape(-1)
        a = flat[pa]
        b = flat[pb]
        bits[r * 64:(r + 1) * 64] = (a > b).astype(np.uint8)
        ties[r] = (a == b).astype(np.int64)
        if r < 3:
            psi = v.evolve_T(psi)
    return bits, ties


def main(seed: int = 80030, n_msgs: int = 8000) -> dict:
    t0 = time.perf_counter()
    v = PDEVariant(PDEParams())
    msgs = H.random_messages(seed, n_msgs, 0, 400)

    bitmat = np.zeros((n_msgs, 256), dtype=np.uint8)
    tie_counts = np.zeros((4, 64), dtype=np.int64)
    for i, m in enumerate(msgs):
        psi = v.absorb(m)
        bits, ties = _squeeze_trace(v, psi)
        bitmat[i] = bits
        tie_counts += ties

    ones = bitmat.mean(axis=0)                      # per-bit P(1)
    ci = 1.96 * np.sqrt(ones * (1 - ones) / n_msgs)
    # monobit z per bit: (k - n/2)/sqrt(n/4)
    z = (bitmat.sum(axis=0) - n_msgs / 2) / np.sqrt(n_msgs / 4)
    worst_bit = int(np.argmax(np.abs(ones - 0.5)))

    # pairwise correlations (summary only)
    centered = bitmat.astype(np.float64) - ones
    std = bitmat.std(axis=0)
    nonconst = std > 0
    corr_max = 0.0
    if nonconst.sum() > 1:
        cm = np.corrcoef(bitmat[:, nonconst].T)
        off = cm[~np.eye(cm.shape[0], dtype=bool)]
        corr_max = float(np.max(np.abs(off)))

    # serial correlation within each output (lag-1 across the 256 bits)
    serial = []
    for i in range(min(n_msgs, 2000)):
        row = bitmat[i].astype(np.float64)
        if row.std() > 0:
            serial.append(np.corrcoef(row[:-1], row[1:])[0, 1])
    serial = np.array(serial)

    # byte-frequency distribution
    byte_hist = np.zeros(256, dtype=np.int64)
    weights = []
    for i in range(n_msgs):
        bs = np.packbits(bitmat[i])
        for x in bs:
            byte_hist[x] += 1
        weights.append(int(bitmat[i].sum()))
    weights = np.array(weights)

    # dependence between the four 64-bit squeeze rounds: correlation of the
    # per-pair bit between round r and round s (same pair index)
    round_blocks = bitmat.reshape(n_msgs, 4, 64)
    round_corr = np.zeros((4, 4))
    for r in range(4):
        for s in range(4):
            xr = round_blocks[:, r, :].astype(np.float64).ravel()
            xs = round_blocks[:, s, :].astype(np.float64).ravel()
            if xr.std() > 0 and xs.std() > 0:
                round_corr[r, s] = np.corrcoef(xr, xs)[0, 1]

    # cell-pair-table sensitivity: alternative tables as named variants
    alt = _alt_pair_bias(v, msgs[:2000])

    results = {
        "phase": "8C_squeeze",
        "metadata": H.env_metadata(),
        "seed": seed,
        "n_msgs": n_msgs,
        "per_bit_one_freq": {
            "mean": float(ones.mean()),
            "min": float(ones.min()),
            "max": float(ones.max()),
            "worst_bit": worst_bit,
            "worst_bit_freq": float(ones[worst_bit]),
            "worst_bit_ci_halfwidth": float(ci[worst_bit]),
            "max_abs_monobit_z": float(np.max(np.abs(z))),
            "num_bits_z_gt_3": int((np.abs(z) > 3).sum()),
        },
        "ties": {
            "total_ties": int(tie_counts.sum()),
            "per_round_pair_max": int(tie_counts.max()),
            "comparisons_per_round_pair": n_msgs,
        },
        "correlation": {
            "max_abs_pairwise": corr_max,
            "serial_lag1_mean": float(np.nanmean(serial)) if serial.size else None,
            "serial_lag1_max_abs": float(np.nanmax(np.abs(serial))) if serial.size else None,
            "round_block_corr": round_corr.tolist(),
        },
        "byte_freq": {
            "expected_per_byte": float(byte_hist.sum() / 256),
            "min": int(byte_hist.min()),
            "max": int(byte_hist.max()),
            "chi2": float(_chi2_uniform(byte_hist)),
            "dof": 255,
        },
        "hamming_weight": {
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": int(weights.min()),
            "max": int(weights.max()),
            "ideal_mean": 128.0,
            "ideal_std": float(np.sqrt(256 * 0.25)),
        },
        "pair_table_sensitivity": alt,
        "runtime_s": round(time.perf_counter() - t0, 2),
    }
    path = H.save_artifact("phase8c_squeeze.json", results)
    print(f"  per-bit P(1) in [{ones.min():.4f},{ones.max():.4f}] "
          f"maxZ={np.max(np.abs(z)):.2f} bits|z|>3={int((np.abs(z)>3).sum())}")
    print(f"  ties={int(tie_counts.sum())}  maxpairwise|corr|={corr_max:.4f}  "
          f"HW mean={weights.mean():.1f}±{weights.std():.1f}  chi2={results['byte_freq']['chi2']:.1f}")
    print("  saved", path, f"({results['runtime_s']}s)")
    return results


def _alt_pair_bias(v: PDEVariant, msgs) -> dict:
    """Bias of two alternative named pair tables vs the normative one."""
    n = len(msgs)
    states = [v.absorb(m) for m in msgs]
    tables = {
        "v0_t_tplus128": [(t, t + 128) for t in range(64)],
        "alt_adjacent": [(2 * t, 2 * t + 1) for t in range(64)],
        "alt_t_tplus64": [(t, t + 64) for t in range(64)],
    }
    out = {}
    for name, pairs in tables.items():
        pa = np.array([a for a, _ in pairs])
        pb = np.array([b for _, b in pairs])
        ones = np.zeros(64)
        for psi in states:
            flat = psi.reshape(-1)
            ones += (flat[pa] > flat[pb]).astype(np.float64)
        ones /= n
        out[name] = {"min": float(ones.min()), "max": float(ones.max()),
                     "max_abs_dev_from_half": float(np.max(np.abs(ones - 0.5)))}
    return out


def _chi2_uniform(hist: np.ndarray) -> float:
    exp = hist.sum() / hist.size
    return float(((hist - exp) ** 2 / exp).sum())


if __name__ == "__main__":
    main()
