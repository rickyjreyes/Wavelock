#!/usr/bin/env python3
"""
ATTACK 3 — Attractor-collapse / effective output entropy.

Sweep a large seed range through the consensus WaveLock pipeline and quantify:
  * exact ψ* collisions and exact commitment (C) collisions
  * fraction of seeds diverging to NaN / Inf / |ψ*|>1e6
  * per-cell value entropy and effective output entropy of ψ*
  * near-duplicate clustering at several rounding tolerances

Evolution is BATCHED over the seed dimension for speed. The per-step math is
identical to wavelock/chain/Wavelock_numpy._evolve (verified in _wl.py); roll
axes shift by +1 because axis 0 is now the batch axis. We assert byte-equality
against the reference on a spot seed before trusting the batch path.

Usage:  python audit/a1_attractor_entropy.py [N] [--checkpoint-every K]
Writes: audit/artifacts/a1_entropy.json  (+ a1_checkpoint.json while running)
"""
from __future__ import annotations
import sys, os, json, time, hashlib, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audit._wl as H

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ART, exist_ok=True)

ALPHA, BETA, THETA = H.ALPHA, H.BETA, H.THETA
EPS, DELTA, DT, STEPS, DAMP = H.EPSILON, H.DELTA, H.DT, H.STEPS, H.DAMPING


def lap_batch(x):
    # x: (B, side, side); laplacian over the spatial axes (1, 2)
    return (-4.0 * x
            + np.roll(x, +1, 1) + np.roll(x, -1, 1)
            + np.roll(x, +1, 2) + np.roll(x, -1, 2))


def evolve_batch(psi0):
    psi = np.array(psi0, dtype=np.float64, copy=True)
    for _ in range(STEPS):
        lap = lap_batch(psi)
        fb = ALPHA * lap / (psi + EPS * np.exp(-BETA * psi ** 2))
        ent = THETA * (psi * lap_batch(np.log(psi ** 2 + DELTA)))
        dpsi = DT * (fb - ent) - DAMP * psi
        psi = psi + dpsi
    return psi


def psi0_batch(seeds, n=4):
    side = H.side_for_n(n)
    out = np.empty((len(seeds), side, side), dtype=np.float64)
    for i, s in enumerate(seeds):
        out[i] = H.psi0_xof(int(s), n)
    return out


def _assert_batch_matches_reference():
    s = 12345
    ref = H.evolve(H.psi0_xof(s, 4))
    bat = evolve_batch(psi0_batch([s]))[0]
    assert np.array_equal(ref, bat), "batched evolve diverges from reference!"


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    ckpt_every = 50_000
    if "--checkpoint-every" in sys.argv:
        ckpt_every = int(sys.argv[sys.argv.index("--checkpoint-every") + 1])
    batch = 2000
    n = 4
    side = H.side_for_n(n)
    ncells = side * side

    _assert_batch_matches_reference()
    t0 = time.time()

    seen_psi = {}      # psi*.tobytes() -> first seed
    seen_C = {}        # C hex -> first seed
    psi_collisions = []
    C_collisions = []
    nan_seeds, inf_seeds, huge_seeds = 0, 0, 0
    finite = 0

    # per-cell streaming histograms (clip to a wide range, 1024 bins) for entropy
    HIST_LO, HIST_HI, HIST_BINS = -1e4, 1e4, 4096
    cell_hist = np.zeros((ncells, HIST_BINS), dtype=np.int64)
    # rounded-fingerprint distinct counts (near-duplicate clustering)
    fp_sets = {0: set(), 1: set(), 2: set(), 3: set()}
    FP_CAP = 300_000  # cap fingerprint sets to bound memory
    # subsample of psi* vectors for downstream clustering / surrogate
    sub = []
    SUB_EVERY = max(1, N // 50_000)

    def checkpoint(done, final=False):
        distinct_C = len(seen_C)
        distinct_psi = len(seen_psi)
        # min-entropy from the most frequent collision bucket on C
        # (we only stored first-seed; collisions list length approximates extra hits)
        rec = {
            "params": {"alpha": ALPHA, "beta": BETA, "theta": THETA, "dt": DT,
                       "steps": STEPS, "damping": DAMP, "n": n, "side": side,
                       "path": "consensus/XOF (WLv3.1)"},
            "seeds_processed": done,
            "finite_seeds": finite,
            "nan_seeds": nan_seeds, "inf_seeds": inf_seeds, "huge_seeds_gt_1e6": huge_seeds,
            "exact_psistar_collisions": len(psi_collisions),
            "exact_commitment_collisions": len(C_collisions),
            "distinct_psistar": distinct_psi,
            "distinct_commitments": distinct_C,
            "psistar_collision_examples": psi_collisions[:10],
            "commitment_collision_examples": C_collisions[:10],
            "fingerprint_distinct": {k: len(v) for k, v in fp_sets.items()},
            "fingerprint_capped": {k: (len(v) >= FP_CAP) for k, v in fp_sets.items()},
            "elapsed_sec": round(time.time() - t0, 1),
            "rate_seeds_per_sec": round(done / max(1e-9, time.time() - t0), 1),
            "final": final,
        }
        if final:
            # effective entropy estimates
            # 1) distinct-commitment lower bound: log2(distinct C)
            rec["entropy_bits_distinct_C_lowerbound"] = round(math.log2(max(1, distinct_C)), 3)
            # 2) per-cell Shannon entropy (bits) from histograms
            cell_ent = []
            for c in range(ncells):
                h = cell_hist[c].astype(np.float64)
                tot = h.sum()
                if tot <= 0:
                    cell_ent.append(0.0); continue
                p = h[h > 0] / tot
                cell_ent.append(float(-(p * np.log2(p)).sum()))
            rec["per_cell_entropy_bits"] = [round(x, 3) for x in cell_ent]
            rec["per_cell_entropy_bits_sum"] = round(float(sum(cell_ent)), 3)
            rec["per_cell_entropy_bits_mean"] = round(float(np.mean(cell_ent)), 3)
            rec["input_entropy_bits_reference"] = 53 * ncells  # ideal one-way target
            if sub:
                arr = np.array(sub)
                np.save(os.path.join(ART, "a1_psistar_subsample.npy"), arr)
                rec["subsample_saved"] = list(arr.shape)
        name = "a1_entropy.json" if final else "a1_checkpoint.json"
        with open(os.path.join(ART, name), "w") as f:
            json.dump(rec, f, indent=2)
        return rec

    done = 0
    next_ckpt = ckpt_every
    for start in range(0, N, batch):
        seeds = range(start, min(start + batch, N))
        p0 = psi0_batch(seeds, n)
        ps = evolve_batch(p0)  # (B, side, side)
        flat = ps.reshape(len(seeds), ncells)
        for i, s in enumerate(seeds):
            v = ps[i]
            if np.isnan(v).any():
                nan_seeds += 1; done += 1; continue
            if np.isinf(v).any():
                inf_seeds += 1; done += 1; continue
            finite += 1
            if np.abs(v).max() > 1e6:
                huge_seeds += 1
            key = v.tobytes()
            if key in seen_psi:
                psi_collisions.append([seen_psi[key], int(s)])
            else:
                seen_psi[key] = int(s)
            C = hashlib.sha256(H.serialize(v)).hexdigest()
            if C in seen_C:
                C_collisions.append([seen_C[C], int(s)])
            else:
                seen_C[C] = int(s)
            # histograms
            idx = np.clip(((flat[i] - HIST_LO) / (HIST_HI - HIST_LO) * HIST_BINS),
                          0, HIST_BINS - 1).astype(np.int64)
            for c in range(ncells):
                cell_hist[c, idx[c]] += 1
            # rounded fingerprints
            for dp, st in fp_sets.items():
                if len(st) < FP_CAP:
                    st.add(tuple(np.round(flat[i], dp)))
            if s % SUB_EVERY == 0:
                sub.append(flat[i].copy())
            done += 1
        if done >= next_ckpt:
            r = checkpoint(done)
            print(f"[a1] {done}/{N} finite={finite} psi-coll={len(psi_collisions)} "
                  f"C-coll={len(C_collisions)} nan={nan_seeds} inf={inf_seeds} "
                  f"rate={r['rate_seeds_per_sec']}/s", flush=True)
            next_ckpt += ckpt_every

    rec = checkpoint(done, final=True)
    print("=== a1 attractor/entropy DONE ===")
    print(json.dumps({k: rec[k] for k in (
        "seeds_processed", "finite_seeds", "nan_seeds", "inf_seeds", "huge_seeds_gt_1e6",
        "exact_psistar_collisions", "exact_commitment_collisions",
        "distinct_commitments", "entropy_bits_distinct_C_lowerbound",
        "per_cell_entropy_bits_sum", "per_cell_entropy_bits_mean",
        "input_entropy_bits_reference", "fingerprint_distinct", "rate_seeds_per_sec",
    )}, indent=2))


if __name__ == "__main__":
    main()
