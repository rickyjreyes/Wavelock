#!/usr/bin/env python3
"""
ATTACK 6 — Floating-point determinism / cross-platform reproducibility.

WaveLock's security claim requires: same input => identical bytes => identical
commitment on every supported platform. WaveLock's PDE is a chaotic, expansive
map (see a2 Lyapunov). We test how fragile the commitment is:

  T1  Repeatability: same seed, repeated runs in-process  -> must match (sanity)
  T2  Threading/BLAS: same seed under different OMP/MKL/OPENBLAS thread counts
      (spawned subprocesses) -> compare commitment
  T3  ULP sensitivity: perturb ONE psi0 cell by 1 ULP (and by 1e-15, 1e-12)
      -> does the commitment change? how large is the resulting ULP-error in
      psi*?  This bounds how little cross-platform noise flips C.
  T4  Reassociation: recompute the per-step reductions / Laplacian via a
      mathematically-equivalent but differently-ordered expression (emulating a
      different BLAS/SIMD/compiler reassociation) -> does psi* (hence C) flip?
  T5  Documented GPU non-consensus guard (vendor admission), quoted.

Usage:  python audit/a6_determinism.py
Writes: audit/artifacts/a6_determinism.json
"""
from __future__ import annotations
import sys, os, json, hashlib, subprocess
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audit._wl as H

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ART, exist_ok=True)


def commit_of_seed(seed, n=4):
    d = H.wavelock(seed, n=n, legacy=False)
    return d["C"], d["psi_star"]


# ---- T4: reassociated kernel (same math, different float op order) ----
def lap_reassoc(x):
    # reference: -4x + roll(+1,0)+roll(-1,0)+roll(+1,1)+roll(-1,1)
    # reassociated: group neighbors first, different addition order/parenthesization
    nx = (np.roll(x, +1, 1) + np.roll(x, +1, 0))
    ny = (np.roll(x, -1, 1) + np.roll(x, -1, 0))
    return (ny + nx) - 4.0 * x


def evolve_reassoc(psi0):
    a, b, th = H.ALPHA, H.BETA, H.THETA
    eps, dl, dt, damp = H.EPSILON, H.DELTA, H.DT, H.DAMPING
    psi = np.asarray(psi0, dtype=np.float64).copy()
    for _ in range(H.STEPS):
        lap = lap_reassoc(psi)
        # reassociate the feedback denominator and the dpsi sum
        denom = (eps * np.exp(-b * psi ** 2)) + psi
        fb = (a * lap) / denom
        ent = th * (psi * lap_reassoc(np.log(psi ** 2 + dl)))
        dpsi = (-(damp * psi)) + dt * (fb - ent)
        psi = psi + dpsi
    return psi


def ulp_perturb(psi0, cell=(0, 0), n_ulp=1):
    p = psi0.copy()
    p[cell] = np.nextafter(p[cell], np.inf) if n_ulp > 0 else np.nextafter(p[cell], -np.inf)
    return p


def main():
    n = 4
    seed = 42
    out = {"params": {"n": n, "seed": seed, "path": "consensus/XOF (WLv3.1)"}}

    # T1 repeatability
    c1, ps1 = commit_of_seed(seed, n)
    c2, ps2 = commit_of_seed(seed, n)
    out["T1_repeatability"] = {
        "commitment_run1": c1, "commitment_run2": c2,
        "identical": c1 == c2,
        "note": "in-process determinism (necessary, not sufficient)",
    }

    # T2 threading / BLAS via subprocesses
    snippet = (
        "import sys,os;sys.path.insert(0,os.getcwd());"
        "import audit._wl as H;"
        f"d=H.wavelock({seed},n={n},legacy=False);print(d['C'])"
    )
    t2 = {}
    for threads in ("1", "2", "4", "8"):
        env = dict(os.environ,
                   OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   OPENBLAS_NUM_THREADS=threads, NUMEXPR_NUM_THREADS=threads,
                   VECLIB_MAXIMUM_THREADS=threads)
        r = subprocess.run([sys.executable, "-c", snippet],
                           capture_output=True, text=True, env=env, cwd=os.getcwd())
        t2[f"threads={threads}"] = r.stdout.strip() or ("ERR:" + r.stderr.strip()[-200:])
    vals = set(t2.values())
    out["T2_threading_blas"] = {
        "commitments": t2,
        "all_identical": len(vals) == 1,
        "note": "BLAS available in this env may be reference-only; result documents this host",
    }

    # T3 ULP sensitivity
    psi0 = H.psi0_xof(seed, n)
    base_ps = H.evolve(psi0)
    base_C = hashlib.sha256(H.serialize(base_ps)).hexdigest()
    t3 = {"base_commitment": base_C}
    for label, mut in (
        ("+1ULP@(0,0)", ulp_perturb(psi0, (0, 0), +1)),
        ("-1ULP@(0,0)", ulp_perturb(psi0, (0, 0), -1)),
        ("+1e-15@(1,1)", None),
        ("+1e-12@(2,2)", None),
    ):
        if mut is None:
            mut = psi0.copy()
            if "1e-15" in label:
                mut[1, 1] += 1e-15
            else:
                mut[2, 2] += 1e-12
        ps = H.evolve(mut)
        C = hashlib.sha256(H.serialize(ps)).hexdigest()
        # measure psi* divergence magnitude
        max_abs = float(np.max(np.abs(ps - base_ps)))
        rel = float(np.max(np.abs((ps - base_ps) / (np.abs(base_ps) + 1e-30))))
        t3[label] = {
            "input_delta_ulps_or_eps": label,
            "commitment_changed": C != base_C,
            "psistar_max_abs_change": max_abs,
            "psistar_max_rel_change": rel,
        }
    out["T3_ulp_sensitivity"] = t3

    # T4 reassociation
    ps_re = evolve_reassoc(psi0)
    C_re = hashlib.sha256(H.serialize(ps_re)).hexdigest()
    out["T4_reassociation"] = {
        "base_commitment": base_C,
        "reassociated_commitment": C_re,
        "commitment_changed": C_re != base_C,
        "psistar_max_abs_change": float(np.max(np.abs(ps_re - base_ps))),
        "psistar_max_rel_change": float(np.max(np.abs((ps_re - base_ps) / (np.abs(base_ps) + 1e-30)))),
        "note": ("Same mathematical kernel, different float op-order (emulates a "
                 "different BLAS/SIMD/compiler/numpy reassociation). A changed "
                 "commitment means independent conforming implementations disagree."),
    }

    # T5 vendor admission (read guard text from repo)
    guard = None
    try:
        src = open(os.path.join(os.getcwd(), "wavelock/chain/WaveLock.py")).read()
        i = src.find("GPU backend is non-consensus")
        if i != -1:
            guard = src[max(0, i - 200):i + 120]
    except Exception as e:
        guard = f"(could not read: {e})"
    out["T5_vendor_admission"] = {
        "gpu_guard_excerpt": guard,
        "note": "WaveLock itself rejects GPU-produced consensus commitments — an "
                "explicit admission that the kernel is not reproducible across backends.",
    }

    with open(os.path.join(ART, "a6_determinism.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=== a6 determinism ===")
    print("T1 in-process identical:", out["T1_repeatability"]["identical"])
    print("T2 threading all identical:", out["T2_threading_blas"]["all_identical"])
    print("T3 ULP sensitivity:")
    for k, v in t3.items():
        if k == "base_commitment":
            continue
        print(f"   {k}: commit_changed={v['commitment_changed']}  "
              f"psi*_max_abs_change={v['psistar_max_abs_change']:.3e}")
    print("T4 reassociation commitment_changed:",
          out["T4_reassociation"]["commitment_changed"],
          " psi*_max_abs_change=%.3e" % out["T4_reassociation"]["psistar_max_abs_change"])


if __name__ == "__main__":
    main()
