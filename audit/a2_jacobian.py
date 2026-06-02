#!/usr/bin/env python3
"""
ATTACK 8 (+2) — Jacobian / local linearization + empirical Lyapunov exponent.

For a 16-dim state (n=4) we can compute the full Jacobian J = d psi*/d psi0 by
finite differences and study its spectrum. This tells us:
  * is the map expansive (sigma_max >> 1) -> chaos -> determinism fragility
  * is it singular/contractive in directions (sigma_min ~ 0) -> info loss / collapse
  * condition number -> feasibility of direct linear inversion
We also measure the empirical Lyapunov exponent (per-step growth of a tiny
perturbation) and, given psi*, attempt Newton/LM inversion psi*->psi0.

Usage:  python audit/a2_jacobian.py
Writes: audit/artifacts/a2_jacobian.json
"""
from __future__ import annotations
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audit._wl as H

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(ART, exist_ok=True)


def F(psi0_flat, side):
    return H.evolve(psi0_flat.reshape(side, side)).ravel()


def jacobian(psi0, side, h=1e-7):
    x0 = psi0.ravel().astype(np.float64)
    f0 = F(x0, side)
    m = x0.size
    J = np.zeros((m, m))
    for j in range(m):
        xp = x0.copy(); xp[j] += h
        xm = x0.copy(); xm[j] -= h
        J[:, j] = (F(xp, side) - F(xm, side)) / (2 * h)
    return J, f0


def lyapunov(psi0, side, eps=1e-12, steps=H.STEPS):
    """Average log growth rate of a random unit perturbation, renormalized."""
    a, b, th = H.ALPHA, H.BETA, H.THETA
    e, dl, dt, dp = H.EPSILON, H.DELTA, H.DT, H.DAMPING

    def step(psi):
        lap = H.wl.laplacian(psi)
        fb = a * lap / (psi + e * np.exp(-b * psi ** 2))
        ent = th * (psi * H.wl.laplacian(np.log(psi ** 2 + dl)))
        return psi + dt * (fb - ent) - dp * psi

    psi = psi0.copy()
    rng = np.random.default_rng(0)
    v = rng.standard_normal(psi.shape); v /= np.linalg.norm(v)
    pert = psi + eps * v
    total = 0.0
    for _ in range(steps):
        psi_n = step(psi)
        pert_n = step(pert)
        d = np.linalg.norm(pert_n - psi_n)
        total += np.log(d / eps)
        # renormalize
        pert = psi_n + (eps / d) * (pert_n - psi_n)
        psi = psi_n
    return total / steps  # per-step Lyapunov exponent (nats/step)


def main():
    n = 4
    side = H.side_for_n(n)
    out = {"params": {"n": n, "side": side, "dims": side * side}}
    seeds = [1, 42, 123, 777, 31337]
    per_seed = []
    for s in seeds:
        psi0 = H.psi0_xof(s, n)
        J, f0 = jacobian(psi0, side)
        sv = np.linalg.svd(J, compute_uv=False)
        lam = lyapunov(psi0, side)
        rec = {
            "seed": s,
            "sigma_max": float(sv[0]),
            "sigma_min": float(sv[-1]),
            "condition_number": float(sv[0] / max(sv[-1], 1e-300)),
            "log10_det_abs": float(np.sum(np.log10(np.maximum(sv, 1e-300)))),
            "effective_rank_1pct": int(np.sum(sv > 0.01 * sv[0])),
            "singular_values": [float(x) for x in sv],
            "lyapunov_per_step_nats": float(lam),
            "lyapunov_total_50steps_nats": float(lam * H.STEPS),
            "amplification_factor_e^L*50": float(np.exp(lam * H.STEPS)),
        }
        per_seed.append(rec)

    out["per_seed"] = per_seed
    out["summary"] = {
        "mean_sigma_max": float(np.mean([r["sigma_max"] for r in per_seed])),
        "mean_condition_number": float(np.mean([r["condition_number"] for r in per_seed])),
        "mean_lyapunov_per_step_nats": float(np.mean([r["lyapunov_per_step_nats"] for r in per_seed])),
        "interpretation": (
            "sigma_max>>1 and positive Lyapunov => expansive/chaotic map; a "
            "last-ULP (~1e-16) difference is amplified by ~exp(L*50), which "
            "explains the a6 reassociation commitment flip. Large condition "
            "number => direct linear inversion is ill-posed."),
    }

    # --- Newton/LM inversion attempt (given psi*, recover psi0) ---
    s = 42
    psi0 = H.psi0_xof(s, n)
    target = H.evolve(psi0).ravel()
    rng = np.random.default_rng(1)
    best = None
    for attempt in range(20):
        x = rng.random(side * side)  # start in [0,1) like a real psi0
        for _ in range(50):
            J, f = jacobian(x.reshape(side, side), side, h=1e-6)
            r = f - target
            try:
                dx = np.linalg.lstsq(J, r, rcond=None)[0]
            except np.linalg.LinAlgError:
                break
            x = x - dx
            if not np.all(np.isfinite(x)):
                break
        if np.all(np.isfinite(x)):
            err = float(np.linalg.norm(F(x, side) - target))
            seed_err = float(np.linalg.norm(x - psi0.ravel()))
            if best is None or err < best["residual"]:
                best = {"residual": err, "distance_to_true_psi0": seed_err,
                        "recovered_equals_true": err < 1e-6}
    out["newton_inversion_given_psistar"] = {
        "note": "Newton/LM in psi0-space given the true psi* (an upper bound on "
                "an attacker who already has psi*; the real attacker only has C).",
        "best": best,
    }

    with open(os.path.join(ART, "a2_jacobian.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=== a2 jacobian / lyapunov ===")
    for r in per_seed:
        print(f"seed={r['seed']:6d}  sigma_max={r['sigma_max']:.3e}  "
              f"sigma_min={r['sigma_min']:.3e}  cond={r['condition_number']:.3e}  "
              f"L/step={r['lyapunov_per_step_nats']:.3f}  "
              f"amp(e^L*50)={r['amplification_factor_e^L*50']:.3e}")
    print("Newton inversion best:", out["newton_inversion_given_psistar"]["best"])


if __name__ == "__main__":
    main()
