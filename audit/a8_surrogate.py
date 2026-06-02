#!/usr/bin/env python3
"""
ATTACK 9 — Neural / surrogate inversion.

Targets and what each actually means:
  * psi0 -> psi*      forward surrogate (how learnable is the map?)
  * psi* -> psi0      inverse surrogate (does seed info survive? = recovery if
                      psi* is known)
  * C   -> psi0       OUT OF SCOPE: this is exactly a SHA-256 preimage; no
                      surrogate can learn it (documented, not attempted).

We train ridge regression and a small MLP on (psi0, psi*) pairs and measure R^2
and per-cell recovery. If psi*->psi0 is learnable, an attacker who observes a
published psi* recovers the seed-equivalent input; if not, the PDE genuinely
destroys local invertibility (corroborating a2's ill-conditioning).

Usage:  python audit/a8_surrogate.py [NTRAIN]
Writes: audit/artifacts/a8_surrogate.json
"""
from __future__ import annotations
import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import audit._wl as H
from audit.a1_attractor_entropy import evolve_batch, psi0_batch
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def make_data(seeds, n=4):
    side = H.side_for_n(n)
    p0 = psi0_batch(seeds, n)
    ps = evolve_batch(p0)
    X0 = p0.reshape(len(seeds), side * side)
    Xs = ps.reshape(len(seeds), side * side)
    return X0, Xs


def r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum()
    return float(1 - ss_res / ss_tot)


def eval_models(Xtr, Ytr, Xte, Yte, clip=1e4):
    # clip targets to keep heavy tails from dominating; report on clipped space
    res = {}
    rid = Ridge(alpha=1.0).fit(Xtr, Ytr)
    res["ridge_R2"] = r2(Yte, rid.predict(Xte))
    sc = StandardScaler().fit(Xtr)
    mlp = MLPRegressor(hidden_layer_sizes=(256, 256), max_iter=300,
                       random_state=0, early_stopping=True)
    mlp.fit(sc.transform(Xtr), Ytr)
    res["mlp_R2"] = r2(Yte, mlp.predict(sc.transform(Xte)))
    return res


def main():
    NTR = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    NTE = 5000
    n = 4
    X0, Xs = make_data(list(range(NTR + NTE)), n)
    # robustly clip psi* (heavy tails) so regression metrics are meaningful
    Xs_c = np.clip(Xs, -1e3, 1e3)
    Xtr0, Xte0 = X0[:NTR], X0[NTR:]
    Xtrs, Xtes = Xs_c[:NTR], Xs_c[NTR:]

    out = {"params": {"n": n, "n_train": NTR, "n_test": NTE,
                      "psistar_clipped_to": 1e3}}

    # forward: psi0 -> psi*
    out["forward_psi0_to_psistar"] = eval_models(Xtr0, Xtrs, Xte0, Xtes)
    # inverse: psi* -> psi0   (the attack: recover the input from the field)
    inv = eval_models(Xtrs, Xtr0, Xtes, Xte0)
    # also report mean abs error vs trivial baseline (predict mean = 0.5)
    rid = Ridge(alpha=1.0).fit(Xtrs, Xtr0)
    pred0 = rid.predict(Xtes)
    mae = float(np.abs(pred0 - Xte0).mean())
    base_mae = float(np.abs(0.5 - Xte0).mean())  # predicting U[0,1) mean
    inv["ridge_MAE_psi0"] = mae
    inv["baseline_MAE_predict_0.5"] = base_mae
    inv["beats_baseline"] = mae < base_mae
    out["inverse_psistar_to_psi0"] = inv

    out["C_to_psi0"] = {
        "attempted": False,
        "reason": ("C->psi0 is a SHA-256 preimage; provably not learnable by a "
                   "surrogate. The meaningful surrogate target is psi*->psi0, "
                   "which presupposes the attacker already has psi* (not just C)."),
    }

    out["verdict"] = (
        "If inverse R^2 is high, the published psi* leaks the input. If low and "
        "it barely beats the predict-0.5 baseline, the PDE destroys local "
        "invertibility (consistent with a2's ~1e11 condition numbers). Either "
        "way, given only C the attacker still faces SHA-256 + seed brute force.")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "artifacts", "a8_surrogate.json"), "w") as f:
        json.dump(out, f, indent=2)

    print("=== a8 surrogate ===")
    print("forward psi0->psi*  R2 ridge/mlp:",
          round(out["forward_psi0_to_psistar"]["ridge_R2"], 4),
          round(out["forward_psi0_to_psistar"]["mlp_R2"], 4))
    print("inverse psi*->psi0  R2 ridge/mlp:",
          round(inv["ridge_R2"], 4), round(inv["mlp_R2"], 4))
    print(f"inverse ridge MAE(psi0)={mae:.4f} vs baseline(predict 0.5)={base_mae:.4f} "
          f"beats_baseline={inv['beats_baseline']}")


if __name__ == "__main__":
    main()
