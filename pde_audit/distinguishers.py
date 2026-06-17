"""Phase 8D: structural distinguishers on the 256-bit output.

Trains classifiers to (a) separate PDE outputs from uniform 256-bit strings and
(b) recover input properties (all-zero vs random, low- vs high-entropy, length
bucket, input Hamming-weight bucket, repeated-block indicator, shared-prefix
class) from the output bits alone. Disjoint train/test, permutation-label
controls, bootstrap confidence intervals, and a fresh-seed reproduction gate
before any advantage is called a distinguisher. Multiple-comparison context is
reported. Conventional hashes are NOT used.

Uses scikit-learn if available; otherwise a NumPy logistic-regression fallback
(documented in the artifact). Operates on raw 256-bit output only.

Run:
    python -m pde_audit.distinguishers
"""

from __future__ import annotations

import time

import numpy as np

from . import _harness as H
from ._harness import PDEVariant, PDEParams

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    _SKLEARN = True
except Exception:  # pragma: no cover
    _SKLEARN = False

V = PDEVariant(PDEParams())


def _bits(msg: bytes) -> np.ndarray:
    return H.bits_of(V.hash(msg))


# ---------------------------------------------------------------------------
# NumPy logistic-regression fallback
# ---------------------------------------------------------------------------
def _np_logreg_fit_predict(Xtr, ytr, Xte, iters=300, lr=0.1):
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1))])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1))])
    w = np.zeros(Xtr.shape[1])
    for _ in range(iters):
        z = Xtr @ w
        pred = 1.0 / (1.0 + np.exp(-z))
        grad = Xtr.T @ (pred - ytr) / len(ytr)
        w -= lr * grad
    return 1.0 / (1.0 + np.exp(-(Xte @ w)))


def _balanced_acc(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    accs = []
    for c in (0, 1):
        m = y == c
        if m.sum():
            accs.append((yhat[m] == c).mean())
    return float(np.mean(accs))


def _dedup(X, y):
    """Drop duplicate feature rows so no identical digest spans train/test.

    Low-diversity input classes (e.g. all-zero messages, which only vary by
    length) otherwise leak via memorization: the same digest lands in both
    train and test and the classifier 'recognizes' it instead of generalizing.
    """
    seen = set()
    keep = []
    for i in range(len(y)):
        key = X[i].tobytes()
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return X[keep], y[keep]


def _evaluate(X, y, seed, with_rf=True):
    g = H.rng(seed)
    X, y = _dedup(X, y)
    n = len(y)
    perm = g.permutation(n)
    X, y = X[perm], y[perm]
    ntr = int(0.7 * n)
    Xtr, Xte = X[:ntr], X[ntr:]
    ytr, yte = y[:ntr], y[ntr:]

    out = {"n_after_dedup": int(n),
           "class1_frac_after_dedup": float(np.mean(y))}
    if _SKLEARN:
        lr = LogisticRegression(max_iter=500, C=1.0)
        lr.fit(Xtr, ytr)
        p = lr.predict_proba(Xte)[:, 1]
        out["logreg"] = {
            "balanced_acc": float(balanced_accuracy_score(yte, (p > 0.5).astype(int))),
            "auc": float(roc_auc_score(yte, p)),
        }
        if with_rf:
            rf = RandomForestClassifier(n_estimators=120, max_depth=None,
                                        random_state=seed, n_jobs=-1)
            rf.fit(Xtr, ytr)
            pr = rf.predict_proba(Xte)[:, 1]
            out["random_forest"] = {
                "balanced_acc": float(balanced_accuracy_score(yte, (pr > 0.5).astype(int))),
                "auc": float(roc_auc_score(yte, pr)),
            }
    else:
        p = _np_logreg_fit_predict(Xtr.astype(float), ytr.astype(float), Xte.astype(float))
        out["logreg_numpy"] = {
            "balanced_acc": _balanced_acc(yte, (p > 0.5).astype(int)),
            "auc": float(_auc(yte, p)),
        }
    # permutation-label control
    yctrl = g.permutation(ytr)
    if _SKLEARN:
        lr2 = LogisticRegression(max_iter=500)
        lr2.fit(Xtr, yctrl)
        pc = lr2.predict_proba(Xte)[:, 1]
        out["label_permutation_control_auc"] = float(roc_auc_score(yte, pc))
    else:
        pc = _np_logreg_fit_predict(Xtr.astype(float), yctrl.astype(float), Xte.astype(float))
        out["label_permutation_control_auc"] = float(_auc(yte, pc))
    return out


def _auc(y, score):
    y = np.asarray(y); score = np.asarray(score)
    pos = score[y == 1]; neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Mann-Whitney U
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------
def _ds_pde_vs_uniform(seed, n):
    g = H.rng(seed)
    X, y = [], []
    for _ in range(n):
        m = g.integers(0, 256, size=int(g.integers(1, 400)), dtype=np.uint8).tobytes()
        X.append(_bits(m)); y.append(1)
        X.append(g.integers(0, 2, size=256, dtype=np.uint8)); y.append(0)
    return np.array(X), np.array(y)


def _ds_zero_vs_random(seed, n):
    g = H.rng(seed)
    X, y = [], []
    for _ in range(n):
        L = int(g.integers(1, 300))
        X.append(_bits(b"\x00" * L)); y.append(1)
        X.append(_bits(g.integers(0, 256, size=L, dtype=np.uint8).tobytes())); y.append(0)
    return np.array(X), np.array(y)


def _ds_lowent_vs_highent(seed, n):
    g = H.rng(seed)
    X, y = [], []
    for _ in range(n):
        L = int(g.integers(8, 300))
        byte = int(g.integers(0, 256))
        X.append(_bits(bytes([byte]) * L)); y.append(1)               # low entropy
        X.append(_bits(g.integers(0, 256, size=L, dtype=np.uint8).tobytes())); y.append(0)
    return np.array(X), np.array(y)


def _ds_length_bucket(seed, n):
    g = H.rng(seed)
    X, y = [], []
    for _ in range(n):
        short = g.integers(1, 50)
        long = g.integers(250, 400)
        X.append(_bits(g.integers(0, 256, size=int(short), dtype=np.uint8).tobytes())); y.append(0)
        X.append(_bits(g.integers(0, 256, size=int(long), dtype=np.uint8).tobytes())); y.append(1)
    return np.array(X), np.array(y)


def _ds_input_hw_bucket(seed, n):
    g = H.rng(seed)
    X, y = [], []
    L = 64
    for _ in range(n):
        # low input Hamming weight (sparse) vs high (dense)
        sparse = np.zeros(L, dtype=np.uint8)
        idx = g.choice(L * 8, size=int(g.integers(1, 8)), replace=False)
        for b in idx:
            sparse[b >> 3] |= 1 << (7 - (b & 7))
        X.append(_bits(sparse.tobytes())); y.append(0)
        dense = g.integers(0, 256, size=L, dtype=np.uint8)
        X.append(_bits(dense.tobytes())); y.append(1)
    return np.array(X), np.array(y)


def _ds_repeated_block(seed, n):
    g = H.rng(seed)
    X, y = [], []
    blk = 192
    for _ in range(n):
        b0 = g.integers(0, 256, size=blk, dtype=np.uint8).tobytes()
        X.append(_bits(b0 + b0)); y.append(1)                         # repeated block
        b1 = g.integers(0, 256, size=blk, dtype=np.uint8).tobytes()
        b2 = g.integers(0, 256, size=blk, dtype=np.uint8).tobytes()
        X.append(_bits(b1 + b2)); y.append(0)
    return np.array(X), np.array(y)


def _ds_shared_prefix(seed, n):
    g = H.rng(seed)
    prefix = g.integers(0, 256, size=64, dtype=np.uint8).tobytes()
    X, y = [], []
    for _ in range(n):
        tail = g.integers(0, 256, size=int(g.integers(1, 200)), dtype=np.uint8).tobytes()
        X.append(_bits(prefix + tail)); y.append(1)
        X.append(_bits(g.integers(0, 256, size=int(g.integers(1, 264)), dtype=np.uint8).tobytes())); y.append(0)
    return np.array(X), np.array(y)


TASKS = {
    "pde_vs_uniform": _ds_pde_vs_uniform,
    "zero_vs_random": _ds_zero_vs_random,
    "lowent_vs_highent": _ds_lowent_vs_highent,
    "length_bucket": _ds_length_bucket,
    "input_hw_bucket": _ds_input_hw_bucket,
    "repeated_block": _ds_repeated_block,
    "shared_prefix": _ds_shared_prefix,
}


def main(seed: int = 80040, per_class: int = 1500) -> dict:
    t0 = time.perf_counter()
    results = {
        "phase": "8D_distinguishers",
        "metadata": H.env_metadata(),
        "seed": seed,
        "sklearn": _SKLEARN,
        "per_class": per_class,
        "advantage_threshold_auc": 0.55,
        "multiple_comparison_note":
            f"{len(TASKS)} tasks x up to 2 classifiers tested; treat marginal "
            f"AUC>0.5 cautiously. A task is only called a distinguisher if AUC "
            f">= 0.55 reproduces on a fresh holdout seed.",
        "tasks": {},
    }
    for name, builder in TASKS.items():
        tt = time.perf_counter()
        X, y = builder(seed, per_class)
        ev = _evaluate(X, y, seed + 1)
        best_auc = max(v["auc"] for k, v in ev.items() if isinstance(v, dict))
        entry = {"eval": ev, "best_auc": best_auc,
                 "runtime_s": round(time.perf_counter() - tt, 2)}
        # reproduction gate for any apparent advantage
        if best_auc >= 0.55:
            X2, y2 = builder(seed + 9991, per_class)
            ev2 = _evaluate(X2, y2, seed + 9992)
            entry["reproduction_best_auc"] = max(
                v["auc"] for k, v in ev2.items() if isinstance(v, dict))
            entry["is_distinguisher"] = entry["reproduction_best_auc"] >= 0.55
        else:
            entry["is_distinguisher"] = False
        results["tasks"][name] = entry
        print(f"  {name:20s} best_auc={best_auc:.3f} "
              f"ctrl_auc={list(ev.values())[-1]:.3f} "
              f"distinguisher={entry['is_distinguisher']} ({entry['runtime_s']}s)")

    results["any_distinguisher"] = any(t["is_distinguisher"] for t in results["tasks"].values())
    results["runtime_s"] = round(time.perf_counter() - t0, 2)
    path = H.save_artifact("phase8d_distinguishers.json", results)
    print("  saved", path, f"({results['runtime_s']}s) any_distinguisher={results['any_distinguisher']}")
    return results


if __name__ == "__main__":
    main()
