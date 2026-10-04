"""Compare demographic-only, SNP-only, and RandomOverSampler baselines across 50 folds."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from model_utils import NUM_FEATURES, apply_scaler, fit_scaler, model_specifications, predict_for_spec

N_FOLDS = 50

def load_dev(data_root: Path):
    X = np.load(data_root / "dev" / "X_real_features_171.npy").astype(np.float32)
    y = np.load(data_root / "dev" / "y_real.npy").astype(np.int64).ravel()
    y[y == 2] = 1
    if X.shape[1] != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, found {X.shape[1]}")
    return X, y


def real_class_weights(y):
    counts = np.bincount(y.astype(int), minlength=2)
    cw = 1.0 / counts
    w = cw[y.astype(int)]
    return w / (w.mean() + 1e-8)


def fit_lr_auc(xtr, ytr, wtr, xev, yev):
    m = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, solver="liblinear")
    m.fit(xtr, ytr, sample_weight=wtr)
    return float(roc_auc_score(yev, m.predict_proba(xev)[:, 1]))


def run(data_root: Path, selection_path: Path, results_dir: Path, max_folds: int | None):
    X, y = load_dev(data_root)
    with open(selection_path, "r", encoding="utf-8") as f:
        selection = json.load(f)
    selected_model = selection["selected_model"]
    specs = model_specifications()
    if selected_model not in specs:
        raise ValueError(f"Unknown selected model: {selected_model}")
    selected_spec = specs[selected_model]

    n_folds = N_FOLDS if max_folds is None else min(max_folds, N_FOLDS)
    aucs = {"Demographics-only Ridge LR": [], "SNP-only Ridge LR": [], "RandomOverSampler": []}

    for fold_idx in range(n_folds):
        fd = data_root / "folds" / f"fold_{fold_idx:02d}"
        tr_idx = np.load(fd / "train_idx.npy").astype(int)
        ev_idx = np.load(fd / "eval_idx.npy").astype(int)
        xtr_real, ytr = X[tr_idx].copy(), y[tr_idx].copy()
        xev_real, yev = X[ev_idx].copy(), y[ev_idx].copy()

        mean, std = fit_scaler(xtr_real)
        xtr = apply_scaler(xtr_real, mean, std)
        xev = apply_scaler(xev_real, mean, std)
        w = real_class_weights(ytr)

        # Age + education + gender.
        aucs["Demographics-only Ridge LR"].append(
            fit_lr_auc(xtr[:, :3], ytr, w, xev[:, :3], yev)
        )

        # All 168 SNPs, no demographics.
        aucs["SNP-only Ridge LR"].append(
            fit_lr_auc(xtr[:, 3:], ytr, w, xev[:, 3:], yev)
        )

        # Resampling baseline using the real training participants.
        ros = RandomOverSampler(random_state=42 + fold_idx)
        x_ros, y_ros = ros.fit_resample(xtr, ytr)
        w_ros = np.ones(len(y_ros), dtype=np.float32)
        probs = predict_for_spec(selected_spec, x_ros, y_ros, w_ros, xev,
                                 seed=42 + fold_idx)
        aucs["RandomOverSampler"].append(float(roc_auc_score(yev, probs)))

        print(f"fold {fold_idx:02d}: " + " | ".join(
            f"{k}={aucs[k][-1]:.3f}" for k in aucs
        ))

    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, vals in aucs.items():
        arr = np.asarray(vals)
        rows.append({
            "baseline": name,
            "mean_auc": float(arr.mean()),
            "sd_auc": float(arr.std()),
            "n_folds": int(len(arr)),
            "selected_model_for_ros": selected_model if name == "RandomOverSampler" else "",
        })
        print(f"{name}: {arr.mean():.3f} ({arr.std():.3f})")

    with open(results_dir / "baseline_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(results_dir / "baseline_fold_auc.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["baseline", "fold", "auc"])
        writer.writeheader()
        for name, vals in aucs.items():
            for fold, auc in enumerate(vals):
                writer.writerow({"baseline": name, "fold": fold, "auc": auc})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/apoe1")
    p.add_argument("--selection", default="results/selected_config.json")
    p.add_argument("--results-dir", default="results/baselines")
    p.add_argument("--max-folds", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(Path(a.data_root), Path(a.selection), Path(a.results_dir), a.max_folds)
