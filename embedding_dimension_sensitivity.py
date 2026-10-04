"""Compare embedding dimensions for the selected late-fusion model."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model_utils import (  # noqa: E402
    NUM_FEATURES,
    apply_scaler,
    draw_synthetic_indices,
    fit_scaler,
    train_nn_predict,
)

EMBED_DIMS = [4, 8, 16]
N_FOLDS = 50


def class_weights(y):
    counts = np.bincount(y.astype(int), minlength=2)
    cw = 1.0 / counts
    w = cw[y.astype(int)]
    return cw, w / (w.mean() + 1e-8)


def run(data_root: Path, selection_path: Path, results_dir: Path, max_folds: int | None):
    with open(selection_path, "r", encoding="utf-8") as f:
        selected = json.load(f)
    ratio = int(selected["selected_ratio"])

    X = np.load(data_root / "dev" / "X_real_features_171.npy").astype(np.float32)
    y = np.load(data_root / "dev" / "y_real.npy").astype(np.int64).ravel()
    y[y == 2] = 1
    if X.shape[1] != NUM_FEATURES:
        raise ValueError(f"Expected {NUM_FEATURES} features, found {X.shape[1]}")

    n_folds = N_FOLDS if max_folds is None else min(max_folds, N_FOLDS)
    rows = []

    for fold_idx in range(n_folds):
        fd = data_root / "folds" / f"fold_{fold_idx:02d}"
        tr_idx = np.load(fd / "train_idx.npy").astype(int)
        ev_idx = np.load(fd / "eval_idx.npy").astype(int)
        Xs = np.load(fd / "synthetic_features.npy").astype(np.float32)
        ys = np.load(fd / "synthetic_y.npy").astype(np.int64).ravel()
        ys[ys == 2] = 1

        if len(Xs) != len(ys):
            raise ValueError(
                f"Fold {fold_idx}: synthetic features and labels have different lengths"
            )

        xtr_real, ytr = X[tr_idx].copy(), y[tr_idx].copy()
        xev_real, yev = X[ev_idx].copy(), y[ev_idx].copy()
        mean, std = fit_scaler(xtr_real)
        xtr = apply_scaler(xtr_real, mean, std)
        xev = apply_scaler(xev_real, mean, std)
        cw, w0 = class_weights(ytr)

        required = len(xtr_real) * ratio
        if len(Xs) < required:
            raise ValueError(f"Fold {fold_idx}: need {required} synthetic records for {ratio}x, found {len(Xs)}")
        # Use the same class-proportional draw as model_comparison.py.
        idx = draw_synthetic_indices(ys, required, seed=1000 + fold_idx)
        xaug = np.concatenate([xtr, apply_scaler(Xs[idx], mean, std)], axis=0)
        yaug = np.concatenate([ytr, ys[idx]], axis=0)
        waug = cw[yaug.astype(int)]
        waug = waug / (waug.mean() + 1e-8)

        for dim in EMBED_DIMS:
            p0 = train_nn_predict("LATE", "CATEGORICAL", xtr, ytr, w0, xev,
                                  seed=42 + fold_idx, embedding_dim=dim)
            pa = train_nn_predict("LATE", "CATEGORICAL", xaug, yaug, waug, xev,
                                  seed=42 + fold_idx, embedding_dim=dim)
            a0 = float(roc_auc_score(yev, p0))
            aa = float(roc_auc_score(yev, pa))
            rows.append({"fold": fold_idx, "embedding_dim": dim, "ratio": 0, "auc": a0})
            rows.append({"fold": fold_idx, "embedding_dim": dim, "ratio": ratio, "auc": aa})
            print(f"fold {fold_idx:02d} dim={dim}: 0x={a0:.3f} | {ratio}x={aa:.3f}")

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "embedding_fold_auc.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "embedding_dim", "ratio", "auc"])
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for dim in EMBED_DIMS:
        for r in [0, ratio]:
            vals = np.asarray([x["auc"] for x in rows if x["embedding_dim"] == dim and x["ratio"] == r])
            summary.append({
                "embedding_dim": dim,
                "ratio": r,
                "mean_auc": float(vals.mean()),
                "sd_auc": float(vals.std()),
                "n_folds": int(len(vals)),
            })
            print(f"dim={dim}, {r}x: {vals.mean():.3f} ({vals.std():.3f})")

    with open(results_dir / "embedding_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/apoe1")
    p.add_argument("--selection", default="results/selected_config.json")
    p.add_argument("--results-dir", default="results/embedding_sensitivity")
    p.add_argument("--max-folds", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(Path(a.data_root), Path(a.selection), Path(a.results_dir), a.max_folds)
