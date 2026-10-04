"""Compare model configurations with and without synthetic augmentation."""

import os
import csv
import json
import time
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from model_utils import (
    NUM_EPOCHS,
    apply_scaler,
    draw_synthetic_indices,
    fit_scaler,
    model_specifications,
    predict_for_spec,
    seed_everything,
)


# Configuration
AUGMENTATION_FACTORS = [3, 5, 7, 10]
NUM_RUNS = 50                 # 5-fold x 10 repeats, produced by prepare_data.py

DEV_DIR = "data/apoe1/dev"
FOLDS_ROOT = "data/apoe1/folds"
RESULTS_DIR = "results"
SELECTED_CONFIG_PATH = os.path.join(RESULTS_DIR, "selected_config.json")
OOF_PRED_PATH = os.path.join(RESULTS_DIR, "oof_dev_predictions.npz")
CV_STATS_PATH = os.path.join(RESULTS_DIR, "cv_pairwise_tests.csv")


def _midrank(values):
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        sorted_ranks[start:end] = 0.5 * (start + end - 1) + 1
        start = end
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = sorted_ranks
    return ranks


def delong_pvalue(y_true, probs_a, probs_b):
    """Return the paired DeLong p-value for two sets of predictions."""
    y_true = np.asarray(y_true).astype(int)
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    if probs_a.shape != probs_b.shape or probs_a.shape != y_true.shape:
        raise ValueError("Paired predictions and labels must have the same shape")

    positive = y_true == 1
    negative = y_true == 0
    m = int(positive.sum())
    n = int(negative.sum())
    if m < 2 or n < 2:
        return float("nan")

    predictions = np.vstack([probs_a, probs_b])
    tx = np.vstack([_midrank(row[positive]) for row in predictions])
    ty = np.vstack([_midrank(row[negative]) for row in predictions])
    tz = np.vstack([_midrank(row) for row in predictions])

    aucs = tz[:, positive].sum(axis=1) / (m * n) - (m + 1) / (2 * n)
    v01 = (tz[:, positive] - tx) / n
    v10 = 1.0 - (tz[:, negative] - ty) / m
    covariance = np.cov(v01) / m + np.cov(v10) / n
    variance = covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    if variance <= 0:
        return float("nan")

    z = (aucs[0] - aucs[1]) / np.sqrt(variance)
    return float(2 * norm.sf(abs(z)))


def holm_correction(pvalues):
    # Return Holm-adjusted p-values
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan, dtype=float)
    finite = np.isfinite(pvalues)
    if not finite.any():
        return adjusted
    vals = pvalues[finite]
    order = np.argsort(vals)
    m = len(vals)
    out = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * vals[idx]
        running_max = max(running_max, candidate)
        out[idx] = min(1.0, running_max)
    adjusted[np.where(finite)[0]] = out
    return adjusted


def load_dev():
    x_num = np.load(os.path.join(DEV_DIR, "X_num.npy")).astype(np.float32)
    x_cat = np.load(os.path.join(DEV_DIR, "X_cat.npy")).astype(np.float32)
    y = np.load(os.path.join(DEV_DIR, "y.npy")).flatten()
    y[y == 2] = 1                          # merge MCI(1) + AD(2) -> impaired(1)
    x = np.hstack([x_num, x_cat]).astype(np.float32)   # [Age, Edu, Gender, 168 SNPs] = 171
    return x, y


def run():
    seed_everything(42)
    X_dev, y_dev = load_dev()
    print(f"Development set: {X_dev.shape}  class balance {np.bincount(y_dev.astype(int))}")

    models = model_specifications()
    levels = [0] + AUGMENTATION_FACTORS
    # results[name][ratio] = list of per-run AUCs (paired across runs by index)
    results = {name: {k: [] for k in levels} for name in models}

    # Save out-of-fold(OOF) predictions
    oof_sum = {name: {k: np.zeros(len(X_dev), dtype=float) for k in levels} for name in models}
    oof_cnt = {name: {k: np.zeros(len(X_dev), dtype=int) for k in levels} for name in models}

    start = time.time()
    for run_idx in range(NUM_RUNS):
        fold_dir = os.path.join(FOLDS_ROOT, f"fold_{run_idx:02d}")
        train_idx = np.load(os.path.join(fold_dir, "train_idx.npy"))
        eval_idx = np.load(os.path.join(fold_dir, "eval_idx.npy"))

        x_tr_real, y_tr_real = X_dev[train_idx].copy(), y_dev[train_idx]
        x_ev, y_ev = X_dev[eval_idx].copy(), y_dev[eval_idx]

        # standardize using real training participants only
        mean, std = fit_scaler(x_tr_real)
        x_tr_s = apply_scaler(x_tr_real, mean, std)
        x_ev_s = apply_scaler(x_ev, mean, std)

        # class weights from the real training data
        cw = 1.0 / np.bincount(y_tr_real.astype(int))
        w0 = cw[y_tr_real.astype(int)]; w0 = w0 / (w0.mean() + 1e-8)

        # this fold-specific synthetic data
        synth = np.load(os.path.join(fold_dir, "synthetic_features.npy")).astype(np.float32)
        synth_y = np.load(os.path.join(fold_dir, "synthetic_y.npy")).flatten()
        synth_y[synth_y == 2] = 1

        if len(synth) != len(synth_y):
            raise ValueError(
                f"Run {run_idx}: synthetic features and labels have different lengths"
            )

        synth_s = apply_scaler(synth, mean, std)   # scale synthetic with real training stats

        # Keep the seed the same across models.
        fit_seed = 42 + run_idx

        for name, spec in models.items():
            # 0x baseline
            p0 = predict_for_spec(
                spec, x_tr_s, y_tr_real, w0, x_ev_s, seed=fit_seed
            )
            auc0 = roc_auc_score(y_ev, p0) if len(np.unique(y_ev)) > 1 else 0.0
            results[name][0].append(auc0)
            oof_sum[name][0][eval_idx] += p0
            oof_cnt[name][0][eval_idx] += 1

            for mult in AUGMENTATION_FACTORS:
                n_add = len(x_tr_real) * mult
                if len(synth_s) < n_add:
                    raise ValueError(
                        f"Run {run_idx}: need {n_add} synthetic records for {mult}x, "
                        f"but only {len(synth_s)} are available."
                    )
                # class-proportional draw, no replacement
                synth_idx = draw_synthetic_indices(
                    synth_y, n_add, seed=1000 + run_idx
                )
                x_aug = np.concatenate([x_tr_s, synth_s[synth_idx]])
                y_aug = np.concatenate([y_tr_real, synth_y[synth_idx]])
                # weights from real training frequencies, applied to all rows
                w_aug = cw[y_aug.astype(int)]; w_aug = w_aug / (w_aug.mean() + 1e-8)
                pk = predict_for_spec(
                    spec, x_aug, y_aug, w_aug, x_ev_s, seed=fit_seed
                )
                auck = roc_auc_score(y_ev, pk) if len(np.unique(y_ev)) > 1 else 0.0
                results[name][mult].append(auck)
                oof_sum[name][mult][eval_idx] += pk
                oof_cnt[name][mult][eval_idx] += 1

        if (run_idx + 1) % 5 == 0:
            print(f"  completed run {run_idx + 1}/{NUM_RUNS}  ({time.time()-start:.0f}s)")

    # Summary with mean (SD) per ratio
    print("\n=== Model Comparison Summary (mean (SD) AUC across the 50 runs) ===")
    header = ["Model"] + [f"{k}x AUC" for k in levels]
    print("".join(f"{value:>16}" for value in header))
    cell = lambda arr: f"{np.mean(arr):.3f} ({np.std(arr):.3f})"
    for name in models:
        print(f"{name:<28}" + "".join(f"{cell(results[name][k]):>16}" for k in levels))

    # Average the repeated OOF predictions for each participant.
    def averaged_oof(name, ratio):
        count = oof_cnt[name][ratio]
        if np.any(count == 0):
            missing = int(np.sum(count == 0))
            raise RuntimeError(f"{missing} development participants have no OOF prediction for "
                               f"{name} at {ratio}x.")
        return oof_sum[name][ratio] / count

    oof_predictions = {
        name: {ratio: averaged_oof(name, ratio) for ratio in levels}
        for name in models
    }

    # Compare each augmented condition with its real-only counterpart.
    test_rows = []
    raw_pvalues = []
    for name in models:
        base = np.asarray(results[name][0], dtype=float)
        for mult in AUGMENTATION_FACTORS:
            aug = np.asarray(results[name][mult], dtype=float)
            delta = float(np.mean(aug - base))
            p_raw = delong_pvalue(
                y_dev,
                oof_predictions[name][0],
                oof_predictions[name][mult],
            )
            test_rows.append({
                "model": name,
                "ratio": int(mult),
                "mean_delta_auc": delta,
                "p_delong_raw": p_raw,
            })
            raw_pvalues.append(p_raw)

    p_holm = holm_correction(raw_pvalues)
    for row, p_adj in zip(test_rows, p_holm):
        row["p_delong_holm"] = float(p_adj) if np.isfinite(p_adj) else float('nan')

    print("\n=== Paired DeLong tests vs 0x (Holm-adjusted across 32 tests) ===")
    print(f"{'Model':<28}{'Ratio':>8}{'mean delta':>12}{'p_raw':>12}{'p_Holm':>12}")
    for row in test_rows:
        p_raw_str = f"{row['p_delong_raw']:.4g}" if np.isfinite(row['p_delong_raw']) else "n/a"
        p_holm_str = f"{row['p_delong_holm']:.4g}" if np.isfinite(row['p_delong_holm']) else "n/a"
        ratio_label = f"{row['ratio']}x"
        print(f"{row['model']:<28}{ratio_label:>8}{row['mean_delta_auc']:>+12.3f}"
              f"{p_raw_str:>12}{p_holm_str:>12}")

    # Select the model/ratio with the highest mean cross-validated AUC.
    selected_model, selected_ratio, selected_auc = max(
        ((name, k, float(np.mean(results[name][k])))
         for name in models for k in AUGMENTATION_FACTORS),
        key=lambda t: t[2],
    )
    selected_test = next(r for r in test_rows
                         if r["model"] == selected_model and r["ratio"] == selected_ratio)

    oof_real = oof_predictions[selected_model][0]
    oof_selected = oof_predictions[selected_model][selected_ratio]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(CV_STATS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "ratio", "mean_delta_auc", "p_delong_raw", "p_delong_holm"])
        writer.writeheader()
        writer.writerows(test_rows)

    with open(SELECTED_CONFIG_PATH, "w") as f:
        json.dump({
            "selected_model": selected_model,
            "selected_ratio": int(selected_ratio),
            "selected_mean_cv_auc": selected_auc,
            "selected_mean_cv_auc_0x": float(np.mean(results[selected_model][0])),
            "selected_mean_delta_auc": selected_test["mean_delta_auc"],
            "selected_delong_p_raw": selected_test["p_delong_raw"],
            "selected_delong_p_holm": selected_test["p_delong_holm"],
            "multiplicity_family_size": len(test_rows),
            "multiple_testing_method": "Holm",
            "num_cv_runs": NUM_RUNS,
            "nn_epochs": NUM_EPOCHS,
        }, f, indent=2)

    np.savez(
        OOF_PRED_PATH,
        y_dev=y_dev.astype(int),
        selected_model=np.array(selected_model),
        selected_ratio=np.array(selected_ratio, dtype=int),
        oof_real_only=oof_real.astype(float),
        oof_selected=oof_selected.astype(float),
        oof_count_real=oof_cnt[selected_model][0].astype(int),
        oof_count_selected=oof_cnt[selected_model][selected_ratio].astype(int),
    )

    print(f"\nSelected configuration: {selected_model} @ {selected_ratio}x "
          f"(mean CV AUC {selected_auc:.3f})")
    print(f"Selected DeLong test: raw p={selected_test['p_delong_raw']:.4g}, "
          f"Holm-adjusted p={selected_test['p_delong_holm']:.4g}")
    print(f"Saved 32-test table to {CV_STATS_PATH}")
    print(f"Saved selected configuration to {SELECTED_CONFIG_PATH}")
    print(f"Saved repeated-OOF development predictions to {OOF_PRED_PATH}")

    print(f"\nTotal time: {(time.time()-start)/60:.1f} min")


if __name__ == "__main__":
    run()
