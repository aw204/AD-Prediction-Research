"""Evaluate the selected model on the held-out cohort."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from scipy.stats import norm

from model_utils import (
    apply_scaler,
    draw_synthetic_indices,
    fit_scaler,
    train_nn_predict,
)

# Settings
NUM_SEEDS = 20
NUM_BOOTSTRAPS = 5000

FEATURES_DIR = "data/apoe1/dev"
SPLITS_DIR = "data/apoe1"
OUT_DIR = "figures"

# Outputs from model_comparison.py
RESULTS_DIR = "results"
OOF_PRED_PATH = os.path.join(RESULTS_DIR, "oof_dev_predictions.npz")
HELDOUT_RESULTS_DIR = os.path.join(RESULTS_DIR, "heldout")
HELDOUT_RESULTS_PATH = os.path.join(HELDOUT_RESULTS_DIR, "heldout_results.json")
HELDOUT_PRED_PATH = os.path.join(HELDOUT_RESULTS_DIR, "heldout_predictions.npz")


def load_heldout():
    # Load held-out features and labels
    heldout_dir = os.path.join(SPLITS_DIR, "heldout")
    x_numerical = np.load(os.path.join(heldout_dir, "X_num.npy")).astype(np.float32)
    x_categorical = np.load(
        os.path.join(heldout_dir, "X_cat.npy")
    ).astype(np.float32)
    y_heldout = np.load(os.path.join(heldout_dir, "y.npy")).flatten()

    # Combine demographic and SNP features.
    x_heldout = np.hstack([x_numerical, x_categorical]).astype(np.float32)
    # Merge MCI and AD into the impaired class.
    y_heldout[y_heldout == 2] = 1
    return x_heldout, y_heldout


def youden_threshold(y_true, predictions):
    # Choose the cutoff with the largest Youden J
    false_pos_rate, true_pos_rate, thresholds = roc_curve(y_true, predictions)
    return thresholds[np.argmax(true_pos_rate - false_pos_rate)]


def metrics_at_threshold(y_true, predictions, threshold):
    matrix = confusion_matrix(
        y_true,
        (predictions >= threshold).astype(int),
        labels=[0, 1],
    )
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    auc = roc_auc_score(y_true, predictions) if len(np.unique(y_true)) > 1 else float('nan')
    return matrix, sensitivity, specificity, auc


def bca_interval(bootstrap_values, observed_value, jackknife_values, alpha=0.05):
    bootstrap_values = np.asarray(bootstrap_values, dtype=float)
    jackknife_values = np.asarray(jackknife_values, dtype=float)
    if not np.all(np.isfinite(bootstrap_values)):
        raise ValueError("bootstrap_values contains non-finite entries.")
    if not np.all(np.isfinite(jackknife_values)):
        raise ValueError("jackknife_values contains non-finite entries.")

    # Bias correction.
    less = np.sum(bootstrap_values < observed_value)
    equal = np.sum(bootstrap_values == observed_value)
    proportion = (less + 0.5 * equal) / len(bootstrap_values)
    proportion = np.clip(
        proportion,
        1 / (2 * len(bootstrap_values)),
        1 - 1 / (2 * len(bootstrap_values)),
    )
    z0 = norm.ppf(proportion)

    # Jackknife acceleration.
    jack_mean = np.mean(jackknife_values)
    differences = jack_mean - jackknife_values
    denominator = 6.0 * np.sum(differences ** 2) ** 1.5
    acceleration = (
        np.sum(differences ** 3) / denominator
        if denominator > 0
        else 0.0
    )

    z_low = norm.ppf(alpha / 2)
    z_high = norm.ppf(1 - alpha / 2)

    def adjusted_quantile(z_alpha):
        denominator = 1 - acceleration * (z0 + z_alpha)
        if denominator == 0:
            return 0.0 if z0 + z_alpha < 0 else 1.0
        adjusted = norm.cdf(
            z0 + (z0 + z_alpha) / denominator
        )
        return float(np.clip(adjusted, 0.0, 1.0))

    quantiles = sorted([adjusted_quantile(z_low), adjusted_quantile(z_high)])
    lower, upper = np.quantile(bootstrap_values, quantiles)
    return float(lower), float(upper)


def paired_participant_bootstrap(y_true, probs_a, thr_a, probs_b, thr_b,
                                 n_boot=NUM_BOOTSTRAPS, seed=42):
    y_true = np.asarray(y_true).astype(int)
    probs_a = np.asarray(probs_a, dtype=float)
    probs_b = np.asarray(probs_b, dtype=float)
    if probs_a.shape != probs_b.shape:
        raise ValueError(f"Probability matrices must have identical shape; got {probs_a.shape} and {probs_b.shape}.")
    if probs_a.ndim != 2 or probs_a.shape[1] != len(y_true):
        raise ValueError(
            f"Expected probability matrices shaped (n_seeds, {len(y_true)}); got {probs_a.shape}.")

    rng = np.random.default_rng(seed)
    class_indices = {cls: np.where(y_true == cls)[0] for cls in np.unique(y_true)}
    if set(class_indices) != {0, 1}:
        raise ValueError(f"Unexpected binary held-out labels.")

    keys = ["auc", "sensitivity", "specificity"]
    real_bootstrap = {k: [] for k in keys}
    augmented_bootstrap = {k: [] for k in keys}
    difference_bootstrap = {k: [] for k in keys}

    for _ in range(n_boot):
        # Resample each class separately.
        idx_parts = [
            rng.choice(class_indices[cls], size=len(class_indices[cls]), replace=True)
            for cls in (0, 1)
        ]
        idx = np.concatenate(idx_parts)
        rng.shuffle(idx)
        yb = y_true[idx]

        seed_metrics_a = []
        seed_metrics_b = []
        for s in range(probs_a.shape[0]):
            _, sensitivity_a, specificity_a, auc_a = metrics_at_threshold(
                yb, probs_a[s, idx], thr_a
            )
            _, sensitivity_b, specificity_b, auc_b = metrics_at_threshold(
                yb, probs_b[s, idx], thr_b
            )
            seed_metrics_a.append((auc_a, sensitivity_a, specificity_a))
            seed_metrics_b.append((auc_b, sensitivity_b, specificity_b))

        mean_a = np.mean(np.asarray(seed_metrics_a, dtype=float), axis=0)
        mean_b = np.mean(np.asarray(seed_metrics_b, dtype=float), axis=0)
        for i, k in enumerate(keys):
            real_bootstrap[k].append(float(mean_a[i]))
            augmented_bootstrap[k].append(float(mean_b[i]))
            difference_bootstrap[k].append(float(mean_b[i] - mean_a[i]))

    def mean_metrics(probs, threshold, y_values, indices=None):
        if indices is None:
            selected_probs = probs
        else:
            selected_probs = probs[:, indices]

        values = []
        for s in range(selected_probs.shape[0]):
            _, sensitivity, specificity, auc = metrics_at_threshold(
                y_values, selected_probs[s], threshold
            )
            values.append((auc, sensitivity, specificity))
        return np.mean(np.asarray(values, dtype=float), axis=0)

    observed_a = mean_metrics(probs_a, thr_a, y_true)
    observed_b = mean_metrics(probs_b, thr_b, y_true)
    observed_d = observed_b - observed_a

    # Jackknife values.
    jack_a = np.empty((len(y_true), len(keys)), dtype=float)
    jack_b = np.empty((len(y_true), len(keys)), dtype=float)
    jack_d = np.empty((len(y_true), len(keys)), dtype=float)

    for i in range(len(y_true)):
        keep = np.ones(len(y_true), dtype=bool)
        keep[i] = False
        y_jack = y_true[keep]
        a_jack = mean_metrics(probs_a, thr_a, y_jack, np.where(keep)[0])
        b_jack = mean_metrics(probs_b, thr_b, y_jack, np.where(keep)[0])
        jack_a[i] = a_jack
        jack_b[i] = b_jack
        jack_d[i] = b_jack - a_jack

    def ci_bca(samples, observed, jackknife):
        return {
            k: bca_interval(samples[k], observed[i], jackknife[:, i])
            for i, k in enumerate(keys)
        }

    return (
        ci_bca(real_bootstrap, observed_a, jack_a),
        ci_bca(augmented_bootstrap, observed_b, jack_b),
        ci_bca(difference_bootstrap, observed_d, jack_d),
    )


def oof_threshold_from_config():
    # Read thresholds from development predictions
    if not os.path.exists(OOF_PRED_PATH):
        raise FileNotFoundError(
            f"Missing {OOF_PRED_PATH}. Run model_comparison.py first.")

    data = np.load(OOF_PRED_PATH, allow_pickle=True)
    y_dev = data["y_dev"].astype(int)
    y_dev = np.where(y_dev == 2, 1, y_dev)

    def get_threshold(probs):
        probs = np.asarray(probs, dtype=float)
        valid = np.isfinite(probs)
        if valid.sum() != len(y_dev):
            raise RuntimeError(f"OOF probability vector has {len(y_dev) - int(valid.sum())} missing values.")
        return float(youden_threshold(y_dev[valid], probs[valid]))

    ratio = int(np.asarray(data["selected_ratio"]).item())
    selected_model = str(np.asarray(data["selected_model"]).item()) if "selected_model" in data else "Late Fusion (Categorical)"
    return (
        get_threshold(data["oof_real_only"]),
        get_threshold(data["oof_selected"]),
        ratio,
        selected_model,
    )


def save_tif(fig, path, width_mm=180, dpi=300):
    # Save figure as RGB TIFF.
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    im = Image.open(buf).convert("RGB")
    target_w = round(width_mm / 25.4 * dpi)
    target_h = round(im.height * target_w / im.width)
    im = im.resize((target_w, target_h), Image.LANCZOS)
    im.save(path, compression="tiff_lzw", dpi=(dpi, dpi))


def plot_confusion_matrices(results):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.weight': 'bold',
        'axes.labelweight': 'bold'
    })

    labels = ["CN", "Impaired"]
    tags = list(results.keys())
    real_tag = next(t for t in tags if t.startswith("0x"))
    aug_tag = next(t for t in tags if t != real_tag)
    panels = [
        (f"Real Data Only ({real_tag.split()[0]})", real_tag, "Blues"),
        (f"Synthetic Augmented ({aug_tag.split()[0]})", aug_tag, "Oranges"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (panel_title, tag, colormap) in zip(axes, panels):
        proportions = results[tag]["proportions"]
        sensitivity = results[tag]["sensitivity"]
        specificity = results[tag]["specificity"]
        auc = results[tag]["auc"]
        auc_ci = results[tag]["ci"]["auc"]

        image = ax.imshow(proportions, cmap=colormap, vmin=0, vmax=1)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        # Set text color based on cell value.
        for row in range(2):
            for col in range(2):
                value = proportions[row, col]
                ax.text(col, row, f"{value:.2f}", ha='center', va='center',
                        fontsize=34, fontweight='bold',
                        color='white' if value > 0.5 else '#333333')

        ax.set_title(f"{panel_title}\nAUC: {auc:.3f}  (95% CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f})\n"
                     f"Sensitivity: {sensitivity:.2f}   Specificity: {specificity:.2f}",
                     fontsize=18, fontweight='bold', pad=18)
        ax.set_xticks([0, 1], labels, fontsize=15, fontweight='bold')
        ax.set_yticks([0, 1], labels, fontsize=15, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=17, fontweight='bold')
        ax.set_ylabel("Actual", fontsize=17, fontweight='bold')

    fig.suptitle("Late Fusion (Categorical): Real Data vs Synthetic Augmentation",
                 fontsize=26, fontweight='bold', y=1.02)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    save_path = os.path.join(OUT_DIR, "heldout_confusion_matrices.tif")
    save_tif(fig, save_path)
    plt.close(fig)
    return save_path


def run():
    X_real = np.load(os.path.join(FEATURES_DIR, "X_real_features_171.npy")).astype(np.float32)
    y_real = np.load(os.path.join(FEATURES_DIR, "y_real.npy")).flatten()
    # Merge MCI and AD into the impaired class.
    y_real[y_real == 2] = 1

    # Load synthetic data.
    final_synth_path = os.path.join(SPLITS_DIR, "final_dev", "synthetic_features.npy")
    final_ysynth_path = os.path.join(SPLITS_DIR, "final_dev", "synthetic_y.npy")
    if not (os.path.exists(final_synth_path) and os.path.exists(final_ysynth_path)):
        raise FileNotFoundError(
            f"Missing final development generator output at {final_synth_path}. Run prepare_data.py "
            "then the final_dev generation in run_nested_generation.sh before held-out evaluation.")
    X_synth = np.load(final_synth_path).astype(np.float32)
    y_synth = np.load(final_ysynth_path).flatten()
    # Merge MCI and AD into the impaired class.
    y_synth[y_synth == 2] = 1

    x_heldout, y_heldout = load_heldout()

    # Load the selected model settings and thresholds.
    thr_real, thr_aug, aug_factor, selected_model = oof_threshold_from_config()
    if selected_model != "Late Fusion (Categorical)":
        raise RuntimeError(
            f"Expected Late Fusion (Categorical), got {selected_model}"
        )
    print(f"Using development OOF Youden thresholds (real={thr_real:.3f}, aug={thr_aug:.3f}) "
          f"and selected ratio {aug_factor}x from {OOF_PRED_PATH}.")

    print("\nLate Fusion Categorical: held-out evaluation (real-only vs augmented)")
    print(f"Development cohort: {X_real.shape}  class balance {np.bincount(y_real.astype(int))}")
    print(f"Held-out set:       {x_heldout.shape}  class balance {np.bincount(y_heldout.astype(int))}\n")

    # Fit the scaler using real development data.
    mean, std = fit_scaler(X_real)
    x_train_scaled = apply_scaler(X_real, mean, std)
    x_heldout_scaled = apply_scaler(x_heldout, mean, std)

    # Weight classes by inverse frequency.
    class_weights_real = 1.0 / np.bincount(y_real.astype(int))
    weights_baseline = class_weights_real[y_real.astype(int)]
    weights_baseline = weights_baseline / (weights_baseline.mean() + 1e-8)

    # Draw synthetic samples.
    required = len(X_real) * aug_factor
    if len(X_synth) < required:
        raise ValueError(
            f"Need {required} synthetic records for {aug_factor}x, "
            f"but only {len(X_synth)} are available."
        )
    # Use the same sampling rule as development.
    synth_indices = draw_synthetic_indices(y_synth, required, seed=42)

    x_augmented = np.concatenate([x_train_scaled, apply_scaler(X_synth[synth_indices], mean, std)])
    y_augmented = np.concatenate([y_real, y_synth[synth_indices]])
    weights_augmented = class_weights_real[y_augmented.astype(int)]
    weights_augmented = weights_augmented / (weights_augmented.mean() + 1e-8)

    conditions = {
        "0x (real only)": (x_train_scaled, y_real, weights_baseline, thr_real),
        f"{aug_factor}x (augmented)": (x_augmented, y_augmented, weights_augmented, thr_aug),
    }

    results = {}
    condition_probs = {}
    condition_thr = {}

    for tag, (x_train, y_train, sample_weights, fixed_threshold) in conditions.items():
        print("\n" + "=" * 70)
        print(f"{tag}  --  per-seed runs")
        print("=" * 70)

        seed_sensitivity, seed_specificity, seed_auc, seed_matrices = [], [], [], []
        seed_heldout_probs = []

        for seed in range(NUM_SEEDS):
            heldout_predictions = train_nn_predict(
                "LATE", "CATEGORICAL",
                x_train, y_train, sample_weights, x_heldout_scaled,
                seed=seed,
            )
            matrix, sensitivity, specificity, auc = metrics_at_threshold(
                y_heldout, heldout_predictions, fixed_threshold
            )
            seed_sensitivity.append(sensitivity)
            seed_specificity.append(specificity)
            seed_auc.append(auc)
            seed_matrices.append(matrix)
            seed_heldout_probs.append(heldout_predictions)

        mean_sensitivity = float(np.mean(seed_sensitivity))
        mean_specificity = float(np.mean(seed_specificity))
        mean_auc = float(np.mean(seed_auc))
        point_threshold = float(fixed_threshold)

        condition_probs[tag] = np.stack(seed_heldout_probs, axis=0)
        condition_thr[tag] = point_threshold

        print("\n" + "-" * 70)
        print(f"SUMMARY for {tag}  (threshold = {point_threshold:.3f}):")
        print(f"  Training variability (mean +/- SD across {NUM_SEEDS} seeds):")
        print(f"    Sensitivity: {mean_sensitivity:.3f} +/- {np.std(seed_sensitivity):.3f}")
        print(f"    Specificity: {mean_specificity:.3f} +/- {np.std(seed_specificity):.3f}")
        print(f"    AUC:         {mean_auc:.3f} +/- {np.std(seed_auc):.3f}")
        print("-" * 70)

        mean_matrix = np.mean(seed_matrices, axis=0)
        proportions = mean_matrix / mean_matrix.sum(axis=1, keepdims=True)
        results[tag] = {
            "proportions": proportions,
            "sensitivity": mean_sensitivity,
            "specificity": mean_specificity,
            "auc": mean_auc,
            "sd_sensitivity": float(np.std(seed_sensitivity)),
            "sd_specificity": float(np.std(seed_specificity)),
            "sd_auc": float(np.std(seed_auc)),
            "threshold": point_threshold,
        }

    # Reuse bootstrap samples for both conditions.
    tags = list(conditions.keys())
    ci_real, ci_aug, ci_delta = paired_participant_bootstrap(
        y_heldout,
        condition_probs[tags[0]], condition_thr[tags[0]],
        condition_probs[tags[1]], condition_thr[tags[1]],
        n_boot=NUM_BOOTSTRAPS,
    )
    results[tags[0]]["ci"] = ci_real
    results[tags[1]]["ci"] = ci_aug

    print("\n" + "=" * 70)
    print(f"Participant-level 95% BCa CIs ({NUM_BOOTSTRAPS} paired stratified bootstrap resamples)")
    print("=" * 70)
    for tag, ci in [(tags[0], ci_real), (tags[1], ci_aug)]:
        print(f"{tag}:")
        for m in ["auc", "sensitivity", "specificity"]:
            print(f"    {m:<12} [{ci[m][0]:.3f}, {ci[m][1]:.3f}]")
    print("\nPaired difference (augmented - real only), 95% BCa CI:")
    point_delta = {
        m: float(results[tags[1]][m] - results[tags[0]][m])
        for m in ["auc", "sensitivity", "specificity"]
    }
    for m in ["auc", "sensitivity", "specificity"]:
        lo, hi = ci_delta[m]
        sig = "" if (lo <= 0 <= hi) else "  (excludes 0)"
        print(f"    delta {m:<12} {point_delta[m]:+0.3f}  95% BCa CI [{lo:+.3f}, {hi:+.3f}]{sig}")

    # Save predictions.
    os.makedirs(HELDOUT_RESULTS_DIR, exist_ok=True)
    np.savez(
        HELDOUT_PRED_PATH,
        y_heldout=y_heldout.astype(int),
        real_only_probs=condition_probs[tags[0]],
        augmented_probs=condition_probs[tags[1]],
        threshold_real=np.array(condition_thr[tags[0]], dtype=float),
        threshold_augmented=np.array(condition_thr[tags[1]], dtype=float),
        selected_ratio=np.array(aug_factor, dtype=int),
    )

    def json_ci(ci_dict):
        return {k: [float(v[0]), float(v[1])] for k, v in ci_dict.items()}

    payload = {
        "n_heldout": int(len(y_heldout)),
        "heldout_class_counts": {
            "CN": int(np.sum(y_heldout == 0)),
            "impaired": int(np.sum(y_heldout == 1)),
        },
        "num_training_seeds": NUM_SEEDS,
        "num_participant_bootstraps": NUM_BOOTSTRAPS,
        "confidence_interval_method": "BCa",
        "selected_model": selected_model,
        "selected_ratio": int(aug_factor),
        "threshold_source": "repeated out-of-fold development predictions",
        "real_only": {
            "mean_auc": results[tags[0]]["auc"],
            "sd_auc_across_seeds": results[tags[0]]["sd_auc"],
            "mean_sensitivity": results[tags[0]]["sensitivity"],
            "sd_sensitivity_across_seeds": results[tags[0]]["sd_sensitivity"],
            "mean_specificity": results[tags[0]]["specificity"],
            "sd_specificity_across_seeds": results[tags[0]]["sd_specificity"],
            "threshold": results[tags[0]]["threshold"],
            "participant_bootstrap_95ci": json_ci(ci_real),
        },
        "augmented": {
            "mean_auc": results[tags[1]]["auc"],
            "sd_auc_across_seeds": results[tags[1]]["sd_auc"],
            "mean_sensitivity": results[tags[1]]["sensitivity"],
            "sd_sensitivity_across_seeds": results[tags[1]]["sd_sensitivity"],
            "mean_specificity": results[tags[1]]["specificity"],
            "sd_specificity_across_seeds": results[tags[1]]["sd_specificity"],
            "threshold": results[tags[1]]["threshold"],
            "participant_bootstrap_95ci": json_ci(ci_aug),
        },
        "paired_difference_augmented_minus_real": {
            "point_estimate": point_delta,
            "participant_bootstrap_95ci": json_ci(ci_delta),
        },
    }
    with open(HELDOUT_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    save_path = plot_confusion_matrices(results)
    print(f"\nSaved held-out predictions to {HELDOUT_PRED_PATH}")
    print(f"Saved held-out summary to {HELDOUT_RESULTS_PATH}")
    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    run()
