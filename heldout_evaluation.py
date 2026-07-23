"""
Evaluate the selected Late Fusion Categorical model on the 109-participant held-out set,
training on real data alone (0x) and with 7x synthetic augmentation. Each condition is
re-trained on all 433 participants once per seed, across 20 seeds. Scaling and the Youden's J
threshold both come from training data only. Report mean and SD across the 20 runs, and
plot the mean confusion matrices.
"""

import os
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Evaluation configuration
AUGMENTATION_FACTOR = 7
NUM_SEEDS = 20  # Independent retraining runs, all scored on the same held-out set
NUM_EPOCHS = 80  # Fixed Number
NUMERICAL_COLS = [0, 1]  # Only scale continuous variables (Age, Education)

FEATURES_DIR = "data/augmented_final"
SPLITS_DIR = "data/apoe1"
OUT_DIR = "figures"


def seed_everything(seed=42):
    # Ensure reproducible synthetic sampling and PyTorch weight initialization
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def fit_scaler(x_train):
    # Compute mean and standard deviation strictly on the training cohort to prevent data leakage
    mean = np.zeros(x_train.shape[1], dtype=np.float32)
    std = np.ones(x_train.shape[1], dtype=np.float32)
    mean[NUMERICAL_COLS] = x_train[:, NUMERICAL_COLS].mean(axis=0)
    std[NUMERICAL_COLS] = x_train[:, NUMERICAL_COLS].std(axis=0) + 1e-8
    return mean, std


def apply_scaler(x, mean, std):
    x_scaled = x.copy()
    x_scaled[:, NUMERICAL_COLS] = (x[:, NUMERICAL_COLS] - mean[NUMERICAL_COLS]) / std[NUMERICAL_COLS]
    return x_scaled


class LateFusionCategorical(nn.Module):
    """
    Late fusion architecture with categorical SNP encoding, the best-performing
    configuration in the cross-validated model comparison. Demographics and genetics
    are processed in separate branches before joining in a shared classification head.
    """
    def __init__(self, num_snps=112, embedding_dim=4):
        super().__init__()
        # Learnable embedding lookup for discrete SNP genotypes (0, 1, 2)
        self.shared_embedding = nn.Embedding(3, embedding_dim)
        genetics_dim = num_snps * embedding_dim

        self.clinical_branch = nn.Sequential(
            nn.Linear(3, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.genetics_branch = nn.Sequential(
            nn.Linear(genetics_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        demographics = x[:, :3]
        embedded = self.shared_embedding(x[:, 3:].long())
        genetics = embedded.view(embedded.size(0), -1)  # Flatten embedded matrix to 1D feature vector

        clinical_features = self.clinical_branch(demographics)
        genetics_features = self.genetics_branch(genetics)
        combined = torch.cat([clinical_features, genetics_features], dim=1)
        # squeeze(1) not squeeze(): a final batch of size 1 would collapse to a scalar
        return self.classification_head(combined).squeeze(1)


def train_and_predict(x_train, y_train, sample_weights, x_heldout, x_train_eval, seed):
    """
    Run one complete training pass and return probabilities on both the held-out set and the training
    cohort. The training predictions are used to pick the decision threshold.
    """
    seed_everything(seed)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
    x_heldout_tensor = torch.tensor(x_heldout, dtype=torch.float32)
    x_train_eval_tensor = torch.tensor(x_train_eval, dtype=torch.float32)

    dataset = TensorDataset(x_train_tensor, y_train_tensor, weights_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LateFusionCategorical()
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCELoss(reduction='none')

    for epoch in range(NUM_EPOCHS):
        model.train()
        for batch_x, batch_y, batch_weights in loader:
            optimizer.zero_grad()
            # Apply per-sample loss weights to balance positive and negative class gradients
            loss = (criterion(model(batch_x), batch_y) * batch_weights).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        heldout_predictions = model(x_heldout_tensor).numpy()
        train_predictions = model(x_train_eval_tensor).numpy()

    return heldout_predictions, train_predictions


def load_heldout():
    # The TabDDPM val and test partitions are recombined here into the 109-participant
    # held-out set. Neither was used to train or select the predictive models.
    feature_parts, label_parts = [], []
    for split in ["val", "test"]:
        x_numerical = np.load(os.path.join(SPLITS_DIR, f"X_num_{split}.npy")).astype(np.float32)
        x_categorical = np.load(os.path.join(SPLITS_DIR, f"X_cat_{split}.npy"),
                                allow_pickle=True).astype(np.float32)
        y = np.load(os.path.join(SPLITS_DIR, f"y_{split}.npy")).flatten()

        # Age, Education, Gender, then 112 SNPs = 115 columns
        feature_parts.append(np.hstack([x_numerical, x_categorical]).astype(np.float32))
        label_parts.append(y)

    x_heldout = np.vstack(feature_parts)
    y_heldout = np.concatenate(label_parts)
    y_heldout[y_heldout == 2] = 1  # Combine MCI and AD into a single impaired class
    return x_heldout, y_heldout


def compute_metrics(y_true, predictions, threshold):
    matrix = confusion_matrix(y_true, (predictions >= threshold).astype(int), labels=[0, 1])
    true_neg, false_pos, false_neg, true_pos = matrix.ravel()
    sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0
    specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) else 0
    auc = roc_auc_score(y_true, predictions) if len(np.unique(y_true)) > 1 else float('nan')
    return matrix, sensitivity, specificity, auc


def print_confusion_matrix(label, matrix, sensitivity, specificity, auc, threshold):
    true_neg, false_pos, false_neg, true_pos = matrix.ravel()
    print(f"\n{label}  (threshold = {threshold:.3f})")
    print("                 Pred CN   Pred Impaired")
    print(f"  True CN          {true_neg:>5}        {false_pos:>5}")
    print(f"  True Impaired    {false_neg:>5}        {true_pos:>5}")
    print(f"  Sensitivity (impaired recall): {sensitivity:.3f}")
    print(f"  Specificity (CN recall):       {specificity:.3f}")
    print(f"  AUC:                           {auc:.3f}")


def youden_threshold(y_true, predictions):
    # Youden's J: the threshold maximizing sensitivity + specificity - 1
    false_pos_rate, true_pos_rate, thresholds = roc_curve(y_true, predictions)
    return thresholds[np.argmax(true_pos_rate - false_pos_rate)]


def plot_confusion_matrices(results):
    """
    Two-panel figure of the mean confusion matrices, 0x on the left and 7x on the right.
    Cells are row-normalized proportions averaged across seeds, so the diagonal entries
    are the mean specificity (top left) and mean sensitivity (bottom right).
    """
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.weight': 'bold',
        'axes.labelweight': 'bold'
    })

    labels = ["CN", "Impaired"]
    panels = [
        ("0x  Real Data Only", "0x (real only)", "Blues"),
        ("7x  Synthetic Augmented", "7x (augmented)", "Oranges"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (panel_title, tag, colormap) in zip(axes, panels):
        proportions = results[tag]["proportions"]
        sensitivity = results[tag]["sensitivity"]
        specificity = results[tag]["specificity"]
        auc = results[tag]["auc"]

        image = ax.imshow(proportions, cmap=colormap, vmin=0, vmax=1)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

        # White text on dark cells, dark text on light ones
        for row in range(2):
            for col in range(2):
                value = proportions[row, col]
                ax.text(col, row, f"{value:.2f}", ha='center', va='center',
                        fontsize=34, fontweight='bold',
                        color='white' if value > 0.5 else '#333333')

        ax.set_title(f"{panel_title}\nAUC: {auc:.3f}\n"
                     f"Sensitivity: {sensitivity:.2f}   Specificity: {specificity:.2f}",
                     fontsize=20, fontweight='bold', pad=18)
        ax.set_xticks([0, 1], labels, fontsize=15, fontweight='bold')
        ax.set_yticks([0, 1], labels, fontsize=15, fontweight='bold')
        ax.set_xlabel("Predicted", fontsize=17, fontweight='bold')
        ax.set_ylabel("Actual", fontsize=17, fontweight='bold')

    fig.suptitle("Late Fusion (Categorical): Real Data vs Synthetic Augmentation",
                 fontsize=26, fontweight='bold', y=1.02)
    fig.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    save_path = os.path.join(OUT_DIR, "heldout_confusion_matrices.png")
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return save_path


def run():
    X_real = np.load(os.path.join(FEATURES_DIR, "X_real_features_115.npy")).astype(np.float32)
    y_real = np.load(os.path.join(FEATURES_DIR, "y_real.npy")).flatten()
    y_real[y_real == 2] = 1  # Combine diagnostic classes into binary outcome (0: Normal, 1: Impaired)

    X_synth = np.load(os.path.join(FEATURES_DIR, "X_synth_features_115.npy")).astype(np.float32)
    y_synth = np.load(os.path.join(FEATURES_DIR, "y_synth.npy")).flatten()
    y_synth[y_synth == 2] = 1

    x_heldout, y_heldout = load_heldout()

    print("Late Fusion Categorical: held-out confusion matrices (0x vs 7x)")
    print(f"Training cohort: {X_real.shape}  class balance {np.bincount(y_real.astype(int))}")
    print(f"Held-out set:    {x_heldout.shape}  class balance {np.bincount(y_heldout.astype(int))}")
    print(f"Retrained across {NUM_SEEDS} seeds; threshold set on training predictions.\n")

    # Continuous feature standardization on the real training cohort only
    mean, std = fit_scaler(X_real)
    x_train_scaled = apply_scaler(X_real, mean, std)
    x_heldout_scaled = apply_scaler(x_heldout, mean, std)

    # Inverse frequency weighting to handle target class imbalance
    class_weights_real = 1.0 / np.bincount(y_real.astype(int))
    weights_baseline = class_weights_real[y_real.astype(int)]
    weights_baseline = weights_baseline / (weights_baseline.mean() + 1e-8)

    # Draw the synthetic sample once with a fixed seed so both conditions are comparable
    seed_everything(42)
    synth_indices = np.random.permutation(len(X_synth))[:len(X_real) * AUGMENTATION_FACTOR]
    x_augmented = np.concatenate([x_train_scaled, apply_scaler(X_synth[synth_indices], mean, std)])
    y_augmented = np.concatenate([y_real, y_synth[synth_indices]])

    # Weights come from class_weights_real (real cohort frequencies), not the augmented mix,
    # so the loss weighting is the same with and without augmentation
    weights_augmented = class_weights_real[y_augmented.astype(int)]
    weights_augmented = weights_augmented / (weights_augmented.mean() + 1e-8)

    conditions = {
        "0x (real only)": (x_train_scaled, y_real, weights_baseline),
        "7x (augmented)": (x_augmented, y_augmented, weights_augmented),
    }

    results = {}

    for tag, (x_train, y_train, sample_weights) in conditions.items():
        print("\n" + "=" * 70)
        print(f"{tag}  --  per-seed runs")
        print("=" * 70)

        seed_sensitivity, seed_specificity, seed_auc, seed_matrices = [], [], [], []

        for seed in range(NUM_SEEDS):
            heldout_predictions, train_predictions = train_and_predict(
                x_train, y_train, sample_weights, x_heldout_scaled, x_train_scaled, seed
            )

            # Each seed derives its own threshold from its own training predictions
            threshold = youden_threshold(y_real, train_predictions)
            matrix, sensitivity, specificity, auc = compute_metrics(
                y_heldout, heldout_predictions, threshold
            )
            seed_sensitivity.append(sensitivity)
            seed_specificity.append(specificity)
            seed_auc.append(auc)
            seed_matrices.append(matrix)
            print_confusion_matrix(f"{tag}  [seed {seed}]", matrix, sensitivity,
                                   specificity, auc, threshold)

        mean_sensitivity = float(np.mean(seed_sensitivity))
        mean_specificity = float(np.mean(seed_specificity))
        mean_auc = float(np.mean(seed_auc))

        print("\n" + "-" * 70)
        print(f"SUMMARY across {NUM_SEEDS} seeds for {tag}:")
        print(f"  Sensitivity: {mean_sensitivity:.3f} +/- {np.std(seed_sensitivity):.3f}")
        print(f"  Specificity: {mean_specificity:.3f} +/- {np.std(seed_specificity):.3f}")
        print(f"  AUC:         {mean_auc:.3f} +/- {np.std(seed_auc):.3f}")
        print("-" * 70)

        # Mean confusion matrix across seeds
        # The diagonal is the mean specificity and sensitivity above.
        mean_matrix = np.mean(seed_matrices, axis=0)
        proportions = mean_matrix / mean_matrix.sum(axis=1, keepdims=True)
        print(f"Mean counts   True CN: {mean_matrix[0, 0]:.1f} / {mean_matrix[0, 1]:.1f}"
              f"   True Impaired: {mean_matrix[1, 0]:.1f} / {mean_matrix[1, 1]:.1f}")

        results[tag] = {
            "proportions": proportions,
            "sensitivity": mean_sensitivity,
            "specificity": mean_specificity,
            "auc": mean_auc,
        }

    save_path = plot_confusion_matrices(results)
    print(f"\nSaved figure to {save_path}")


if __name__ == "__main__":
    run()
