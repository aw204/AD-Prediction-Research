"""
Compare seven model and encoding configurations on the 433-participant training
cohort, using 5-fold stratified cross-validation repeated 10 times (50 folds), at
synthetic-to-real augmentation ratios of 0x, 3x, 5x, 7x, and 10x.

Synthetic samples are added to the training folds only. Neural networks train for a
fixed 80 epochs in all conditions, and each held-out fold is scored once at the end.
"""

import os
import random
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


# Benchmark configuration
AUGMENTATION_FACTORS = [3, 5, 7, 10]
NUMERICAL_COLS = [0, 1]  # Only scale continuous variables (Age, Education)
NUM_SPLITS = 5
NUM_REPEATS = 10  # 5 splits x 10 repeats = 50 total cross-validation folds
NUM_EPOCHS = 80  # Fixed; no epoch selection, matches heldout_evaluation.py


def seed_everything(seed=42):
    # Ensure reproducible fold splitting, synthetic sampling, and PyTorch weight initialization
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def fit_scaler(x_train):
    # Compute mean and standard deviation strictly on the training fold to prevent data leakage
    mean = np.zeros(x_train.shape[1], dtype=np.float32)
    std = np.ones(x_train.shape[1], dtype=np.float32)
    mean[NUMERICAL_COLS] = x_train[:, NUMERICAL_COLS].mean(axis=0)
    std[NUMERICAL_COLS] = x_train[:, NUMERICAL_COLS].std(axis=0) + 1e-8
    return mean, std


def apply_scaler(x, mean, std):
    x_scaled = x.copy()
    x_scaled[:, NUMERICAL_COLS] = (x[:, NUMERICAL_COLS] - mean[NUMERICAL_COLS]) / std[NUMERICAL_COLS]
    return x_scaled


class FusionNet(nn.Module):
    """
    Flexible PyTorch architecture supporting both Early and Late fusion strategies,
    as well as numeric (raw 0,1,2 values) vs categorical (learned 4D embeddings) SNP encodings.
    """
    def __init__(self, fusion_type, encoding_type, num_snps=112, embedding_dim=4):
        super().__init__()
        self.fusion_type = fusion_type
        self.encoding_type = encoding_type
        self.num_snps = num_snps
        self.embedding_dim = embedding_dim

        genetics_dim = num_snps * embedding_dim if encoding_type == "CATEGORICAL" else num_snps
        
        # Learnable embedding lookup for discrete SNP genotypes (0, 1, 2)
        if encoding_type == "CATEGORICAL":
            self.shared_embedding = nn.Embedding(3, embedding_dim)

        if fusion_type == "EARLY":
            # Early Fusion: Concatenate clinical demographics and genetic features at input
            input_dim = 3 + genetics_dim
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
        else:
            # Late Fusion: Separate processing branches before joining in a joint representation head
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

    def _embed_genetics(self, x_genetics):
        if self.encoding_type == "CATEGORICAL":
            embedded = self.shared_embedding(x_genetics.long())
            return embedded.view(embedded.size(0), -1)  # Flatten embedded matrix to 1D feature vector
        return x_genetics

    def forward(self, x):
        demographics = x[:, :3]
        genetics = self._embed_genetics(x[:, 3:])

        if self.fusion_type == "EARLY":
            combined = torch.cat([demographics, genetics], dim=1)
            # squeeze(1) not squeeze(): a final batch of size 1 would collapse to a scalar
            return self.network(combined).squeeze(1)

        clinical_features = self.clinical_branch(demographics)
        genetics_features = self.genetics_branch(genetics)
        combined = torch.cat([clinical_features, genetics_features], dim=1)
        return self.classification_head(combined).squeeze(1)


def train_nn(fusion_type, encoding_type, x_train, y_train, sample_weights, x_test, y_test):
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)

    dataset = TensorDataset(x_train_tensor, y_train_tensor, weights_tensor)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = FusionNet(fusion_type, encoding_type)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCELoss(reduction='none')

    # Train a fixed number of epochs. The evaluation fold is scored once at the end,
    # so it never influences training, matching how the sklearn models are handled.
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
        predictions = model(x_test_tensor).numpy()
    final_auc = roc_auc_score(y_test, predictions) if len(np.unique(y_test)) > 1 else 0

    return final_auc, predictions


def train_sklearn(model_name, encoding_type, x_train, y_train, sample_weights, x_test, y_test):
    if model_name == "RIDGE":
        classifier = LogisticRegression(penalty='l2', C=1.0, max_iter=2000, solver='liblinear')
        classifier.fit(x_train, y_train, sample_weight=sample_weights)
        predictions = classifier.predict_proba(x_test)[:, 1]

    elif model_name == "XGB":
        import pandas as pd
        from xgboost import XGBClassifier

        is_categorical = (encoding_type == "CATEGORICAL")
        df_train = pd.DataFrame(x_train)
        df_test = pd.DataFrame(x_test)

        # Cast SNP features to pandas category type for native XGBoost categorical tree splits
        if is_categorical:
            for col in range(3, 115):
                df_train[col] = df_train[col].astype(int).astype('category')
                df_test[col] = df_test[col].astype(int).astype('category')

        classifier = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            enable_categorical=is_categorical, tree_method='hist',
            verbosity=0, n_jobs=2, random_state=42
        )
        classifier.fit(df_train, y_train, sample_weight=sample_weights)
        predictions = classifier.predict_proba(df_test)[:, 1]

    elif model_name == "CAT":
        import pandas as pd
        from catboost import CatBoostClassifier

        categorical_features = list(range(3, 115)) if encoding_type == "CATEGORICAL" else None
        df_train = pd.DataFrame(x_train)
        df_test = pd.DataFrame(x_test)

        if categorical_features:
            for col in categorical_features:
                df_train[col] = df_train[col].astype(int)
                df_test[col] = df_test[col].astype(int)

        classifier = CatBoostClassifier(
            iterations=300, depth=4, learning_rate=0.05,
            verbose=0, allow_writing_files=False, thread_count=2,
            random_seed=42
        )
        classifier.fit(df_train, y_train, sample_weight=sample_weights, cat_features=categorical_features)
        predictions = classifier.predict_proba(df_test)[:, 1]

    auc = roc_auc_score(y_test, predictions) if len(np.unique(y_test)) > 1 else 0
    return auc, predictions


def run():
    seed_everything(42)
    data_dir = "data/augmented_final"

    X_real = np.load(os.path.join(data_dir, "X_real_features_115.npy")).astype(np.float32)
    y_real = np.load(os.path.join(data_dir, "y_real.npy")).flatten()
    y_real[y_real == 2] = 1  # Combine diagnostic classes into binary outcome (0: Normal, 1: Impaired)

    X_synth = np.load(os.path.join(data_dir, "X_synth_features_115.npy")).astype(np.float32)
    y_synth = np.load(os.path.join(data_dir, "y_synth.npy")).flatten()
    y_synth[y_synth == 2] = 1

    print(f"Loaded real dataset: {X_real.shape} | Synthetic: {X_synth.shape}")
    print(f"Class balance (real): {np.bincount(y_real.astype(int))}\n")

    # Map model display names to (framework, algorithm/architecture, feature_encoding)
    models = {
        "Ridge LR (Numeric)":        ("SKLEARN", "RIDGE", "NUMERIC"),
        "Early Fusion (Numeric)":    ("Neural Network", "EARLY", "NUMERIC"),
        "Early Fusion (Categorical)":("Neural Network", "EARLY", "CATEGORICAL"),
        "Late Fusion (Numeric)":     ("Neural Network", "LATE",  "NUMERIC"),
        "Late Fusion (Categorical)": ("Neural Network", "LATE",  "CATEGORICAL"),
        "XGBoost (Numeric)":         ("SKLEARN", "XGB",   "NUMERIC"),
        "CatBoost (Categorical)":    ("SKLEARN", "CAT",   "CATEGORICAL")
    }

    max_augmentation = max(AUGMENTATION_FACTORS)
    results = {}
    start_time = time.time()
    all_levels = [0] + AUGMENTATION_FACTORS

    for name, (model_family, model_or_fusion, encoding_type) in models.items():
        print(f"\n--- {name} ---")
        print(f"{'Fold':<8}{'0x':>8}{'3x':>8}{'5x':>8}{'7x':>8}{'10x':>8}")

        auc_lists = {k: [] for k in all_levels}

        # Reset global seed prior to each model run to guarantee identical CV fold splits and synthetic sampling
        seed_everything(42)
        cross_validator = RepeatedStratifiedKFold(n_splits=NUM_SPLITS, n_repeats=NUM_REPEATS, random_state=42)
        num_folds = cross_validator.get_n_splits()

        for fold_idx, (train_idx, test_idx) in enumerate(cross_validator.split(X_real, y_real)):
            x_train_real, y_train_real = X_real[train_idx].copy(), y_real[train_idx]
            x_test_real, y_test_real = X_real[test_idx].copy(), y_real[test_idx]

            # Fit continuous feature standardization on the real training fold only
            mean, std = fit_scaler(x_train_real)
            x_train_scaled = apply_scaler(x_train_real, mean, std)
            x_test_scaled = apply_scaler(x_test_real, mean, std)

            # Inverse frequency weighting to handle target class imbalance
            class_weights_real = 1.0 / np.bincount(y_train_real.astype(int))
            weights_baseline = class_weights_real[y_train_real.astype(int)]
            weights_baseline = weights_baseline / (weights_baseline.mean() + 1e-8)

            # Evaluate 0x baseline (real data only)
            if model_family == "Neural Network":
                auc_baseline, _ = train_nn(model_or_fusion, encoding_type, x_train_scaled, y_train_real, weights_baseline, x_test_scaled, y_test_real)
            else:
                auc_baseline, _ = train_sklearn(model_or_fusion, encoding_type, x_train_scaled, y_train_real, weights_baseline, x_test_scaled, y_test_real)
            
            auc_lists[0].append(auc_baseline)

            fold_row = f"{fold_idx + 1:02d}/{num_folds:<4}{auc_baseline:>8.3f}"

            # Randomly select maximum synthetic samples once per fold and slice incrementally for 3x, 5x, 7x, 10x
            synth_indices_max = np.random.permutation(len(X_synth))[:len(x_train_real) * max_augmentation]
            for multiplier in AUGMENTATION_FACTORS:
                synth_indices = synth_indices_max[:len(x_train_real) * multiplier]
                x_synth_scaled = apply_scaler(X_synth[synth_indices], mean, std)
                x_augmented = np.concatenate([x_train_scaled, x_synth_scaled])
                y_augmented = np.concatenate([y_train_real, y_synth[synth_indices]])

                # Weights come from class_weights_real (real fold frequencies), not the augmented mix,
                # so the loss weighting stays fixed across augmentation ratios
                weights_augmented = class_weights_real[y_augmented.astype(int)]
                weights_augmented = weights_augmented / (weights_augmented.mean() + 1e-8)

                if model_family == "Neural Network":
                    auc_augmented, _ = train_nn(model_or_fusion, encoding_type, x_augmented, y_augmented, weights_augmented, x_test_scaled, y_test_real)
                else:
                    auc_augmented, _ = train_sklearn(model_or_fusion, encoding_type, x_augmented, y_augmented, weights_augmented, x_test_scaled, y_test_real)

                auc_lists[multiplier].append(auc_augmented)
                fold_row += f"{auc_augmented:>8.3f}"

            print(fold_row)

        results[name] = {k: np.array(auc_lists[k]) for k in all_levels}
        mean_row = "MEAN   " + "".join([f"{results[name][k].mean():>8.3f}" for k in all_levels])
        print(mean_row)
        print(f"Elapsed time: {time.time() - start_time:.0f}s")

    # Final summary table
    print("\n=== Model Comparison Summary ===")
    print(f"{'Model':<28}{'0x AUC':>10}{'3x AUC':>10}{'5x AUC':>10}{'7x AUC':>10}{'10x AUC':>10}{'Max Uplift':>12}")
    
    summary = []
    for name in models:
        mean_0x  = results[name][0].mean()
        mean_3x  = results[name][3].mean()
        mean_5x  = results[name][5].mean()
        mean_7x  = results[name][7].mean()
        mean_10x = results[name][10].mean()

        best_augmented = max(mean_3x, mean_5x, mean_7x, mean_10x)
        uplift = best_augmented - mean_0x
        summary.append((name, mean_0x, mean_3x, mean_5x, mean_7x, mean_10x, uplift))
        print(f"{name:<28}{mean_0x:>10.3f}{mean_3x:>10.3f}{mean_5x:>10.3f}{mean_7x:>10.3f}{mean_10x:>10.3f}{uplift:>+12.3f}")

    best_overall = max(
        [(name, k, results[name][k].mean()) for name in models for k in all_levels],
        key=lambda r: r[2]
    )
    best_uplift = max(summary, key=lambda r: r[6])
    
    print(f"\nHighest Overall AUC: {best_overall[0]} @ {best_overall[1]}x ({best_overall[2]:.3f})")
    print(f"Largest Uplift:      {best_uplift[0]} ({best_uplift[6]:+.3f})")


if __name__ == "__main__":
    run()
