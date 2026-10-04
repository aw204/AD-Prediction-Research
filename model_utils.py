"""Utilities shared by the model training and evaluation scripts."""

import os
import random
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

NUM_SNPS = 168
NUM_FEATURES = 171  # Age, Education, Gender, 168 SNPs
NUMERICAL_COLS = [0, 1]
NUM_EPOCHS = 80
BATCH_SIZE = 64


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scaler(x_train: np.ndarray):
    """Fit scaling parameters for age and education only."""
    mean = np.zeros(x_train.shape[1], dtype=np.float32)
    std = np.ones(x_train.shape[1], dtype=np.float32)
    mean[NUMERICAL_COLS] = x_train[:, NUMERICAL_COLS].mean(axis=0)
    std[NUMERICAL_COLS] = x_train[:, NUMERICAL_COLS].std(axis=0) + 1e-8
    return mean, std


def apply_scaler(x: np.ndarray, mean: np.ndarray, std: np.ndarray):
    x_scaled = x.copy().astype(np.float32, copy=False)
    x_scaled[:, NUMERICAL_COLS] = (
        x_scaled[:, NUMERICAL_COLS] - mean[NUMERICAL_COLS]
    ) / std[NUMERICAL_COLS]
    return x_scaled


def draw_synthetic_indices(
    y_synth: np.ndarray,
    n_samples: int,
    seed: int = 42,
) -> np.ndarray:
    # Draw class-proportional synthetic samples without replacement.
    labels = np.asarray(y_synth).astype(int).ravel()
    if n_samples < 0:
        raise ValueError("n_samples must be non-negative")
    if n_samples > len(labels):
        raise ValueError(
            f"Requested {n_samples} synthetic records, but only {len(labels)} are available."
        )

    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) == 0:
        raise ValueError("Synthetic labels are empty")

    # Match class proportions using largest-remainder allocation.
    expected = n_samples * counts.astype(float) / counts.sum()
    quotas = np.floor(expected).astype(int)
    remainder = int(n_samples - quotas.sum())
    order = np.argsort(-(expected - quotas))
    quotas[order[:remainder]] += 1

    if np.any(quotas > counts):
        raise ValueError(
            "The synthetic pool cannot satisfy the requested class-proportional draw "
            "without replacement."
        )

    rng = np.random.default_rng(seed)
    selected_parts = []
    for cls, quota in zip(classes, quotas):
        class_indices = np.flatnonzero(labels == cls)
        if quota:
            selected_parts.append(rng.choice(class_indices, size=int(quota), replace=False))

    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    rng.shuffle(selected)
    if len(selected) != n_samples:
        raise RuntimeError(
            f"Synthetic draw returned {len(selected)} records; expected {n_samples}."
        )
    return selected


def auc_or_nan(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, probs))


class FusionNet(nn.Module):
    """Early/late fusion with numeric or categorical SNP encoding."""

    def __init__(
        self,
        fusion_type: str,
        encoding_type: str,
        num_snps: int = NUM_SNPS,
        embedding_dim: int = 4,
    ):
        super().__init__()
        self.fusion_type = fusion_type
        self.encoding_type = encoding_type
        self.num_snps = num_snps
        self.embedding_dim = embedding_dim

        genetics_dim = num_snps * embedding_dim if encoding_type == "CATEGORICAL" else num_snps

        if encoding_type == "CATEGORICAL":
            self.shared_embedding = nn.Embedding(3, embedding_dim)

        if fusion_type == "EARLY":
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
                nn.Sigmoid(),
            )
        elif fusion_type == "LATE":
            self.clinical_branch = nn.Sequential(
                nn.Linear(3, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.genetics_branch = nn.Sequential(
                nn.Linear(genetics_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
            )
            self.classification_head = nn.Sequential(
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")

    def _embed_genetics(self, x_genetics: torch.Tensor):
        if self.encoding_type == "CATEGORICAL":
            embedded = self.shared_embedding(x_genetics.long())
            return embedded.reshape(embedded.size(0), -1)
        if self.encoding_type == "NUMERIC":
            return x_genetics
        raise ValueError(f"Unknown encoding_type: {self.encoding_type}")

    def forward(self, x: torch.Tensor):
        demographics = x[:, :3]
        genetics = self._embed_genetics(x[:, 3:])

        if self.fusion_type == "EARLY":
            combined = torch.cat([demographics, genetics], dim=1)
            return self.network(combined).squeeze(1)

        clinical_features = self.clinical_branch(demographics)
        genetics_features = self.genetics_branch(genetics)
        combined = torch.cat([clinical_features, genetics_features], dim=1)
        return self.classification_head(combined).squeeze(1)


def _fit_nn(
    fusion_type: str,
    encoding_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    seed: int,
    embedding_dim: int = 4,
):
    seed_everything(seed)

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    weights_tensor = torch.tensor(sample_weights, dtype=torch.float32)

    dataset = TensorDataset(x_train_tensor, y_train_tensor, weights_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = FusionNet(
        fusion_type,
        encoding_type,
        num_snps=NUM_SNPS,
        embedding_dim=embedding_dim,
    )
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    criterion = nn.BCELoss(reduction="none")

    for _ in range(NUM_EPOCHS):
        model.train()
        for batch_x, batch_y, batch_weights in loader:
            optimizer.zero_grad()
            loss = (criterion(model(batch_x), batch_y) * batch_weights).mean()
            loss.backward()
            optimizer.step()

    return model


def _predict_nn(model: nn.Module, x: np.ndarray):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(x, dtype=torch.float32)).cpu().numpy()


def train_nn_predict(
    fusion_type: str,
    encoding_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    x_eval: np.ndarray,
    seed: int = 42,
    embedding_dim: int = 4,
):
    model = _fit_nn(
        fusion_type,
        encoding_type,
        x_train,
        y_train,
        sample_weights,
        seed=seed,
        embedding_dim=embedding_dim,
    )
    return _predict_nn(model, x_eval)


def _fit_sklearn(
    model_name: str,
    encoding_type: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    seed: int,
):
    if model_name == "RIDGE":
        classifier = LogisticRegression(
            penalty="l2", C=1.0, max_iter=2000, solver="liblinear", random_state=seed
        )
        classifier.fit(x_train, y_train, sample_weight=sample_weights)
        return classifier

    if model_name == "NO_PENALTY":
        classifier = LogisticRegression(
            penalty=None, max_iter=2000, solver="lbfgs", random_state=seed
        )
        classifier.fit(x_train, y_train, sample_weight=sample_weights)
        return classifier

    if model_name == "XGB":
        import pandas as pd
        from xgboost import XGBClassifier

        is_categorical = encoding_type == "CATEGORICAL"
        df_train = pd.DataFrame(x_train)
        if is_categorical:
            for col in range(3, NUM_FEATURES):
                df_train[col] = df_train[col].astype(int).astype("category")

        classifier = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            enable_categorical=is_categorical,
            tree_method="hist",
            verbosity=0,
            n_jobs=2,
            random_state=seed,
        )
        classifier.fit(df_train, y_train, sample_weight=sample_weights)
        return classifier

    if model_name == "CAT":
        import pandas as pd
        from catboost import CatBoostClassifier

        categorical_features = list(range(3, NUM_FEATURES)) if encoding_type == "CATEGORICAL" else None
        df_train = pd.DataFrame(x_train)
        if categorical_features:
            for col in categorical_features:
                df_train[col] = df_train[col].astype(int)

        classifier = CatBoostClassifier(
            iterations=300,
            depth=4,
            learning_rate=0.05,
            verbose=0,
            allow_writing_files=False,
            thread_count=2,
            random_seed=seed,
        )
        classifier.fit(
            df_train,
            y_train,
            sample_weight=sample_weights,
            cat_features=categorical_features,
        )
        return classifier

    raise ValueError(f"Unknown sklearn model: {model_name}")


def _predict_sklearn(classifier: Any, model_name: str, encoding_type: str, x: np.ndarray):
    if model_name in {"XGB", "CAT"}:
        import pandas as pd

        df = pd.DataFrame(x)
        if encoding_type == "CATEGORICAL":
            for col in range(3, NUM_FEATURES):
                if model_name == "XGB":
                    df[col] = df[col].astype(int).astype("category")
                else:
                    df[col] = df[col].astype(int)
        return classifier.predict_proba(df)[:, 1]

    return classifier.predict_proba(x)[:, 1]


def model_specifications():
    """Return the model configurations used in the analysis."""
    return {
        "Ridge LR (Numeric)": ("SKLEARN", "RIDGE", "NUMERIC"),
        "Logistic Regression (No Penalty)": ("SKLEARN", "NO_PENALTY", "NUMERIC"),
        "Early Fusion (Numeric)": ("NN", "EARLY", "NUMERIC"),
        "Early Fusion (Categorical)": ("NN", "EARLY", "CATEGORICAL"),
        "Late Fusion (Numeric)": ("NN", "LATE", "NUMERIC"),
        "Late Fusion (Categorical)": ("NN", "LATE", "CATEGORICAL"),
        "XGBoost (Numeric)": ("SKLEARN", "XGB", "NUMERIC"),
        "CatBoost (Categorical)": ("SKLEARN", "CAT", "CATEGORICAL"),
    }


def predict_for_spec(
    spec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    x_eval: np.ndarray,
    seed: int = 42,
):
    family, model_or_fusion, encoding_type = spec
    if family == "NN":
        return train_nn_predict(
            model_or_fusion,
            encoding_type,
            x_train,
            y_train,
            sample_weights,
            x_eval,
            seed=seed,
        )

    classifier = _fit_sklearn(
        model_or_fusion,
        encoding_type,
        x_train,
        y_train,
        sample_weights,
        seed=seed,
    )
    return _predict_sklearn(classifier, model_or_fusion, encoding_type, x_eval)


def predict_two_for_spec(
    spec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    x_eval_a: np.ndarray,
    x_eval_b: np.ndarray,
    seed: int = 42,
):
    """Fit once and return probabilities for two evaluation matrices."""
    family, model_or_fusion, encoding_type = spec
    if family == "NN":
        model = _fit_nn(
            model_or_fusion,
            encoding_type,
            x_train,
            y_train,
            sample_weights,
            seed=seed,
        )
        return _predict_nn(model, x_eval_a), _predict_nn(model, x_eval_b)

    classifier = _fit_sklearn(
        model_or_fusion,
        encoding_type,
        x_train,
        y_train,
        sample_weights,
        seed=seed,
    )
    return (
        _predict_sklearn(classifier, model_or_fusion, encoding_type, x_eval_a),
        _predict_sklearn(classifier, model_or_fusion, encoding_type, x_eval_b),
    )
