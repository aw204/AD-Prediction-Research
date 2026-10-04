"""Prepare APOE data and create the training and evaluation splits.
Feature order:
[Age, Education, Gender, SNP_0, ..., SNP_167].
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

N_SNPS = 168
DEV_SIZE = 433
HELDOUT_SIZE = 109
N_SPLITS = 5
N_REPEATS = 10
DEFAULT_SEED = 42


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None)


def load_source_split(raw_dir: Path, prefix: str) -> pd.DataFrame:
    geno = _read_table(raw_dir / f"apoe_{prefix}_data.txt")
    if geno.shape[1] != N_SNPS:
        raise ValueError(f"{prefix}: expected {N_SNPS} SNP columns, found {geno.shape[1]}")
    geno.columns = [f"SNP_{i}" for i in range(N_SNPS)]

    dx = _read_table(raw_dir / f"{prefix}_dx.txt")
    age = _read_table(raw_dir / f"{prefix}_age.txt")
    gender = _read_table(raw_dir / f"{prefix}_gender.txt")
    educ = _read_table(raw_dir / f"{prefix}_education.txt")

    n = len(geno)
    for name, frame in [("dx", dx), ("age", age), ("gender", gender), ("education", educ)]:
        if len(frame) != n:
            raise ValueError(f"{prefix}: genotype rows={n}, but {name} rows={len(frame)}")
        if frame.shape[1] < 2:
            raise ValueError(f"{prefix}: {name} file must contain subject ID and value columns")

    # Check that subject IDs match across files.
    ids = [f.iloc[:, 0].astype(str).str.strip('"') for f in [dx, age, gender, educ]]
    if not all(ids[0].reset_index(drop=True).equals(x.reset_index(drop=True)) for x in ids[1:]):
        raise ValueError(f"{prefix}: subject IDs are not aligned across dx/age/gender/education files")

    clinical = pd.DataFrame({
        "SubjectID": ids[0].values,
        "Diagnosis": pd.to_numeric(dx.iloc[:, 1], errors="coerce"),
        "Age": pd.to_numeric(age.iloc[:, 1], errors="coerce"),
        "Gender": pd.to_numeric(gender.iloc[:, 1], errors="coerce"),
        "Education": pd.to_numeric(educ.iloc[:, 1], errors="coerce"),
    })
    return pd.concat([clinical, geno.reset_index(drop=True)], axis=1)


def validate_analysis_dataframe(df: pd.DataFrame) -> list[str]:
    snp_cols = [f"SNP_{i}" for i in range(N_SNPS)]

    if df[["Age", "Gender", "Education"]].isna().any().any():
        raise ValueError("Age/Gender/Education contain missing values")

    if df[snp_cols].isna().any().any():
        raise ValueError("SNP data contain missing values")

    snp_values = np.unique(df[snp_cols].to_numpy())
    if not set(snp_values) <= {0, 1, 2}:
        raise ValueError("Unexpected SNP genotype values")

    dx_values = sorted(df["Diagnosis"].unique().tolist())
    if dx_values != [0, 1, 2]:
        raise ValueError("Unexpected diagnosis classes")

    gender_values = sorted(df["Gender"].unique().tolist())
    if not set(gender_values).issubset({0, 1}):
        raise ValueError("Unexpected gender values")

    return snp_cols


def feature_matrix(df: pd.DataFrame, snp_cols: list[str]) -> np.ndarray:
    return np.hstack([
        df[["Age", "Education", "Gender"]].to_numpy(dtype=np.float32),
        df[snp_cols].to_numpy(dtype=np.float32),
    ]).astype(np.float32)


def save_predictive_cohort(out_dir: Path, df: pd.DataFrame, snp_cols: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_num = df[["Age", "Education"]].to_numpy(dtype=np.float32)
    x_cat = np.hstack([
        df["Gender"].astype(int).astype(str).to_numpy().reshape(-1, 1),
        df[snp_cols].astype(int).astype(str).to_numpy(),
    ])
    y = df["Diagnosis"].to_numpy(dtype=np.int64)
    X = feature_matrix(df, snp_cols)

    np.save(out_dir / "X_num.npy", x_num)
    np.save(out_dir / "X_cat.npy", x_cat, allow_pickle=True)
    np.save(out_dir / "y.npy", y)
    np.save(out_dir / "X_real_features_171.npy", X)
    np.save(out_dir / "y_real.npy", y)
    np.save(out_dir / "participant_id.npy", df["SubjectID"].astype(str).to_numpy(), allow_pickle=True)


def write_info(out_dir: Path, n_snps: int) -> None:
    info = {
        "name": "apoe1",
        "id": "apoe1",
        "task_type": "multiclass",
        "n_classes": 3,
        "num_col_indices": [0, 1], # age and education
        "cat_col_indices": list(range(1 + n_snps)),  # gender + 168 SNPs
        "target_col_indices": [0],
        "label_decode": {"0": "CN", "1": "MCI", "2": "AD"},
    }
    with open(out_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)


def save_tabddpm_partition(out_dir: Path, split_name: str, df: pd.DataFrame, snp_cols: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    x_num = df[["Age", "Education"]].to_numpy(dtype=np.float32)
    x_cat = np.hstack([
        df["Gender"].astype(int).astype(str).to_numpy().reshape(-1, 1),
        df[snp_cols].astype(int).astype(str).to_numpy(),
    ])
    y = df["Diagnosis"].to_numpy(dtype=np.int64)

    np.save(out_dir / f"X_num_{split_name}.npy", x_num)
    np.save(out_dir / f"X_cat_{split_name}.npy", x_cat, allow_pickle=True)
    np.save(out_dir / f"y_{split_name}.npy", y)
    np.save(out_dir / f"Y_{split_name}.npy", y)
    np.save(out_dir / f"participant_id_{split_name}.npy",
            df["SubjectID"].astype(str).to_numpy(), allow_pickle=True)


def internal_tabddpm_split(part_df: pd.DataFrame, out_dir: Path, snp_cols: list[str], seed: int) -> dict:
    # Create a stratified 80/10/10 split using only part_df.
    positions = np.arange(len(part_df))
    tr_pos, tmp_pos = train_test_split(
        positions,
        test_size=0.20,
        random_state=seed,
        stratify=part_df["Diagnosis"].to_numpy(),
    )
    val_pos, test_pos = train_test_split(
        tmp_pos,
        test_size=0.50,
        random_state=seed,
        stratify=part_df.iloc[tmp_pos]["Diagnosis"].to_numpy(),
    )

    split_pos = {"train": tr_pos, "val": val_pos, "test": test_pos}
    for name, pos in split_pos.items():
        save_tabddpm_partition(out_dir, name, part_df.iloc[pos], snp_cols)
    write_info(out_dir, len(snp_cols))

    return {
        "train": int(len(tr_pos)),
        "val": int(len(val_pos)),
        "test": int(len(test_pos)),
    }


def counts3(y: np.ndarray) -> list[int]:
    return np.bincount(np.asarray(y, dtype=int), minlength=3).tolist()


def prepare_data(raw_dir: Path, out_dir: Path, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.concat([
        load_source_split(raw_dir, "train"),
        load_source_split(raw_dir, "test"),
    ], ignore_index=True)

    df = df.dropna(subset=["Diagnosis"]).copy().reset_index(drop=True)

    # Convert the original diagnosis and gender codes to start at 0.
    df["Diagnosis"] = df["Diagnosis"].astype(int) - 1
    df["Gender"] = df["Gender"].astype(int) - 1
    df["Education"] = df["Education"].astype(int)
    df["Age"] = df["Age"].astype(float)

    snp_cols = validate_analysis_dataframe(df)
    print(f"Analysis cohort: {len(df)} participants")
    print(f"SNPs used: {len(snp_cols)}")
    print(f"Full CN/MCI/AD counts: {counts3(df['Diagnosis'].to_numpy())}")

    # Set aside the held-out partition before making any of the other splits.
    dev_idx, heldout_idx = train_test_split(
        np.arange(len(df)),
        train_size=DEV_SIZE,
        test_size=HELDOUT_SIZE,
        random_state=seed,
        stratify=df["Diagnosis"].to_numpy(),
    )
    dev_df = df.iloc[dev_idx].copy().reset_index(drop=True)
    heldout_df = df.iloc[heldout_idx].copy().reset_index(drop=True)

    save_predictive_cohort(out_dir / "dev", dev_df, snp_cols)
    save_predictive_cohort(out_dir / "heldout", heldout_df, snp_cols)

    print(f"Development: {len(dev_df)} CN/MCI/AD={counts3(dev_df['Diagnosis'].to_numpy())}")
    print(f"Held-out:    {len(heldout_df)} CN/MCI/AD={counts3(heldout_df['Diagnosis'].to_numpy())}")

    # Create the final synthetic training data.
    final_sizes = internal_tabddpm_split(dev_df, out_dir / "final_dev", snp_cols, seed)

    # Create the repeated cross-validation folds for the development group.
    folds_root = out_dir / "folds"
    folds_root.mkdir(parents=True, exist_ok=True)
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)

    for fold_idx, (train_pos, eval_pos) in enumerate(rskf.split(np.zeros(len(dev_df)), dev_df["Diagnosis"])):
        fold_dir = folds_root / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.save(fold_dir / "train_idx.npy", train_pos.astype(np.int64))
        np.save(fold_dir / "eval_idx.npy", eval_pos.astype(np.int64))

        outer_train_df = dev_df.iloc[train_pos].copy().reset_index(drop=True)
        internal_tabddpm_split(outer_train_df, fold_dir, snp_cols, seed)

        # Save the IDs as a check on the split.
        np.save(fold_dir / "outer_train_participant_id.npy",
                dev_df.iloc[train_pos]["SubjectID"].astype(str).to_numpy(), allow_pickle=True)
        np.save(fold_dir / "outer_eval_participant_id.npy",
                dev_df.iloc[eval_pos]["SubjectID"].astype(str).to_numpy(), allow_pickle=True)

    print(f"Wrote {N_SPLITS * N_REPEATS} fold-specific TabDDPM directories under {folds_root}")
    print(f"Final 433-only TabDDPM internal split: {final_sizes}")
    print(f"Saved prepared data to {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default="data/apoe_txt")
    p.add_argument("--out-dir", default="data/apoe1")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_data(Path(args.raw_dir), Path(args.out_dir), args.seed)
