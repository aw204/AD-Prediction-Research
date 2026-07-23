"""
Prepare APOE dataset splits into numpy arrays and info.json for TabDDPM model training.
Drop rows without diagnosis label.
Remove monomorphic SNPs before splitting.
Of the 168 SNPs, 56 are monomorphic across the cohort and are dropped, leaving 112.
"""

import json
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = "data/apoe_txt"
OUT_DIR = "data/apoe1"


def load_split_data(prefix):
    snps = pd.read_csv(os.path.join(RAW_DIR, f'apoe_{prefix}_data.txt'), sep=r'\s+', header=None)
    snps.columns = [f"SNP_{i}" for i in range(snps.shape[1])]
    
    dx = pd.read_csv(os.path.join(RAW_DIR, f'{prefix}_dx.txt'), sep=r'\s+', header=None)
    age = pd.read_csv(os.path.join(RAW_DIR, f'{prefix}_age.txt'), sep=r'\s+', header=None)
    gender = pd.read_csv(os.path.join(RAW_DIR, f'{prefix}_gender.txt'), sep=r'\s+', header=None)
    educ = pd.read_csv(os.path.join(RAW_DIR, f'{prefix}_education.txt'), sep=r'\s+', header=None)

    return pd.concat([
        dx[[1]].rename(columns={1: 'Diagnosis'}),
        age[[1]].rename(columns={1: 'Age'}),
        gender[[1]].rename(columns={1: 'Gender'}),
        educ[[1]].rename(columns={1: 'Education'}),
        snps
    ], axis=1)


def prepare_data():
    if os.path.exists(OUT_DIR):
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # Combine train/test text files and drop rows that are missing diagnosis labels
    df = pd.concat([load_split_data('train'), load_split_data('test')], ignore_index=True)
    df = df.dropna(subset=['Diagnosis']).copy()

    # Convert to 0-indexed integers for PyTorch compatibility (1/2/3 -> 0/1/2)
    df['Diagnosis'] = df['Diagnosis'].astype(int) - 1
    df['Gender'] = df['Gender'].astype(float).astype(int) - 1
    df['Education'] = df['Education'].astype(float).astype(int)

    # Filter out monomorphic (zero-variance) SNPs. Screened on the full cohort, before splitting
    all_snps = [col for col in df.columns if col.startswith('SNP_')]
    valid_snp_cols = [col for col in all_snps if df[col].nunique() >= 2]
    
    print(f"Total records: {len(df)} | Retained active SNPs: {len(valid_snp_cols)}")

    # Take unique genotype combinations to guarantee pattern coverage in train
    combo_cols = ['Diagnosis', 'Gender'] + valid_snp_cols
    unique_combos = df.drop_duplicates(subset=combo_cols)

    anchor_idx = unique_combos.index.tolist()
    rem_idx = df.index.difference(anchor_idx).tolist()

    # Fill remaining training set up to 433 rows
    n_padding = 433 - len(anchor_idx)
    np.random.seed(42)
    pad_idx = np.random.choice(rem_idx, size=n_padding, replace=False).tolist()
    train_idx = anchor_idx + pad_idx

    # Split leftover rows between validation and test
    leftover_idx = list(set(df.index) - set(train_idx))
    val_idx, test_idx = train_test_split(leftover_idx, test_size=0.50, random_state=42)

    print(f"Splits -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        sub_df = df.loc[idx]

        y = sub_df['Diagnosis'].values.astype(np.int64)
        x_num = sub_df[['Age', 'Education']].values.astype(np.float32)

        # Categorical features formatted as strings for TabDDPM (Gender + SNPs)
        g_str = sub_df['Gender'].astype(str).values.reshape(-1, 1)
        s_str = sub_df[valid_snp_cols].fillna(0).astype(float).astype(int).astype(str).values
        x_cat = np.hstack([g_str, s_str])

        np.save(os.path.join(OUT_DIR, f'X_num_{name}.npy'), x_num)
        np.save(os.path.join(OUT_DIR, f'X_cat_{name}.npy'), x_cat, allow_pickle=True)
        np.save(os.path.join(OUT_DIR, f'y_{name}.npy'), y)
        np.save(os.path.join(OUT_DIR, f'Y_{name}.npy'), y)

    info = {
        "name": "apoe1",
        "id": "apoe1",
        "task_type": "multiclass",
        "n_classes": 3,
        "num_col_indices": [0, 1],
        "cat_col_indices": list(range(1 + len(valid_snp_cols))),
        "target_col_indices": [0],
        "label_decode": {"0": "CN", "1": "MCI", "2": "AD"}
    }
    with open(os.path.join(OUT_DIR, 'info.json'), 'w') as f:
        json.dump(info, f, indent=4)

    print(f"Saved dataset files to {OUT_DIR}/")


if __name__ == "__main__":
    prepare_data()
