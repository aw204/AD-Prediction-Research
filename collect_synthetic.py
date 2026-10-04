"""Collect TabDDPM output into the 171-feature format for the predictive models.

Reads X_num_train.npy, X_cat_train.npy, and y_train.npy from
TabDDPM output directory: exp/apoe_fold/ddpm_tune_best/, and writes
synthetic_features.npy and synthetic_y.npy into the output directory.
"""

import argparse
from pathlib import Path
import numpy as np

N_FEATURES = 171
N_CAT = 169  # gender + 168 SNPs


def collect_synthetic(target: str, data_root: Path, synth_src: Path):
    x_num = np.load(synth_src / "X_num_train.npy").astype(np.float32)
    x_cat = np.load(synth_src / "X_cat_train.npy").astype(np.float32)
    y = np.load(synth_src / "y_train.npy").astype(np.int64).ravel()

    if x_num.shape[1] != 2:
        raise ValueError(f"Expected 2 continuous columns, found {x_num.shape[1]}")
    if x_cat.shape[1] != N_CAT:
        raise ValueError(f"Expected {N_CAT} categorical columns (gender + 168 SNPs), found {x_cat.shape[1]}")
    X = np.hstack([x_num, x_cat]).astype(np.float32)
    if X.shape[1] != N_FEATURES:
        raise ValueError(f"Expected {N_FEATURES} features, found {X.shape[1]}")
    if len(X) != len(y):
        raise ValueError("Synthetic feature and label row counts differ")

    if target == "final_dev":
        out = data_root / "final_dev"
    else:
        out = data_root / "folds" / f"fold_{int(target):02d}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "synthetic_features.npy", X)
    np.save(out / "synthetic_y.npy", y)
    print(f"[{target}] saved {len(X)} synthetic records -> {out}")
    print(f"[{target}] CN/MCI/AD synthetic counts: {np.bincount(y, minlength=3)}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("target", help="Fold number (0-49) or final_dev")
    p.add_argument("--data-root", default="data/apoe1")
    p.add_argument("--synth-src", default="exp/apoe_fold/ddpm_tune_best")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    collect_synthetic(a.target, Path(a.data_root), Path(a.synth_src))
