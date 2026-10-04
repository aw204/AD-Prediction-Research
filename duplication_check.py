"""Check whether synthetic SNP vectors exactly match real ones."""

import argparse
import csv
from pathlib import Path

import numpy as np

N_SNPS = 168
SNP_START = 3

def row_keys(a: np.ndarray):
    a = np.ascontiguousarray(a)
    return [row.tobytes() for row in a]

def run(data_root: Path, results_dir: Path):
    Xr = np.load(data_root / "dev" / "X_real_features_171.npy").astype(np.float32)
    Xs = np.load(data_root / "final_dev" / "synthetic_features.npy").astype(np.float32)

    R = np.rint(Xr[:, SNP_START:SNP_START + N_SNPS]).astype(np.int8)
    S = np.rint(Xs[:, SNP_START:SNP_START + N_SNPS]).astype(np.int8)

    real_keys = set(row_keys(R))
    synth_keys = row_keys(S)
    match_mask = np.array([k in real_keys for k in synth_keys], dtype=bool)

    n_match = int(match_mask.sum())
    n_syn = int(len(S))
    pct = 100.0 * n_match / n_syn
    n_unique_syn = int(len(set(synth_keys)))

    results_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "real_records": int(len(R)),
        "synthetic_records": n_syn,
        "synthetic_exact_real_snp_matches": n_match,
        "synthetic_exact_real_snp_match_percent": pct,
        "unique_synthetic_snp_vectors": n_unique_syn,
    }
    with open(results_dir / "exact_duplication_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    np.save(results_dir / "synthetic_exact_match_mask.npy", match_mask)

    print(f"Real development genotype vectors: {len(R)}")
    print(f"Synthetic genotype vectors: {n_syn}")
    print(f"Synthetic vectors exactly matching >=1 real vector: {n_match} ({pct:.3f}%)")
    print(f"Unique synthetic genotype vectors: {n_unique_syn}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default="data/apoe1")
    p.add_argument("--results-dir", default="results/duplication")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(Path(a.data_root), Path(a.results_dir))
