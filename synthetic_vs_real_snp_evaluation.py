"""
SNP fidelity evaluation: Compare real vs synthetic SNP allele and genotype frequencies
across the 112 active SNPs, the polymorphic subset of the 168-SNP APOE panel.
Create a 2-panel figure snp_fidelity.png.
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Figure styling
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 16,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 14
})

# Feature layout: Col0 = Age, Col 1= Education, Col 2=Gender, Col 3:115 = 112 SNPs
DATA_DIR = "data/augmented_final"
OUT_DIR = "figures"
SNP_START = 3
N_SNPS = 112


def plot_snp_fidelity():
    real_file = os.path.join(DATA_DIR, "X_real_features_115.npy")
    synth_file = os.path.join(DATA_DIR, "X_synth_features_115.npy")

    if not (os.path.exists(real_file) and os.path.exists(synth_file)):
        print(f"Error: Could not find dataset files in {DATA_DIR}")
        return

    X_real = np.load(real_file).astype(np.float32)
    X_synth = np.load(synth_file).astype(np.float32)

    cols = slice(SNP_START, SNP_START + N_SNPS)
    
    # round float outputs to discrete allele counts (0, 1, 2)
    R_snps = np.clip(np.rint(X_real[:, cols]), 0, 2).astype(int)
    S_snps = np.clip(np.rint(X_synth[:, cols]), 0, 2).astype(int)

    # allele frequency = mean count / 2
    real_af = R_snps.mean(axis=0) / 2.0
    synth_af = S_snps.mean(axis=0) / 2.0

    gp_real = np.column_stack([(R_snps == g).mean(axis=0) for g in range(3)])
    gp_synth = np.column_stack([(S_snps == g).mean(axis=0) for g in range(3)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.5))

    # Panel 1: Allele Frequency 
    axes[0].plot([0, 1], [0, 1], 'k--', lw=2.0)  # identity line: points on it match exactly
    axes[0].scatter(real_af, synth_af, color='#D55E00', alpha=0.8, 
                    edgecolor='black', linewidth=0.5, s=80)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Allele Frequency", pad=12, fontsize=18)
    axes[0].set_xlabel("Real Allele Frequency")
    axes[0].set_ylabel("Synthetic Allele Frequency")
    axes[0].grid(alpha=0.3)

    # Panel 2: Genotype Frequency
    geno_colors = ['#1b9e77', '#7570b3', '#d95f02']
    axes[1].plot([0, 1], [0, 1], 'k--', lw=2.0)
    for g in range(3):
        axes[1].scatter(gp_real[:, g], gp_synth[:, g], alpha=0.8, s=75,
                        edgecolor='black', linewidth=0.5, color=geno_colors[g],
                        label=f"Genotype {g}")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Genotype Frequency", pad=12, fontsize=18)
    axes[1].set_xlabel("Real Genotype Frequency")
    axes[1].set_ylabel("Synthetic Genotype Frequency")
    axes[1].legend(loc='upper left')
    axes[1].grid(alpha=0.3)

    plt.suptitle("SNP Frequency Fidelity", y=1.08, fontsize=24)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    save_path = os.path.join(OUT_DIR, "snp_fidelity.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved figure to {save_path}")


if __name__ == "__main__":
    plot_snp_fidelity()
