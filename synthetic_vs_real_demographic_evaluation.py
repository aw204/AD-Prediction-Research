"""
Real vs synthetic demographic fidelity for TabDDPM. Create four plots:
  tab_ddpm_continuous_feature.tif    KDE overlays, age and education
  tab_ddpm_qq_plots.tif              Q-Q plots, age and education
  tab_ddpm_categorical_bar.tif       diagnostic class proportions
  tab_ddpm_gender_by_diagnostic.tif  gender composition within each class
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
from PIL import Image

# Plot styling
plt.rcParams.update({
    'font.family': 'DejaVu Sans',  
    'font.size': 14,
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold'
})

COLOR_REAL = '#0072B2'   # Blue
COLOR_SYNTH = '#D55E00'  # Orange
GENDER_COLORS = ["#4D4D4D", "#BDBDBD"]  # Male (dark grey), Female (light grey)

DATA_DIR = "data/augmented_final"
OUT_DIR = "figures"

# X columns: 0=Age, 1=Education, 2=Gender, 3..114=112 SNPs

def save_tif(fig, path, width_mm=180, dpi=300):
    
    """Save `fig` as RGB, LZW-compressed TIFF, 300 dpi,
    column width (180 mm two-column, 85 mm one-column).
    Preserve figure's layout and aspect ratio. 
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    im = Image.open(buf).convert("RGB")              
    target_w = round(width_mm / 25.4 * dpi)           
    target_h = round(im.height * target_w / im.width) 
    im = im.resize((target_w, target_h), Image.LANCZOS)
    im.save(path, compression="tiff_lzw", dpi=(dpi, dpi))


def generate_eval_plots():
    real_path = os.path.join(DATA_DIR, "X_real_features_115.npy")
    synth_path = os.path.join(DATA_DIR, "X_synth_features_115.npy")
    y_real_path = os.path.join(DATA_DIR, "y_real.npy")
    y_synth_path = os.path.join(DATA_DIR, "y_synth.npy")

    if not (os.path.exists(real_path) and os.path.exists(synth_path)):
        print(f"Error: Could not find dataset files in {DATA_DIR}")
        return

    X_real = np.load(real_path).astype(np.float32)
    X_synth = np.load(synth_path).astype(np.float32)
    y_real = np.load(y_real_path).flatten().astype(int)
    y_synth = np.load(y_synth_path).flatten().astype(int)

    # Gender is stored at column 2
    g_real = np.round(X_real[:, 2]).astype(int)
    g_synth = np.round(X_synth[:, 2]).astype(int)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Continuous Distributions (Age & Education)
    cont_cols = [0, 1]
    cont_names = ["Age (years)", "Education (years)"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, col, name in zip(axes, cont_cols, cont_names):
        sns.kdeplot(X_real[:, col], fill=True, color=COLOR_REAL, label='Real',
                    ax=ax, alpha=0.3, linewidth=2.5)
        sns.kdeplot(X_synth[:, col], fill=True, color=COLOR_SYNTH, label='Synthetic',
                    ax=ax, alpha=0.3, linewidth=2.5)
        ax.set_title(name, pad=10, fontsize=18)
        ax.set_xlabel("Feature Value")
        ax.set_ylabel("Density")
        ax.set_ylim(top=ax.get_ylim()[1] * 1.18)  # Add headroom above peak
        ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
        ax.grid(alpha=0.3)

    plt.suptitle("Distribution Fidelity: Age and Education", y=1.05, fontsize=22)
    plt.tight_layout()
    save_tif(plt.gcf(), os.path.join(OUT_DIR, "tab_ddpm_continuous_feature.tif"))
    plt.close()

    # Diagnostic Class Distribution Bar Chart
    # These match by construction (is_y_cond), not learned. Sanity check only.
    pct_real = np.bincount(y_real, minlength=3) / len(y_real) * 100
    pct_synth = np.bincount(y_synth, minlength=3) / len(y_synth) * 100

    labels = ['CN', 'MCI', 'AD']
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    bars_real = plt.bar(x - width/2, pct_real, width, label='Real',
                        color=COLOR_REAL, edgecolor='black', linewidth=0.5)
    bars_synth = plt.bar(x + width/2, pct_synth, width, label='Synthetic',
                         color=COLOR_SYNTH, edgecolor='black', linewidth=0.5)

    plt.ylabel('Percentage', fontsize=14)
    plt.xlabel('Diagnostic Class', fontsize=14)
    plt.title('Diagnostic Class Distribution', pad=15, fontsize=22)
    plt.xticks(x, labels, fontsize=14)
    plt.legend(title='Dataset', title_fontsize=12, fontsize=12)

    ax = plt.gca()
    ax.set_ylim(0, max(pct_real.max(), pct_synth.max()) * 1.15)
    
    # Annotate percentage labels on top of bars
    for bar in list(bars_real) + list(bars_synth):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.6,
                f"{h:.2f}%", ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_tif(plt.gcf(), os.path.join(OUT_DIR, "tab_ddpm_categorical_bar.tif"))
    plt.close()

    # Quantile-Quantile (Q-Q) Plots
    probs = np.linspace(0.01, 0.99, 99)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, col, name in zip(axes, cont_cols, cont_names):
        real_q = np.quantile(X_real[:, col], probs)
        synth_q = np.quantile(X_synth[:, col], probs)
        lo = min(real_q.min(), synth_q.min())
        hi = max(real_q.max(), synth_q.max())

        ax.plot([lo, hi], [lo, hi], 'k--', lw=1.5, label='Perfect Fidelity')
        ax.scatter(real_q, synth_q, color=COLOR_SYNTH, alpha=0.7, s=25)
        ax.set_title(name, pad=10, fontsize=18)
        ax.set_xlabel("Real Quantiles")
        ax.set_ylabel("Synthetic Quantiles")
        ax.legend(loc='best')
        ax.grid(alpha=0.3)

    plt.suptitle("Quantile-Quantile Fidelity: Age and Education", y=1.05, fontsize=22)
    plt.tight_layout()
    save_tif(plt.gcf(), os.path.join(OUT_DIR, "tab_ddpm_qq_plots.tif"))
    plt.close()

    # Gender Composition by Diagnostic Class
    dx_labels = ["CN", "MCI", "AD"]
    gender_labels = ["Male", "Female"]

    def calc_composition(y, g):
        comp = np.zeros((3, 2))
        for d in range(3):
            mask = (y == d)
            if mask.sum() > 0:
                comp[d] = np.bincount(g[mask], minlength=2)[:2] / mask.sum() * 100
        return comp

    comp_real = calc_composition(y_real, g_real)
    comp_synth = calc_composition(y_synth, g_synth)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    panels = [(comp_real, "Real", COLOR_REAL), (comp_synth, "Synthetic", COLOR_SYNTH)]

    for ax, (comp, name, tint) in zip(axes, panels):
        ax.set_facecolor(tint)
        ax.patch.set_alpha(0.35)
        for spine in ax.spines.values():
            spine.set_edgecolor(tint)
            spine.set_linewidth(3)

        bottom = np.zeros(3)
        for gi, glab in enumerate(gender_labels):
            ax.bar(np.arange(3), comp[:, gi], bottom=bottom, label=glab,
                   color=GENDER_COLORS[gi], edgecolor='black', linewidth=0.5)
            bottom += comp[:, gi]

        ax.set_title(name, pad=10, fontsize=18)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(dx_labels, fontsize=14)
        ax.set_xlabel("Diagnostic Class")
        ax.set_ylabel("Gender Composition (%)")
        ax.set_ylim(0, 118)
        ax.legend(title="Gender", fontsize=9, title_fontsize=10,
                  loc='upper center', ncol=2, framealpha=0.9)
        ax.grid(alpha=0.3, axis='y')

    plt.suptitle("Gender Composition by Diagnostic Class", y=1.03, fontsize=22)
    plt.tight_layout()
    save_tif(plt.gcf(), os.path.join(OUT_DIR, "tab_ddpm_gender_by_diagnostic.tif"))
    plt.close()

    print(f"Successfully generated all 4 fidelity plots in {OUT_DIR}")


if __name__ == "__main__":
    generate_eval_plots()
