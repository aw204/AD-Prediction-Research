# TabDDPM Augmentation and Cognitive Impairment Prediction in Alzheimer's Disease

This repository contains the code used for the paper *Improving Cognitive Impairment Prediction in Alzheimer's Disease Using Tabular Diffusion Augmentation and Late Fusion Deep Learning*.

The project trains a Tabular Denoising Diffusion Probabilistic Model (TabDDPM) on ADNI *APOE* genomic and demographic data and evaluates whether synthetic data augmentation improves classification of cognitive impairment in Alzheimer's disease. Seven prediction models are implemented and compared, including linear baseline, gradient-boosted trees, and early and late fusion neural networks.

## Data

The model uses data extracted from the Alzheimer's Disease Neuroimaging Initiative (ADNI; adni.loni.usc.edu). The dataset includes:
- age and years of education as continuous features
- gender as a categorical feature
- 168 *APOE*-region SNPs encoded as genotype states 0, 1, and 2
- diagnostic status as the conditioning label for synthetic generation

For predictive modeling, mild cognitive impairment and Alzheimer's disease are combined into one impaired class. ADNI data are not included in this repository and must be obtained through the ADNI data-access process.

## Workflow

- Prepare the demographic, genotype, and diagnostic data for TabDDPM.
- Tune and train the diffusion model.
- Generate synthetic records and examine demographic and SNP fidelity.
- Compare seven prediction models at augmentation ratios of 0× (real data only), 3×, 5×, 7×, and 10×.
- Evaluate the selected categorical late fusion model on the 109-participant held-out set.

## Environment

Machine type: ``Linux 11, 16 vCPUs, 64 GB RAM``.

The TabDDPM pipeline runs in the conda environment:``bash conda activate tddpm``.

The config sets `device = "cpu"`. All scripts run without a GPU, runtimes are manageable at this cohort size. 

## Scripts

- `prepare_data.py` - prepares the raw clinical and genomic data into TabDDPM-ready numpy arrays.
- `synthetic_vs_real_demographic_evaluation.py` - evaluates demographic fidelity of the synthetic data.
- `synthetic_vs_real_snp_evaluation.py` - evaluates SNP fidelity of the synthetic data.
- `model_comparison.py` - compares the seven prediction models across augmentation ratios.
- `heldout_evaluation.py` - evaluates the selected model on the held-out set and creates confusion matrices.

## Prepare Data (`prepare_data.py`)
Creates the TabDDPM input files: `X_num.npy` (continuous features), `X_cat.npy` (categorical features), `y.npy` (diagnostic labels), and `info.json` (column types and metadata). 80% of the data (N = 433) is used for TabDDPM training and prediction model cross-validation, and 20% (N = 109) is used as the held-out set.

## Generate Synthetic Data Using TabDDPM Pipeline

### 1. Hyperparameter Tuning

```bash
python scripts/tune_ddpm.py apoe 433 synthetic mlp ddpm_tune
```

The tuned hyperparameter configuration is written to `exp/apoe1/ddpm_tune_best/config.toml` and includes:

- **Diffusion type:** Gaussian diffusion for continuous features, multinomial diffusion for categorical features and SNPs
- **Conditioning:** class-conditional generation on diagnostic status
- **Model parameters:** MLP hidden layers, dropout rate
- **Training and sampling:** diffusion steps, learning rate, weight decay, batch size, steps

### 2. Model Training & Synthetic Sampling

```bash
python scripts/pipeline.py --config exp/apoe1/ddpm_tune_best/config.toml --train --sample
```

### 3. Multi-seed Evaluation

```bash
python scripts/eval_seeds.py --config exp/apoe1/ddpm_tune_best/config.toml 10 ddpm synthetic mlp 5
```

## Evaluate Synthetic Data Fidelity

Compares the 10,500 synthetic records against the real 433-participant training data.

**Demographic features** (`synthetic_vs_real_demographic_evaluation.py`)
- Age and education: distribution overlap assessed with KDE overlays and Q-Q plots
- Gender: composition within each diagnostic group

**SNP features** (`synthetic_vs_real_snp_evaluation.py`)
- Allele frequencies: real vs. synthetic allele frequency across SNPs
- Genotype frequencies: proportions of the three genotype states (0, 1, 2) compared per SNP

## Compare Prediction Models (`model_comparison.py`)

Evaluates seven model and encoding configurations:

1. **Ridge logistic regression** - linear baseline
2. **XGBoost** - gradient-boosted trees, numeric genotypes
3. **CatBoost** - gradient-boosted trees, native categorical genotypes
4. **Early Fusion, numeric** - features concatenated at the input layer, SNPs as raw genotype values
5. **Early Fusion, categorical** - features concatenated at the input layer, SNPs as learned embeddings
6. **Late Fusion, numeric** - separate demographic and genomic branches, SNPs as raw genotype values
7. **Late Fusion, categorical** - separate demographic and genomic branches, SNPs as learned embeddings

Models are compared at synthetic-to-real augmentation ratios of 0× (real data only), 3×, 5×, 7×, and 10× using 5-fold stratified cross-validation repeated 10 times (50 folds total).

## Evaluate Held-Out Set (`heldout_evaluation.py`)

The top-performing model (**Late Fusion Categorical**) is evaluated on the held-out cohort (N = 109), retrained on the 433-participant training cohort with 0× (real data only) and 7× synthetic augmentation.

* **Standardization:** statistics are fit on the training cohort only and applied unchanged to the held-out set.
* **Threshold selection:** decision thresholds are chosen on training predictions using Youden's J statistic ($\max(\text{Sensitivity} + \text{Specificity} - 1)$) and applied unchanged to the held-out set.
* **Multi-seed runs:** the model is trained 20 times with different random seeds (`N_SEEDS = 20`).
* **Reporting:** sensitivity, specificity, and AUC are reported as the mean and standard deviation across runs. A seed-averaged confusion matrix is also produced by averaging predicted probabilities across the 20 runs.


## TabDDPM reference

This project uses the open-source TabDDPM implementation from Yandex Research:

- Code repository: https://github.com/yandex-research/tab-ddpm
- Kotelnikov, A., Baranchuk, D., Rubachev, I., and Babenko, A. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models*. Proceedings of the 40th International Conference on Machine Learning, 17564–17579.
