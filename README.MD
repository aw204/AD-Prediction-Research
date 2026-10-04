# TabDDPM Augmentation and Cognitive Impairment Classification in Alzheimer's Disease

This repository contains scripts for the paper *Improving Cognitive Impairment Classification in Alzheimer's Disease Using Tabular Diffusion Augmentation and Late Fusion Deep Learning*. The scripts evaluate Tabular Denoising Diffusion Probabilistic Model (TabDDPM) augmentation and late-fusion models using ADNI demographic data and 168 SNPs from the APOE region.

## Data

The model uses data extracted from the Alzheimer's Disease Neuroimaging Initiative (ADNI; adni.loni.usc.edu). The dataset includes:

- age and years of education as continuous features
- gender as a categorical feature
- 168 *APOE*-region SNPs encoded as genotype states 0, 1, and 2
- diagnostic status as the conditioning label for synthetic generation

For predictive modeling, mild cognitive impairment (MCI) and Alzheimer's disease (AD) are combined into a single cognitively impaired class. ADNI data are not included in this repository and must be obtained through the ADNI data-access process.

The final analysis cohort contains 542 participants. A stratified 80/20 split is used to create a 433-participant development cohort and a 109-participant held-out cohort. The held-out cohort is reserved for final evaluation and is not used during TabDDPM development, predictive-model selection, preprocessing estimation, or threshold selection.

For TabDDPM, continuous inputs are stored in `X_num.npy` (age and education), while categorical inputs are stored in `X_cat.npy` (gender and the 168 SNP genotypes). The predictive-model scripts reconstruct the 171-feature matrix by concatenating these arrays.

## Workflow

1. Prepare the demographic, genotype, and diagnostic data.
2. Create repeated stratified cross-validation splits within the 433-participant development cohort.
3. Within each outer cross-validation run, train and tune a separate TabDDPM using that run's outer training partition and generate run-specific synthetic data.
4. Compare eight predictive-model configurations at 0x, 3x, 5x, 7x, and 10x augmentation.
5. Run demographic, SNP, feature-set, resampling, embedding-dimension, and duplication checks.
6. Evaluate the selected model on the 109-participant held-out cohort.

## Environment

The TabDDPM pipeline runs in the conda environment:

```bash
conda activate tddpm
```

The configuration uses `device = "cpu"` and does not require a GPU.

## Main scripts

- `prepare_data.py` - prepares the clinical and genomic data and creates the development and held-out splits
- `run_nested_generation.sh` - runs run-specific TabDDPM tuning and synthetic generation
- `tune_ddpm.py` - performs Optuna-based TabDDPM hyperparameter tuning
- `collect_synthetic.py` - combines TabDDPM output into the 171-feature format used by the predictive-model scripts and saves the run-specific synthetic datasets
- `synthetic_vs_real_demographic_evaluation.py` - evaluates demographic fidelity
- `synthetic_vs_real_snp_evaluation.py` - evaluates SNP fidelity
- `duplication_check.py` - checks whether any synthetic 168-SNP genotype vector exactly matches a real development participant
- `model_comparison.py` - compares the eight predictive-model configurations across augmentation ratios
- `model_utils.py` - contains shared preprocessing, model definitions, and prediction functions used by several scripts
- `baseline_comparisons.py` - runs the demographics-only, SNP-only, and RandomOverSampler comparisons
- `embedding_dimension_sensitivity.py` - compares categorical embedding dimensions
- `heldout_evaluation.py` - evaluates the selected model on the held-out cohort and runs the participant-level bootstrap analysis

## 1. Prepare the data

```bash
python scripts/prepare_data.py
```

`prepare_data.py` creates the TabDDPM inputs, including `X_num.npy`, `X_cat.npy`, `y.npy`, and `info.json`, along with the development and held-out partitions.

The 433-participant development cohort is used for model development. The 109-participant held-out cohort remains separate until the final evaluation.

## 2. Run-specific TabDDPM generation

```bash
bash scripts/run_nested_generation.sh
```

Repeated stratified 5-fold cross-validation with 10 repeats is run within the development cohort, producing 50 outer cross-validation runs.

TabDDPM is developed only from the real participants in the outer training partition. Those participants are further divided using an internal stratified 80/10/10 split for generator training, validation, and an internal test split. The outer evaluation participants are not used for generator training, tuning, preprocessing, or synthetic generation.

`run_nested_generation.sh` calls `tune_ddpm.py` separately for each outer run using that run's prepared TabDDPM training, validation, and test data. Optuna tunes the TabDDPM hyperparameters separately within each run.

After hyperparameter selection, the selected run-specific generator produces a synthetic pool large enough to support the maximum 10x augmentation condition. The internal test subset is not used for hyperparameter selection or final generator fitting.

TabDDPM uses Gaussian diffusion for age and education and multinomial diffusion for gender and SNP genotype states. Generation is conditioned on diagnostic status.

`run_nested_generation.sh` calls `collect_synthetic.py` automatically for every outer run and for `final_dev`. It combines the generated continuous and categorical features into the 171-column format used by the downstream predictive models and saves the synthetic pool.

To collect output manually for one outer run, pass its stored fold index:

```bash
python scripts/collect_synthetic.py 00
```

To collect output for the final development set, use:

```bash
python scripts/collect_synthetic.py final_dev
```

## 3. Synthetic-data evaluation

Demographic fidelity is evaluated with:

```bash
python scripts/synthetic_vs_real_demographic_evaluation.py
```

The demographic checks include age and education distributions, Q-Q plots, and gender composition within diagnostic groups.

SNP fidelity is evaluated with:

```bash
python scripts/synthetic_vs_real_snp_evaluation.py
```

The SNP analysis compares real and synthetic allele and genotype frequencies across all 168 SNPs and summarizes variant capture by minor allele frequency band.

Exact replication of complete genotype vectors is checked with:

```bash
python scripts/duplication_check.py
```

This compares each synthetic sample's complete 168-SNP genotype vector with the genotype vectors of all 433 real development-cohort participants.

## 4. Predictive-model comparison

```bash
python scripts/model_comparison.py
```
This evaluates the eight predictive-model configurations at 0x, 3x, 5x, 7x, and 10x augmentation using the same 50 outer cross-validation runs.

Eight model configurations are evaluated:

1. **Ridge logistic regression** - L2-regularized linear baseline
2. **Unregularized logistic regression** - logistic regression without an L1 or L2 penalty
3. **XGBoost** - gradient-boosted trees with numeric genotype inputs
4. **CatBoost** - gradient-boosted trees with categorical genotype inputs
5. **Early Fusion, numeric** - demographic and SNP features concatenated at the input layer
6. **Early Fusion, categorical** - demographic features combined with learned SNP embeddings at the input layer
7. **Late Fusion, numeric** - separate demographic and genomic branches with numeric SNP inputs
8. **Late Fusion, categorical** - separate demographic and genomic branches with learned SNP embeddings

Each model is evaluated at 0x (real data only), 3x, 5x, 7x, and 10x synthetic augmentation using the same 50 outer cross-validation runs. Synthetic samples are added only to the outer training partition; outer evaluation is always performed on real participants. Synthetic records are drawn class-proportionally without replacement, and the requested augmentation size must be available in the run-specific synthetic pool.

The augmented conditions are compared with the corresponding 0x condition using paired DeLong tests, with Holm correction for multiple comparisons.

```bash
python scripts/baseline_comparisons.py
```
This performs three additional analyses using the same outer cross-validation folds:

- demographics-only L2-regularized logistic regression using age, education, and gender
- SNP-only L2-regularized logistic regression using all 168 SNPs
- RandomOverSampler as a non-generative class-rebalancing control using the selected predictive-model architecture

```bash
python scripts/embedding_dimension_sensitivity.py
```
This evaluates the categorical late-fusion model using SNP embedding dimensions of 4, 8, and 16 under real-only training and the selected augmentation ratio.

## 5. Final held-out evaluation

```bash
python scripts/heldout_evaluation.py
```

The best model from development is evaluated on the 109-participant held-out cohort under two conditions: real data only (0x) and real data plus the selected synthetic augmentation ratio.

The threshold maximizing Youden's J statistic is selected from development-cohort predictions and then applied to the held-out participants.

The held-out model is retrained with different random seeds, and AUC, sensitivity, and specificity are reported as mean and standard deviation.

Uncertainty is estimated with paired stratified bootstrap resampling. BCa 95% confidence intervals are calculated for the metrics and paired differences.

## Reproducibility seeds

The development/held-out split and cross-validation runs are generated with seed `42`. Held-out evaluation retrains the selected model with seeds `0`–`19` and reports mean ± SD across the 20 runs. The participant-level bootstrap uses seed `42`.

## TabDDPM reference

This project uses the open-source TabDDPM implementation from Yandex Research:

- Code repository: https://github.com/yandex-research/tab-ddpm
- Kotelnikov, A., Baranchuk, D., Rubachev, I., and Babenko, A. (2023). *TabDDPM: Modelling Tabular Data with Diffusion Models*. Proceedings of the 40th International Conference on Machine Learning, 17564-17579.
