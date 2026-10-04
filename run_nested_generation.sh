#!/usr/bin/env bash
# Generate synthetic data for each outer cross-validation run and for the final development set.
# tune_ddpm.py is called with
#   python scripts/tune_ddpm.py apoe_fold <outer_train_n> synthetic mlp ddpm_tune
# Run prepare_data.py first. 

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-data/apoe1}"
WORK_DATA="${WORK_DATA:-data/apoe_fold}"
WORK_EXP="${WORK_EXP:-exp/apoe_fold}"
BASE_CONFIG="${BASE_CONFIG:-base_config.toml}"
MAX_RATIO="${MAX_RATIO:-10}"
NUM_FOLDS="${NUM_FOLDS:-50}"

if [[ ! -d "$DATA_ROOT/folds" ]]; then
    echo "ERROR: $DATA_ROOT/folds not found — run prepare_data.py first" >&2
    exit 1
fi

run_one () {
    local source_dir="$1"
    local target="$2"
    local outer_n="$3"

    echo "Running ${target} with ${outer_n} training participants"
    rm -rf "$WORK_DATA" "$WORK_EXP"
    cp -r "$source_dir" "$WORK_DATA"
    mkdir -p "$WORK_EXP"
    cp "$BASE_CONFIG" "$WORK_EXP/config.toml"

    # uses only the data copied into data/apoe_fold.
    python scripts/tune_ddpm.py apoe_fold "$outer_n" synthetic mlp ddpm_tune

    best_config="$WORK_EXP/ddpm_tune_best/config.toml"
    if [[ ! -f "$best_config" ]]; then
        echo "ERROR: expected tuned config at $best_config" >&2
        exit 1
    fi

    #  Generate enough samples for the largest augmentation ratio.
    n_samples=$(( outer_n * MAX_RATIO ))
    python scripts/set_tabddpm_sample_count.py "$best_config" "$n_samples"
    python scripts/pipeline.py --config "$best_config" --sample

    python scripts/collect_synthetic.py "$target" --data-root "$DATA_ROOT" \
        --synth-src "$WORK_EXP/ddpm_tune_best"

    if [[ "$target" == "final_dev" ]]; then
        dest="$DATA_ROOT/final_dev"
    else
        printf -v fold_name "fold_%02d" "$target"
        dest="$DATA_ROOT/folds/$fold_name"
    fi
    cp "$best_config" "$dest/selected_tabddpm_config.toml"

    # Fail if the final sampled pool is too small.
    python - "$dest" "$n_samples" <<'PY'
import sys, numpy as np
from pathlib import Path
d=Path(sys.argv[1]); need=int(sys.argv[2])
X=np.load(d/'synthetic_features.npy'); y=np.load(d/'synthetic_y.npy')
assert len(X)==len(y)
assert len(X)>=need, f"{d}: synthetic pool {len(X)} < required {need}"
print(f"verified {d}: {len(X)} synthetic records")
PY
}

for i in $(seq 0 $((NUM_FOLDS - 1))); do
    fold_dir=$(printf "%s/folds/fold_%02d" "$DATA_ROOT" "$i")
    outer_n=$(python - "$fold_dir/train_idx.npy" <<'PY'
import sys, numpy as np
print(len(np.load(sys.argv[1])))
PY
)
    run_one "$fold_dir" "$i" "$outer_n"
done

# only the 433 development participants are represented in final_dev/.
run_one "$DATA_ROOT/final_dev" "final_dev" 433

echo "All synthetic pools are complete."
