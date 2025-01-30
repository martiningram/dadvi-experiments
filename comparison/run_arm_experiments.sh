# This script runs the experiments for the ARM models
# It does this by reading the ARM model names from a file, and piping them into a bash loop.
TARGET_DIR='./experiment_runs/november_2024/models/'
COVERAGE_TARGET_DIR='./experiment_runs/november_2024/coverage/'

while read MODEL_NAME; do
    echo "$MODEL_NAME"

    echo "Running doubling DADVI"
    python fit_doubling_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" 0.25

    echo "Running DADVI"
    python fit_dadvi.py "$MODEL_NAME" "$TARGET_DIR"

    echo "Running MCMC"
    python fit_mcmc.py "$MODEL_NAME" "$TARGET_DIR"

    echo "Running DADVI with Direct LRVB"
    python fit_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" Direct

    echo "Running SADVI with mean field"
    python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" advi

    echo "Running SADVI with full rank"
    python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" fullrank_advi

    echo "Running coverage experiments"
    # Run coverage
    for min_m_power in 3 4 5 6; do
            python run_multiple_dadvi.py \
        	--model-name "$MODEL_NAME" \
        	--target-dir "$COVERAGE_TARGET_DIR" \
        	--min-m-power $min_m_power \
        	--n-reruns 100 \
        	--warm-start
    done;

    echo "Running RAABBVI"
    python fit_raabbvi.py "$MODEL_NAME" "$TARGET_DIR"

done < all_arm_names.txt
