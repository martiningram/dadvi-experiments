# This script runs the experiments for the ARM models
# It does this by reading the ARM model names from a file, and piping them into a bash loop.
TARGET_DIR='./experiment_runs/december_2023'
COVERAGE_TARGET_DIR='./experiment_runs/coverage/'
TEST_RUN=true

if [ "$TEST_RUN" = true ] ; then
	ADDED_ARGS='--test-run'
else 
	ADDED_ARGS=''
fi

while read MODEL_NAME; do
    echo "$MODEL_NAME"

    echo "Running DADVI"
    python fit_dadvi.py \
	    --model-name "$MODEL_NAME" \
	    --target-dir "$TARGET_DIR" \
	    ${TEST_RUN:+"$ADDED_ARGS"}

    echo "Running MCMC"
    python fit_mcmc.py \
	    --model-name "$MODEL_NAME" \
	    --target-dir "$TARGET_DIR" \
	    ${TEST_RUN:+"$ADDED_ARGS"}

    if [ "$TEST_RUN" = false ] ; then

        # TODO: Work out how to do the test run here;
        # it doesn't work because the matrix has to be positive definite

        echo "Running DADVI with Direct LRVB"
        python fit_dadvi_lrvb.py \
            --model-name "$MODEL_NAME" \
            --target-dir "$TARGET_DIR" \
            --lrvb-method Direct

    fi

    echo "Running SADVI with mean field"
    python fit_pymc_sadvi.py \
            --model-name "$MODEL_NAME" \
            --target-dir "$TARGET_DIR" \
            --advi-method advi \
	        ${TEST_RUN:+"$ADDED_ARGS"}

    echo "Running SADVI with full rank"
    python fit_pymc_sadvi.py \
            --model-name "$MODEL_NAME" \
            --target-dir "$TARGET_DIR" \
            --advi-method fullrank_advi \
	        ${TEST_RUN:+"$ADDED_ARGS"}

    break

    echo "Running doubling DADVI"
    python fit_doubling_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" 0.25

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
