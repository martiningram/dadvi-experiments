TARGET_DIR='/media/martin/External Drive/projects/lrvb_paper/dadvi_runs_october_2022'
COVERAGE_TARGET_DIR='/media/martin/External Drive/projects/lrvb_paper/coverage_warm_starts'

while read MODEL_NAME; do
    echo "$MODEL_NAME"

    # Run inference
    # python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" advi
    # python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" fullrank_advi
    # python fit_raabbvi.py "$MODEL_NAME" "$TARGET_DIR"
    # python fit_dadvi.py "$MODEL_NAME" "$TARGET_DIR"
    # python fit_mcmc.py "$MODEL_NAME" "$TARGET_DIR"
    # python fit_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" "$TARGET_DIR" CG
    # python fit_doubling_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR"

    # Run coverage
    python run_multiple_dadvi.py \
	--model-name "$MODEL_NAME" \
	--target-dir "$COVERAGE_TARGET_DIR" \
	--min-m-power 6 \
	--n-reruns 100 \
	--warm-start

done < all_arm_names.txt


