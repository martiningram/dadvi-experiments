TARGET_DIR='/media/martin/External Drive/projects/lrvb_paper/dadvi_runs_january_2023'
COVERAGE_TARGET_DIR='/media/martin/External Drive/projects/lrvb_paper/coverage_warm_starts_rerun'

for MODEL_NAME in microcredit occ_det tennis potus; do

	# python fit_raabbvi.py "$MODEL_NAME" "$TARGET_DIR"
	# python fit_dadvi.py "$MODEL_NAME" "$TARGET_DIR"
	# python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" advi
	# python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" fullrank_advi
	# python fit_mcmc.py "$MODEL_NAME" "$TARGET_DIR"

	# python fit_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" Direct
        # python fit_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" CG

	# python fit_doubling_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" 0.25

        # Run coverage
	for min_m_power in 3 4 5 6; do

		python run_multiple_dadvi.py \
			--model-name "$MODEL_NAME" \
			--target-dir "$COVERAGE_TARGET_DIR" \
			--min-m-power $min_m_power \
			--n-reruns 10 \
			--warm-start

	done;

done
