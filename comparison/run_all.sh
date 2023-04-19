TARGET_DIR='/home/martin.ingram/experiment_runs/march_2023/'
COVERAGE_TARGET_DIR='/home/martin.ingram/experiment_runs/march_2023_coverage/'

# for MODEL_NAME in microcredit occ_det tennis potus; do
for MODEL_NAME in tennis occ_det potus; do

        echo "$MODEL_NAME"

	# echo "Running DADVI"
	# python fit_dadvi.py "$MODEL_NAME" "$TARGET_DIR"

	# echo "Running NUTS"
	# python fit_mcmc.py "$MODEL_NAME" "$TARGET_DIR"

	# echo "Running SADVI mean field"
	# python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" advi

	# echo "Running RAABBVI"
	# python fit_raabbvi.py "$MODEL_NAME" "$TARGET_DIR"

	# if [ $MODEL_NAME != 'potus' ]
	# then
		# echo "Running bigger models also."

		# echo "Running doubling DADVI"
		# python fit_doubling_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" 0.25

		# echo "Running LRVB Direct"
		# python fit_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" Direct

		# echo "Running PyMC SADVI Full rank"
		# python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" fullrank_advi
	# fi

        # Run coverage
	for M in 8 16 32 64; do

		echo "Running with $M fixed draws"

		python run_multiple_dadvi_no_doubling.py \
			--model-name "$MODEL_NAME" \
			--target-dir "$COVERAGE_TARGET_DIR" \
			--M $M \
			--n-reruns 100 \
			--warm-start

	done;

done
