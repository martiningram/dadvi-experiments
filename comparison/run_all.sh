TARGET_DIR='/home/martin.ingram/experiment_runs/march_2023/'
COVERAGE_TARGET_DIR='/media/martin/External Drive/projects/lrvb_paper/coverage_warm_starts_rerun'

# for MODEL_NAME in microcredit occ_det tennis potus; do
for MODEL_NAME in tennis; do

        echo "$MODEL_NAME"

	# echo "Running DADVI"
	# python fit_dadvi.py "$MODEL_NAME" "$TARGET_DIR"

	# echo "Running NUTS"
	# python fit_mcmc.py "$MODEL_NAME" "$TARGET_DIR"

	# echo "Running SADVI mean field"
	# python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" advi

	echo "Running RAABBVI"
	python fit_raabbvi.py "$MODEL_NAME" "$TARGET_DIR"

	if [ $MODEL_NAME != 'potus' ]
	then
		echo "Hi there"
		# echo "Running bigger models also."

		# echo "Running doubling DADVI"
		# python fit_doubling_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" 0.25

		# echo "Running LRVB Direct"
		# python fit_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" Direct

		# echo "Running PyMC SADVI Full rank"
		# python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" fullrank_advi
	fi

        # Run coverage
	# for min_m_power in 3 4 5 6; do

	# 	python run_multiple_dadvi.py \
	# 		--model-name "$MODEL_NAME" \
	# 		--target-dir "$COVERAGE_TARGET_DIR" \
	# 		--min-m-power $min_m_power \
	# 		--n-reruns 10 \
	# 		--warm-start

	# done;

done
