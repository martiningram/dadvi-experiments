TARGET_DIR='./experiment_runs/december_2023/'
COVERAGE_TARGET_DIR='./experiment_runs/coverage/'

# Check tennis data exists and fetch it if not
if [ ! -f ./data/tennis_atp/atp_players.csv ]; then
	echo "Fetching tennis data as it was not found"
	bash fetch_tennis_data.sh
fi

# for MODEL_NAME in microcredit occ_det tennis potus; do
for MODEL_NAME in potus; do

        echo "$MODEL_NAME"

	# echo "Running DADVI"
	# python fit_dadvi.py "$MODEL_NAME" "$TARGET_DIR"

	# if [ $MODEL_NAME == 'tennis' ]
	# then
	# 	echo "Computing the LRVB correction for 20 tennis matchups"
	# 	python compute_tennis_matchups.py \
	# 		--experiment-base-dir "$TARGET_DIR"
	# fi

	# if [ $MODEL_NAME == 'occ_det' ]
	# then
	# 	echo "Computing the LRVB correction for 20 species"
	# 	python compute_occu_predictions.py \
	# 		--experiment-base-dir "$TARGET_DIR"
	# fi

	# if [ $MODEL_NAME == 'potus' ]
	# then
	# 	echo "Computing the LRVB correction for the final vote share"
	# 	python compute_potus_predictions.py \
	# 		--experiment-base-dir "$TARGET_DIR"
	# fi

	# echo "Running NUTS"
	# python fit_mcmc.py "$MODEL_NAME" "$TARGET_DIR"

	# echo "Running SADVI mean field"
	# python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" advi

	# echo "Running RAABBVI"
	# python fit_raabbvi.py "$MODEL_NAME" "$TARGET_DIR"

	# if [ $MODEL_NAME != 'potus' ]
	# then
	# 	echo "Running bigger models also."

	# 	echo "Running doubling DADVI"
	# 	python fit_doubling_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" 0.25

	# 	echo "Running LRVB Direct"
	# 	python fit_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" Direct

	# 	echo "Running PyMC SADVI Full rank"
	# 	python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" fullrank_advi
	# fi

        # Run coverage
	# for M in 8 16 32 64; do
	for M in 16; do

		echo "Running with $M fixed draws"

		python run_multiple_dadvi_no_doubling.py \
			--model-name "$MODEL_NAME" \
			--target-dir "$COVERAGE_TARGET_DIR" \
			--M $M \
			--n-reruns 2 \
			--warm-start
			# --n-reruns 100 \

	done;

	if [ $MODEL_NAME == 'tennis' ]
	then
		echo "Computing the coverage summary for 20 tennis matchups"
		python summarise_tennis_coverage.py \
			--coverage-base-dir "$COVERAGE_TARGET_DIR"
	fi

	if [ $MODEL_NAME == 'occ_det' ]
	then
		echo "Computing the coverage summary for 20 species predictions"
		python summarise_occu_coverage.py \
			--coverage-base-dir "$COVERAGE_TARGET_DIR"
	fi

	if [ $MODEL_NAME == 'potus' ]
	then
		echo "Computing the coverage summary for the final vote share"
		python summarise_potus_coverage.py \
			--coverage-base-dir "$COVERAGE_TARGET_DIR"
	fi



done
