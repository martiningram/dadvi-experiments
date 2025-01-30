TARGET_DIR='./experiment_runs/november_2024/models/'
COVERAGE_TARGET_DIR='./experiment_runs/november_2024/coverage/'

# Check tennis data exists and fetch it if not
if [ ! -f ./data/tennis_atp/atp_players.csv ]; then
	echo "Fetching tennis data as it was not found"
	bash fetch_tennis_data.sh
fi

for MODEL_NAME in microcredit occ_det tennis potus; do

        echo "$MODEL_NAME"

	echo "Running DADVI"
	python fit_dadvi.py "$MODEL_NAME" "$TARGET_DIR"

	### Make draws from LRVB corrected posterior distributions

	# Depending on the model, we compute different quantities of interest
	# (tennis matchups, species predictions, final vote share), and then
	# we make draws and save them in the experiments folder under "lrvb_cg".
	if [ $MODEL_NAME == 'tennis' ]
	then
		echo "Computing the LRVB correction for 20 tennis matchups"
		python compute_tennis_matchups.py \
			--experiment-base-dir "$TARGET_DIR"
	fi

	if [ $MODEL_NAME == 'occ_det' ]
	then
		echo "Computing the LRVB correction for 20 species"
		python compute_occu_predictions.py \
			--experiment-base-dir "$TARGET_DIR"
	fi

	if [ $MODEL_NAME == 'potus' ]
	then
		echo "Computing the LRVB correction for the final vote share"
		python compute_potus_predictions.py \
			--experiment-base-dir "$TARGET_DIR"
	fi


	### Run comparison models

	echo "Running NUTS"
	python fit_mcmc.py "$MODEL_NAME" "$TARGET_DIR"

	echo "Running SADVI mean field"
	python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" advi

	echo "Running RAABBVI"
	python fit_raabbvi.py "$MODEL_NAME" "$TARGET_DIR"

	# POTUS is so big that we don't even attempt to run some of the models:
	if [ $MODEL_NAME != 'potus' ]
	then
		echo "Running bigger models also."

		echo "Running doubling DADVI"
		python fit_doubling_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" 0.25

		echo "Running LRVB Direct"
		python fit_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" Direct

		echo "Running PyMC SADVI Full rank"
		python fit_pymc_sadvi.py "$MODEL_NAME" "$TARGET_DIR" fullrank_advi
	fi

	### COVERAGE EXPERIMENTS

	# Run the DADVI fit multiple times for each number of fixed draws, so that
	# we can analyse the quality of the confidence intervals
	for M in 8 16 32 64; do

		echo "Running with $M fixed draws"

		python run_multiple_dadvi_no_doubling.py \
			--model-name "$MODEL_NAME" \
			--target-dir "$COVERAGE_TARGET_DIR" \
			--M $M \
			--n-reruns 100 \
			--warm-start
	done;

	# For these big models, we care only about the coverage regarding the
	# summaries of interest, so we compute their confidence intervals here:
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
