# MODEL_NAME=potus

for MODEL_NAME in microcredit occ_det tennis potus; do

	# python fit_raabbvi.py "$MODEL_NAME"
	# python fit_dadvi.py "$MODEL_NAME"
	# python fit_pymc_sadvi.py "$MODEL_NAME" advi
	# python fit_pymc_sadvi.py "$MODEL_NAME" fullrank_advi
	# python fit_mcmc.py "$MODEL_NAME"
	# python fit_dadvi_lrvb.py "$MODEL_NAME"
	# python fit_doubling_dadvi_lrvb.py "$MODEL_NAME"

        # Run coverage
        python run_multiple_dadvi.py "$MODEL_NAME"

done
