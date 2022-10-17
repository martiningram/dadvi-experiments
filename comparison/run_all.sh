# MODEL_NAME=potus

# for MODEL_NAME in occ_det potus tennis microcredit; do
for MODEL_NAME in tennis; do

	# TODO: Add SADVI, full rank SADVI
	# python fit_raabbvi.py "$MODEL_NAME"
	# python fit_dadvi.py "$MODEL_NAME"
	# python fit_pymc_sadvi.py "$MODEL_NAME" advi
	# python fit_pymc_sadvi.py "$MODEL_NAME" fullrank_advi
	# python fit_mcmc.py "$MODEL_NAME"
	# python fit_dadvi_lrvb.py "$MODEL_NAME"
	python fit_doubling_dadvi_lrvb.py "$MODEL_NAME"

done
