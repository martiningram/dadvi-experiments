TARGET_DIR='./'

while IFS= read -r MODEL_NAME; do
    echo "$MODEL_NAME"

    # Run inference
    # python fit_pymc_sadvi.py "$MODEL_NAME" advi
    # python fit_pymc_sadvi.py "$MODEL_NAME" fullrank_advi
    # python fit_raabbvi.py "$MODEL_NAME"
    # python fit_dadvi.py "$MODEL_NAME"
    # python fit_mcmc.py "$MODEL_NAME"
    python fit_dadvi_lrvb.py "$MODEL_NAME" "$TARGET_DIR" CG
    # python fit_doubling_dadvi_lrvb.py "$MODEL_NAME"

    # Run coverage
    python run_multiple_dadvi.py "$MODEL_NAME" 4
    python run_multiple_dadvi.py "$MODEL_NAME" 5
    python run_multiple_dadvi.py "$MODEL_NAME" 6

done < all_arm_names.txt


