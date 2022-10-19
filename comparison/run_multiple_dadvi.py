"""
Run DADVI multiple times to investigate coverage.
TODO: Consider moving into a separate folder (but then need to make utils available also)
"""
import os
import sys

from utils import load_model_by_name
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.doubling_dadvi import (
    optimise_dadvi_by_doubling,
    fit_dadvi_and_estimate_covariances
)
from dadvi.core import find_dadvi_optimum
import numpy as np
import pandas as pd

n_reruns = 100

model_name = sys.argv[1]
m = load_model_by_name(model_name)

base_target_dir = '/media/martin/External Drive/projects/lrvb_paper/coverage_redone_m_64'
target_dir = os.path.join(base_target_dir, model_name)
os.makedirs(target_dir, exist_ok=True)

jax_funs = get_jax_functions_from_pymc(m)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
init_means = np.zeros(jax_funs["n_params"])
init_log_vars = np.zeros(jax_funs["n_params"]) - 3
init_var_params = np.concatenate([init_means, init_log_vars])

# For the first run, we use doubling to pick an M
opt_result = optimise_dadvi_by_doubling(
    init_var_params,
    dadvi_funs,
    seed=2,
    verbose=True,
    start_m_power=6,
    max_freq_to_posterior_ratio=0.5,
)

# Get the results from our reference run:
opt = opt_result["dadvi_result"]["optimisation_result"]
dadvi_res = opt["opt_result"].x
zs = opt_result["zs"]
freq_sds = opt_result["dadvi_result"]["frequentist_mean_sds"]
means = np.split(dadvi_res, 2)[0]
m_picked = zs.shape[0]

reference_results = {"means": means, "freq_sds": freq_sds, "m_picked": m_picked}

rerun_results = list()

for cur_run in range(n_reruns):

    cur_seed = 1000 + cur_run
    np.random.seed(cur_seed)
    cur_z = np.random.randn(m_picked, means.shape[0])

    result = fit_dadvi_and_estimate_covariances(init_var_params, cur_z, dadvi_funs=dadvi_funs)

    freq_sds_rerun = result['frequentist_mean_sds']

    opt = result['optimisation_result']

    rerun_means = np.split(opt["opt_result"].x, 2)[0]

    rerun_results.append({"means": rerun_means, "seed": cur_seed, 'freq_sds': freq_sds_rerun})

rerun_df = pd.DataFrame(rerun_results)

rerun_df["reference_means"] = np.tile(
    reference_results["means"], (n_reruns, 1)
).tolist()
rerun_df["reference_freq_sds"] = np.tile(
    reference_results["freq_sds"], (n_reruns, 1)
).tolist()

# Turn them into arrays
rerun_df["reference_means"] = rerun_df["reference_means"].apply(np.array)
rerun_df["reference_freq_sds"] = rerun_df["reference_freq_sds"].apply(np.array)

rerun_df["M"] = reference_results["m_picked"]

rerun_df.to_pickle(os.path.join(target_dir, "coverage_results.pkl"))
