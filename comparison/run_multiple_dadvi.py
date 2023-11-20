from jax.config import config

config.update("jax_enable_x64", True)
"""
Run DADVI multiple times to investigate coverage.
TODO: Consider moving into a separate folder (but then need to make utils available also)
"""
import os
import sys

from utils import load_model_by_name
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.pymc_to_jax import (
    get_jax_functions_from_pymc,
    get_flattened_indices_and_param_names,
)
from dadvi.doubling_dadvi import (
    optimise_dadvi_by_doubling,
    fit_dadvi_and_estimate_covariances,
)
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from utils import get_run_datetime_and_hostname

parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--target-dir", required=True, type=str)
parser.add_argument("--min-m-power", required=False, type=int, default=6)
parser.add_argument("--n-reruns", required=False, type=int, default=100)
parser.add_argument("--warm-start", required=False, action="store_true")
args = parser.parse_args()

model_name = args.model_name
min_m_power = args.min_m_power
n_reruns = args.n_reruns

m = load_model_by_name(model_name)

base_target_dir = os.path.join(args.target_dir, f"M_{2**min_m_power}")
target_dir = os.path.join(base_target_dir, model_name)
os.makedirs(target_dir, exist_ok=True)

jax_funs = get_jax_functions_from_pymc(m)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
init_means = np.zeros(jax_funs["n_params"])
init_log_vars = np.zeros(jax_funs["n_params"]) - 3
init_var_params = np.concatenate([init_means, init_log_vars])

flat_indices, flat_param_names = get_flattened_indices_and_param_names(jax_funs)

# TODO: We don't double any more -- so maybe we shouldn't be calling this
opt_result = optimise_dadvi_by_doubling(
    init_var_params,
    dadvi_funs,
    seed=2,
    verbose=True,
    start_m=2**min_m_power,
    max_m=2**min_m_power,  # No doubling!
    max_freq_to_posterior_ratio=0.5,
)

# Pick the last one
opt_result = opt_result[max(opt_result.keys())]

# Get the results from our reference run:
opt = opt_result["dadvi_result"]["optimisation_result"]
dadvi_res = opt["opt_result"].x
zs = opt_result["zs"]
freq_sds = opt_result["dadvi_result"]["frequentist_mean_sds"]
means = np.split(dadvi_res, 2)[0]
m_picked = zs.shape[0]

reference_results = {
    "means": means,
    "freq_sds": freq_sds,
    "m_picked": m_picked,
    "newton_step_norm": opt["newton_step_norm"],
    "var_params": dadvi_res,
    "scipy_opt_result": opt["opt_result"],
}

rerun_results = list()

for cur_run in range(n_reruns):
    print(f"On {cur_run} of {n_reruns}")

    cur_seed = 1000 + cur_run
    np.random.seed(cur_seed)
    cur_z = np.random.randn(m_picked, means.shape[0])

    if args.warm_start:
        rerun_var_params = reference_results["var_params"]
    else:
        rerun_var_params = init_var_params

    result = fit_dadvi_and_estimate_covariances(
        rerun_var_params, cur_z, dadvi_funs=dadvi_funs
    )

    freq_sds_rerun = result["frequentist_mean_sds"]

    opt = result["optimisation_result"]

    rerun_means = np.split(opt["opt_result"].x, 2)[0]

    newton_step_norm = opt["newton_step_norm"]

    rerun_results.append(
        {
            "means": rerun_means,
            "means_with_names": jax_funs["unflatten_fun"](rerun_means),
            "seed": cur_seed,
            "freq_sds": freq_sds_rerun,
            "freq_sds_with_names": jax_funs["unflatten_fun"](freq_sds),
            "newton_step_norm": newton_step_norm,
            "scipy_opt_result": opt["opt_result"],
            "lrvb_hvp_calls": result["lrvb_hvp_calls"],
            "lrvb_freq_cov_grad_calls": result["lrvb_freq_cov_grad_calls"],
            "names": flat_param_names,
            "indices": flat_indices,
            **get_run_datetime_and_hostname(),
        }
    )

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
rerun_df["reference_newton_step_norm"] = reference_results["newton_step_norm"]
rerun_df["reference_scipy_opt_result"] = [
    reference_results["scipy_opt_result"]
] * rerun_df.shape[0]

rerun_df.to_pickle(os.path.join(target_dir, "coverage_results.pkl"))
