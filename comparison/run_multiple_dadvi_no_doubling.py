from jax.config import config

config.update("jax_enable_x64", True)
import os
import pickle

from utils import load_model_by_name
from dadvi.jax import build_dadvi_funs
from dadvi.core import find_dadvi_optimum
from dadvi.pymc.pymc_to_jax import (
    get_jax_functions_from_pymc,
    get_flattened_indices_and_param_names,
)
import numpy as np
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--model-name", required=True)
parser.add_argument("--M", required=True, type=int)
parser.add_argument("--target-dir", required=True, type=str)
parser.add_argument("--n-reruns", required=False, type=int, default=100)
parser.add_argument("--warm-start", required=False, action="store_true")
args = parser.parse_args()

model_name = args.model_name
n_reruns = args.n_reruns
M = args.M

m = load_model_by_name(model_name)

base_target_dir = os.path.join(args.target_dir, f"M_{M}")
target_dir = os.path.join(base_target_dir, model_name)
os.makedirs(target_dir, exist_ok=True)

jax_funs = get_jax_functions_from_pymc(m)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
init_means = np.zeros(jax_funs["n_params"])
init_log_vars = np.zeros(jax_funs["n_params"]) - 3
init_var_params = np.concatenate([init_means, init_log_vars])

flat_param_names, flat_param_indices = get_flattened_indices_and_param_names(jax_funs)

np.random.seed(2)
zs = np.random.randn(M, jax_funs["n_params"])

# For the first run, we use doubling to pick an M
opt_result = find_dadvi_optimum(
    init_var_params,
    zs,
    dadvi_funs,
)

# Get the results from our reference run:
reference_results = {"seed": 2, "opt_result": opt_result, "z": zs}

with open(os.path.join(target_dir, "reference.pkl"), "wb") as f:
    pickle.dump(reference_results, f)

completed_runs = 0
attempts = 0

while completed_runs < n_reruns:
    print(f"On {completed_runs} of {n_reruns}")

    cur_seed = 1000 + attempts
    np.random.seed(cur_seed)
    cur_z = np.random.randn(M, jax_funs["n_params"])
    attempts += 1

    if args.warm_start:
        rerun_var_params = reference_results["opt_result"]["opt_result"].x
    else:
        rerun_var_params = init_var_params

    try:
        result = find_dadvi_optimum(rerun_var_params, cur_z, dadvi_funs=dadvi_funs)
    except Exception as e:
        print(f"Optimisation failed with error: {e}")
        print(f"Retrying with new seed.")

    rerun_results = {
        "seed": cur_seed,
        "opt_result": result,
        "z": cur_z,
    }

    with open(os.path.join(target_dir, f"{completed_runs}.pkl"), "wb") as f:
        pickle.dump(rerun_results, f)

    completed_runs += 1
