import numpy as np
import pickle
import os
import json
from time import time
from glob import glob
from jax.config import config
from jax import vmap
from collections import defaultdict
from jax.scipy.special import expit

config.update("jax_enable_x64", True)

from utils import get_potus_model
from config import POTUS_JSON_PATH
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.pymc.jax_api import DADVIResult
from utils import get_run_datetime_and_hostname


def compute_distribution_from_draws(draws, function):
    chain_dim = draws[list(draws.keys())[0]].shape[:2]

    # Ditch the chain dim
    draws_flat = {x: y.reshape(-1, *y.shape[2:]) for x, y in draws.items()}

    results = vmap(function)(draws_flat)

    results = results.reshape(chain_dim[0], -1)

    return results


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--experiment-base-dir', required=True, type=str,
                        help='Directory containing the experimental results')
    args = parser.parse_args()

    EXPERIMENT_BASE_DIR = args.experiment_base_dir
    POTUS_DADVI_PATH = os.path.join(EXPERIMENT_BASE_DIR, 'dadvi_results', 'dadvi_info', 'potus.pkl')

    # Load occ_det results
    occ_res = pickle.load(open(POTUS_DADVI_PATH, "rb"))
    model = get_potus_model(POTUS_JSON_PATH)
    potus_data = json.load(open(POTUS_JSON_PATH))
    np_data = {x: np.squeeze(np.array(y)) for x, y in potus_data.items()}

    national_cov_matrix_error_sd = np.sqrt(
        np.squeeze(
            np_data["state_weights"].reshape(1, -1)
            @ (np_data["state_covariance_0"] @ np_data["state_weights"].reshape(-1, 1))
        )
    )
    ss_cov_mu_b_T = (
        np_data["state_covariance_0"]
        * (np_data["mu_b_T_scale"] / national_cov_matrix_error_sd) ** 2
    )
    cholesky_ss_cov_mu_b_T = np.linalg.cholesky(ss_cov_mu_b_T)


    def compute_final_vote_share(params):
        rel_means = params["raw_mu_b_T"]

        mean_shares = cholesky_ss_cov_mu_b_T @ rel_means + np_data["mu_b_prior"]

        return expit(mean_shares) @ np_data["state_weights"]


    # Get the JAX functions
    jax_funs = get_jax_functions_from_pymc(model)
    dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])

    dadvi_res = DADVIResult(
        occ_res["fixed_draws"],
        occ_res["opt_result"]["opt_result"].x,
        jax_funs["unflatten_fun"],
        dadvi_funs=dadvi_funs,
    )

    draw_files = glob(f"{EXPERIMENT_BASE_DIR}/*/draw_dicts/potus.npz")

    full_res = defaultdict(list)

    total_runtime = 0.0
    total_hvp = 0

    cur_fun = compute_final_vote_share

    cur_start_time = time()
    cur_res = dadvi_res.get_frequentist_sd_and_lrvb_correction_of_scalar_valued_function(
        cur_fun
    )
    cur_end_time = time()
    total_runtime += cur_end_time - cur_start_time
    total_hvp += cur_res["n_hvp_calls"]

    cur_draws = np.random.normal(loc=cur_res["mean"], scale=cur_res["lrvb_sd"], size=1000)

    full_res["lrvb_cg"].append(cur_draws.reshape(1, -1))

    for cur_file in draw_files:
        short_name = "_".join(cur_file.split("/")[-3].split("_")[:-1])

        # Skip if lrvb_cg -- recomputing
        if short_name == "lrvb_cg":
            continue

        cur_loaded = dict(np.load(cur_file))
        cur_result = compute_distribution_from_draws(cur_loaded, cur_fun)
        full_res[short_name].append(cur_result)

    full_res = {x: np.stack(y, axis=-1) for x, y in full_res.items()}

    # Update the npzs with draws
    for cur_file in draw_files:
        short_name = "_".join(cur_file.split("/")[-3].split("_")[:-1])

        if short_name == "lrvb_cg":
            continue

        cur_loaded = dict(np.load(cur_file))
        cur_result = full_res[short_name]
        cur_loaded["final_vote_share"] = cur_result
        np.savez(cur_file, **cur_loaded)

    target_folder = f"{EXPERIMENT_BASE_DIR}/lrvb_cg_results/draw_dicts/"

    os.makedirs(target_folder, exist_ok=True)
    np.savez(os.path.join(target_folder, "potus.npz"), final_vote_share=full_res["lrvb_cg"])

    # Save the runtime etc also
    runtime_cost = {
        "lrvb_hvp_calls": total_hvp,
        "lrvb_runtime": total_runtime,
        **get_run_datetime_and_hostname(),
    }

    target_folder = f"{EXPERIMENT_BASE_DIR}/lrvb_cg_results/lrvb_cg_info/"

    os.makedirs(target_folder, exist_ok=True)

    target_file = os.path.join(target_folder, f"potus.pkl")

    with open(target_file, "wb") as f:
        pickle.dump(runtime_cost, f)
