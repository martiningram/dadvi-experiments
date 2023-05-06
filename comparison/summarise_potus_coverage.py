import os
import pickle
import json
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.jax_api import DADVIResult
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
from jax.scipy.special import expit
import pandas as pd
from utils import get_potus_model
from config import POTUS_JSON_PATH


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

# Get the JAX functions
jax_funs = get_jax_functions_from_pymc(model)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])

reruns = glob(
     "/Users/martin.ingram/Projects/PhD/dadvi_experiments/comparison/big_model_coverage/march_2023_coverage/*/potus/*.pkl"
     # "/home/martin.ingram/experiment_runs/march_2023_coverage/*/potus/*.pkl"
)

target_dir = "./big_model_coverage/summaries/potus/"
# target_dir = "/home/martin.ingram/experiment_runs/coverage_summaries_big_models"


def compute_final_vote_share(params):
    rel_means = params["raw_mu_b_T"]

    mean_shares = cholesky_ss_cov_mu_b_T @ rel_means + np_data["mu_b_prior"]

    return expit(mean_shares) @ np_data["state_weights"]


def build_dadvi_res(pickle_file):
    potus_res = pickle.load(open(pickle_file, "rb"))

    dadvi_res = DADVIResult(
        potus_res["z"],
        potus_res["opt_result"]["opt_result"].x,
        jax_funs["unflatten_fun"],
        dadvi_funs=dadvi_funs,
    )

    return potus_res, dadvi_res


def compute_quantities(occu_res, dadvi_res):
    results = list()
    cur_fun = compute_final_vote_share

    lrvb_res = (
        dadvi_res.get_frequentist_sd_and_lrvb_correction_of_scalar_valued_function(
            cur_fun
        )
    )

    results.append(lrvb_res)

    means = np.array([x["mean"] for x in results])
    freq_sds = np.array([x["freq_sd"] for x in results])
    seed = occu_res["seed"]
    newton_step_norm = occu_res["opt_result"]["newton_step_norm"]
    M = occu_res["z"].shape[0]
    scipy_opt_result = occu_res["opt_result"]["opt_result"]

    result_series = pd.Series(
        {
            "means": means,
            "seed": seed,
            "freq_sds": freq_sds,
            "newton_step_norm": newton_step_norm,
            "scipy_opt_result": scipy_opt_result,
            "M": M,
        }
    )

    return result_series


os.makedirs(target_dir, exist_ok=True)

for cur_rerun in tqdm(reruns):
    print(cur_rerun)
    potus_res, dadvi_res = build_dadvi_res(cur_rerun)

    split_path = cur_rerun.split('/')
    m_num = split_path[-3]
    rerun_num = split_path[-1].split('.')[0]
    target_file = os.path.join(target_dir, f'{m_num}_{rerun_num}.pkl')

    if os.path.isfile(target_file):
        print('Already exists; skipping.')
        continue

    if potus_res['z'].shape[0] < 10:
        continue

    try:
        quantities = compute_quantities(potus_res, dadvi_res)
    except Exception as e:
        print(e)
        print(f'Failed to compute {cur_rerun}. Skipping.')
        continue
    quantities["filename"] = cur_rerun


    pickle.dump(quantities, open(target_file, 'wb'))
