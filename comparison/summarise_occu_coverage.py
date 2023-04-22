import os
import pickle
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.jax_api import DADVIResult
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
from functools import partial
import pandas as pd
from utils import get_occ_det_model_from_pickle
from config import OCC_DET_PICKLE_PATH


model = get_occ_det_model_from_pickle(OCC_DET_PICKLE_PATH)
occ_pickle = pickle.load(open(OCC_DET_PICKLE_PATH, "rb"))

# Get the JAX functions
jax_funs = get_jax_functions_from_pymc(model)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])

reruns = glob(
    "/Users/martin.ingram/Projects/PhD/dadvi_experiments/comparison/big_model_coverage/march_2023_coverage/*/occ_det/*.pkl"
)

target_dir = "./big_model_coverage/summaries/occ_det/"


def pres_prob(params, sample_loc, species_id):
    logit_prob = (
        sample_loc @ params["w_env"][species_id] + params["intercept"][species_id][0]
    )

    return logit_prob


# Load occ_det results
occ_pickle = pickle.load(open(OCC_DET_PICKLE_PATH, "rb"))

np.random.seed(2)
species_chosen = np.random.choice(occ_pickle["y_df"].shape[1], size=20, replace=False)

sample_loc = occ_pickle["X_env_mat"][200]


def build_dadvi_res(pickle_file):
    tennis_res = pickle.load(open(pickle_file, "rb"))

    dadvi_res = DADVIResult(
        tennis_res["z"],
        tennis_res["opt_result"]["opt_result"].x,
        jax_funs["unflatten_fun"],
        dadvi_funs=dadvi_funs,
    )

    return tennis_res, dadvi_res


def compute_quantities(occu_res, dadvi_res):
    results = list()

    for species_id in species_chosen:
        cur_fun = partial(pres_prob, species_id=species_id, sample_loc=sample_loc)

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


full_results = list()

for cur_rerun in tqdm(reruns):
    occu_res, dadvi_res = build_dadvi_res(cur_rerun)
    quantities = compute_quantities(occu_res, dadvi_res)
    quantities["filename"] = cur_rerun

    full_results.append(quantities)

result = pd.DataFrame(full_results)
os.makedirs(target_dir, exist_ok=True)
result.to_pickle(os.path.join(target_dir, "occ_det.pkl"))
