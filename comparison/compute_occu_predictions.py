import os
from functools import partial
from glob import glob
from jax.config import config
from jax import vmap
from collections import defaultdict

config.update("jax_enable_x64", True)

import numpy as np
import pickle
from utils import get_occ_det_model_from_pickle
from config import OCC_DET_PICKLE_PATH
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.pymc.jax_api import DADVIResult


def pres_prob(params, sample_loc, species_id):

    logit_prob = (
        sample_loc @ params["w_env"][species_id] + params["intercept"][species_id][0]
    )

    return logit_prob


def compute_distribution_from_draws(draws, function):

    chain_dim = draws[list(draws.keys())[0]].shape[:2]

    # Ditch the chain dim
    draws_flat = {x: y.reshape(-1, *y.shape[2:]) for x, y in draws.items()}

    results = vmap(function)(draws_flat)

    results = results.reshape(chain_dim[0], -1)

    return results


OCCU_DADVI_PATH = "/home/martin.ingram/experiment_runs/march_2023/dadvi_results/dadvi_info/occ_det.pkl"
EXPERIMENT_BASE_DIR = '/home/martin.ingram/experiment_runs/march_2023'

# Load occ_det results
occ_res = pickle.load(open(OCCU_DADVI_PATH, "rb"))
model = get_occ_det_model_from_pickle(OCC_DET_PICKLE_PATH)
occ_pickle = pickle.load(open(OCC_DET_PICKLE_PATH, "rb"))

# Get the JAX functions
jax_funs = get_jax_functions_from_pymc(model)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])

dadvi_res = DADVIResult(
    occ_res["fixed_draws"],
    occ_res["opt_result"]["opt_result"].x,
    jax_funs["unflatten_fun"],
    dadvi_funs=dadvi_funs,
)

np.random.seed(2)
species_chosen = np.random.choice(occ_pickle["y_df"].shape[1], size=20, replace=False)

sample_loc = occ_pickle["X_env_mat"][200]

draw_files = glob(
   f"{EXPERIMENT_BASE_DIR}/*/draw_dicts/occ_det.npz"
)

full_res = defaultdict(list)

for species_id in species_chosen:

    print(species_id)

    cur_fun = partial(pres_prob, species_id=species_id, sample_loc=sample_loc)

    cur_res = (
        dadvi_res.get_frequentist_sd_and_lrvb_correction_of_scalar_valued_function(
            cur_fun
        )
    )

    cur_draws = np.random.normal(
        loc=cur_res["mean"], scale=cur_res["lrvb_sd"], size=1000
    )

    full_res["lrvb_cg"].append(cur_draws.reshape(1, -1))

    for cur_file in draw_files:

        short_name = "_".join(cur_file.split("/")[-3].split("_")[:-1])
        cur_loaded = dict(np.load(cur_file))
        cur_result = compute_distribution_from_draws(cur_loaded, cur_fun)
        full_res[short_name].append(cur_result)

full_res = {x: np.stack(y, axis=-1) for x, y in full_res.items()}

# Update the npzs with draws
for cur_file in draw_files:

    short_name = "_".join(cur_file.split("/")[-3].split("_")[:-1])
    cur_loaded = dict(np.load(cur_file))
    cur_result = full_res[short_name]
    cur_loaded["presence_prediction"] = cur_result
    np.savez(cur_file, **cur_loaded)

# TODO: Need timing summaries also
# Also make one for LRVB_CG containing only this field
target_folder = (
    f"{EXPERIMENT_BASE_DIR}/lrvb_cg_results/draw_dicts/"
)

os.makedirs(target_folder, exist_ok=True)
np.savez(
    os.path.join(target_folder, "occ_det.npz"), presence_prediction=full_res["lrvb_cg"]
)
