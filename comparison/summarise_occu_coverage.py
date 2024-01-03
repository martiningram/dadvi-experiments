import os
import pickle
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.jax_api import DADVIResult
import numpy as np
from glob import glob
from tqdm import tqdm
from functools import partial
import pandas as pd
from utils import get_occ_det_model_from_pickle
from config import OCC_DET_PICKLE_PATH
from coverage_helpers import add_columns, save_dfs_by_M
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument('--coverage-base-dir', required=True)
args = parser.parse_args()

model = get_occ_det_model_from_pickle(OCC_DET_PICKLE_PATH)
occ_pickle = pickle.load(open(OCC_DET_PICKLE_PATH, "rb"))

# Get the JAX functions
jax_funs = get_jax_functions_from_pymc(model)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])

reruns = glob(
    os.path.join(args.coverage_base_dir, '*', 'occ_det', '*.pkl')
)


def pres_prob(params, sample_loc, species_id):
    logit_prob = (
        sample_loc @ params["w_env"][species_id] + params["intercept"][species_id][0]
    )

    return logit_prob


# Load occ_det results
occ_pickle = pickle.load(open(OCC_DET_PICKLE_PATH, "rb"))

n_species = 20

np.random.seed(2)
species_chosen = np.random.choice(occ_pickle["y_df"].shape[1], size=n_species, replace=False)

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
    print(cur_rerun)

    split_path = cur_rerun.split('/')
    m_num = split_path[-3]
    rerun_num = split_path[-1].split('.')[0]

    occu_res, dadvi_res = build_dadvi_res(cur_rerun)

    if occu_res['z'].shape[0] < 10:
        continue

    try:
        quantities = compute_quantities(occu_res, dadvi_res)
    except Exception as e:
        print(e)
        print(f'Failed to compute {cur_rerun}. Skipping.')
        continue

    quantities["filename"] = cur_rerun

    full_results.append(quantities)

result = pd.DataFrame(full_results)

# Add naming
names = ['presence_prediction' for _ in range(n_species)]
indices = list(range(n_species))

names_repeated = [names for _ in range(result.shape[0])]
indices_repeated = [indices for _ in range(result.shape[0])]

result['names'] = names_repeated
result['indices'] = indices_repeated

result = add_columns(result)

save_dfs_by_M(result, 'occ_det', args.coverage_base_dir)
