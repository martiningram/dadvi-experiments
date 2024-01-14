import os
from utils import fetch_tennis_model
from config import SACKMANN_DIR
import pickle
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.jax_api import DADVIResult
import numpy as np
from glob import glob
from tqdm import tqdm
from functools import partial
import pandas as pd
from argparse import ArgumentParser
from coverage_helpers import add_columns, save_dfs_by_M

parser = ArgumentParser()
parser.add_argument('--coverage-base-dir', required=True)
args = parser.parse_args()

tennis_model = fetch_tennis_model(1969, sackmann_dir=SACKMANN_DIR)
model = tennis_model["model"]
jax_funs = get_jax_functions_from_pymc(model)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
encoder = tennis_model["encoder"]

reruns = glob(
    os.path.join(args.coverage_base_dir, '*', 'tennis', '*.pkl')
)

n_pairs = 20

np.random.seed(2)
p1_choices = np.random.choice(encoder.classes_, size=n_pairs, replace=False)
p2_choices = np.random.choice(
    [x for x in encoder.classes_ if x not in p1_choices], size=n_pairs, replace=False
)


def build_dadvi_res(pickle_file):
    tennis_res = pickle.load(open(pickle_file, "rb"))

    dadvi_res = DADVIResult(
        tennis_res["z"],
        tennis_res["opt_result"]["opt_result"].x,
        jax_funs["unflatten_fun"],
        dadvi_funs=dadvi_funs,
    )

    return tennis_res, dadvi_res


def skill_difference(params, p1_id, p2_id):
    return params["player_skills"][p1_id] - params["player_skills"][p2_id]


def compute_quantities(tennis_res, dadvi_res):
    pairs = [*zip(p1_choices, p2_choices)]

    results = list()

    for cur_p1, cur_p2 in pairs:

        print(cur_p1, cur_p2)

        p1_id, p2_id = encoder.transform([cur_p1, cur_p2])

        lrvb_differences = (
            dadvi_res.get_frequentist_sd_and_lrvb_correction_of_scalar_valued_function(
                partial(skill_difference, p1_id=p1_id, p2_id=p2_id)
            )
        )

        results.append(lrvb_differences)

    means = np.array([x["mean"] for x in results])
    freq_sds = np.array([x["freq_sd"] for x in results])
    seed = tennis_res["seed"]
    newton_step_norm = tennis_res["opt_result"]["newton_step_norm"]
    M = tennis_res["z"].shape[0]
    scipy_opt_result = tennis_res["opt_result"]["opt_result"]

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

for i, cur_rerun in tqdm(enumerate(reruns)):
    print(f'Rerun {i}')
    tennis_res, dadvi_res = build_dadvi_res(cur_rerun)
    quantities = compute_quantities(tennis_res, dadvi_res)
    quantities["filename"] = cur_rerun

    full_results.append(quantities)

result = pd.DataFrame(full_results)

# Add naming
names = ['match_predictions' for _ in range(n_pairs)]
indices = list(range(n_pairs))

names_repeated = [names for _ in range(result.shape[0])]
indices_repeated = [indices for _ in range(result.shape[0])]

result['names'] = names_repeated
result['indices'] = indices_repeated

result = add_columns(result)

save_dfs_by_M(result, 'tennis', args.coverage_base_dir)
