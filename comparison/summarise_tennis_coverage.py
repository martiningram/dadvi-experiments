import os
from utils import fetch_tennis_model
from config import SACKMANN_DIR
import pickle
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.jax_api import DADVIResult
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
from functools import partial
import pandas as pd

tennis_model = fetch_tennis_model(1969, sackmann_dir=SACKMANN_DIR)
model = tennis_model["model"]
jax_funs = get_jax_functions_from_pymc(model)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
encoder = tennis_model["encoder"]

reruns = glob(
    "/Users/martin.ingram/Projects/PhD/dadvi_experiments/comparison/big_model_coverage/march_2023_coverage/*/tennis/*.pkl"
)

target_dir = "./big_model_coverage/summaries/tennis/"

np.random.seed(2)
p1_choices = np.random.choice(encoder.classes_, size=20, replace=False)
p2_choices = np.random.choice(
    [x for x in encoder.classes_ if x not in p1_choices], size=20, replace=False
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

for cur_rerun in tqdm(reruns):
    tennis_res, dadvi_res = build_dadvi_res(cur_rerun)
    quantities = compute_quantities(tennis_res, dadvi_res)
    quantities["filename"] = cur_rerun

    full_results.append(quantities)

result = pd.DataFrame(full_results)
os.makedirs(target_dir, exist_ok=True)
result.to_pickle(os.path.join(target_dir, "tennis.pkl"))
