import os
import pickle
import numpy as np
from functools import partial
from collections import defaultdict
from glob import glob
from time import time

from utils import fetch_tennis_model
from config import SACKMANN_DIR
from dadvi.pymc.jax_api import DADVIResult
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.jax import build_dadvi_funs
from jax import vmap
from utils import get_run_datetime_and_hostname


### TENNIS
def skill_difference(params, p1_id, p2_id):

    return params["player_skills"][p1_id] - params["player_skills"][p2_id]


def compute_distribution_from_draws(draws, function):

    # Ditch the chain dim
    draws_flat = {x: y.reshape(-1, *y.shape[2:]) for x, y in draws.items()}

    results = vmap(function)(draws_flat)

    return results


if __name__ == '__main__':

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--experiment-base-dir', required=True, type=str,
                        help='Directory containing the experimental results')
    args = parser.parse_args()

    EXPERIMENT_BASE_DIR = args.experiment_base_dir
    TENNIS_PICKLE_FILE = os.path.join(EXPERIMENT_BASE_DIR, 'dadvi_results', 'dadvi_info', 'tennis.pkl')

    tennis_res = pickle.load(open(TENNIS_PICKLE_FILE, "rb"))
    tennis_model = fetch_tennis_model(1969, sackmann_dir=SACKMANN_DIR)
    model = tennis_model["model"]
    jax_funs = get_jax_functions_from_pymc(model)
    dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
    encoder = tennis_model["encoder"]

    dadvi_res = DADVIResult(
        tennis_res["fixed_draws"],
        tennis_res["opt_result"]["opt_result"].x,
        jax_funs["unflatten_fun"],
        dadvi_funs=dadvi_funs,
    )

    np.random.seed(2)
    p1_choices = np.random.choice(encoder.classes_, size=20, replace=False)
    p2_choices = np.random.choice(
        [x for x in encoder.classes_ if x not in p1_choices], size=20, replace=False
    )


    draw_files = glob(
       f"{EXPERIMENT_BASE_DIR}/*/draw_dicts/tennis.npz"
    )

    pairs = [*zip(p1_choices, p2_choices)]

    lrvb_results = dict()
    draw_results = defaultdict(dict)

    total_runtime = 0.
    total_hvp = 0

    # For DADVI, we use the LRVB correction using the delta method
    for cur_p1, cur_p2 in pairs:

        p1_id, p2_id = encoder.transform([cur_p1, cur_p2])

        cur_start_time = time()
        lrvb_differences = (
            dadvi_res.get_frequentist_sd_and_lrvb_correction_of_scalar_valued_function(
                partial(skill_difference, p1_id=p1_id, p2_id=p2_id)
            )
        )
        cur_end_time = time()
        total_runtime += (cur_end_time - cur_start_time)
        total_hvp += lrvb_differences['n_hvp_calls']

        lrvb_results[f"{cur_p1} vs {cur_p2}"] = lrvb_differences

        # Add the chain dimension on
        draw_results[f"{cur_p1} vs {cur_p2}"]["lrvb_cg"] = np.random.normal(
            loc=lrvb_differences["mean"], scale=lrvb_differences["lrvb_sd"], size=1000
        ).reshape(1, -1)

    # For the others, we just compute using the draws
    for cur_p1, cur_p2 in pairs:

        p1_id, p2_id = encoder.transform([cur_p1, cur_p2])

        for cur_file in draw_files:

            short_name = "_".join(cur_file.split("/")[-3].split("_")[:-1])

            if short_name == 'lrvb_cg':
                continue

            cur_loaded = np.load(cur_file)

            chain_dim = cur_loaded[list(cur_loaded.keys())[0]].shape[0]
            cur_result = compute_distribution_from_draws(
                cur_loaded, partial(skill_difference, p1_id=p1_id, p2_id=p2_id)
            )

            draw_results[f"{cur_p1} vs {cur_p2}"][short_name] = cur_result.reshape(
                chain_dim, -1
            )

    # Now we have all the draw results together. Add them to the npzs.
    # Maybe let's just flatten them to make it easier.
    methods_stacked = defaultdict(list)

    for cur_matchup, cur_method_lookup in draw_results.items():
        for cur_method, cur_draws in cur_method_lookup.items():
            methods_stacked[cur_method].append(cur_draws)

    methods_stacked = {x: np.stack(y, axis=-1) for x, y in methods_stacked.items()}

    # Add these to the draw dicts
    for cur_file in draw_files:

        if short_name == 'lrvb_cg':
            continue

        cur_loaded = dict(np.load(cur_file))
        short_name = "_".join(cur_file.split("/")[-3].split("_")[:-1])
        cur_loaded["match_predictions"] = methods_stacked[short_name]

        # Update with new stuff
        np.savez(cur_file, **cur_loaded)

    # Make a final one for lrvb_cg
    target_dir = (
        f"{EXPERIMENT_BASE_DIR}/lrvb_cg_results/draw_dicts/"
    )

    os.makedirs(target_dir, exist_ok=True)

    np.savez(
        os.path.join(target_dir, "tennis.npz"), match_predictions=methods_stacked["lrvb_cg"]
    )

    # Save the runtime etc also
    runtime_cost = {
            'lrvb_hvp_calls': total_hvp,
            'lrvb_runtime': total_runtime,
            **get_run_datetime_and_hostname()
    }

    target_folder = (
        f"{EXPERIMENT_BASE_DIR}/lrvb_cg_results/lrvb_cg_info/"
    )

    os.makedirs(target_folder, exist_ok=True)

    target_file = os.path.join(target_folder, f"tennis.pkl")

    with open(target_file, "wb") as f:
        pickle.dump(runtime_cost, f)
