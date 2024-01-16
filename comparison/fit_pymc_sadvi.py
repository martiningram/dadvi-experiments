import sys
import time
from os.path import join
from os import makedirs

import pymc as pm
import numpy as np
import pandas as pd
import pickle

from utils import (
    load_model_by_name,
    fit_pymc_sadvi,
    pymc_advi_history_callback,
    estimate_kl_pymc_advi,
)
from dadvi.pymc.utils import get_unconstrained_variable_names
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.jax import build_dadvi_funs
from utils import get_run_datetime_and_hostname

if __name__ == "__main__":
    from fitting_helpers import get_parser_with_default_args

    parser = get_parser_with_default_args()
    parser.add_argument("--advi-method", required=True)
    args, _ = parser.parse_known_args()

    model_name = args.model_name
    target_dir = args.target_dir
    method = args.advi_method
    test_run = args.test_run

    n_draws = 10 if test_run else 1000
    n_steps = 10 if test_run else 100000

    model = load_model_by_name(model_name)

    print("Fitting")
    start_time = time.time()

    with model as m:
        fit_result_sadvi = fit_pymc_sadvi(
            m,
            n_draws=n_draws,
            n_steps=n_draws,
            method=method,
            convergence_crit="default",
            extra_callbacks=[pymc_advi_history_callback],
        )

    end_time = time.time()
    runtime = end_time - start_time
    print("Done")

    if method == "advi":
        kl_hist = pymc_advi_history_callback.kl_history
        jax_funs = get_jax_functions_from_pymc(model)
        dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
        kl_estimates = [
            {
                "step": i,
                "kl_estimate": estimate_kl_pymc_advi(
                    cur_params["means"], cur_params["sds"], dadvi_funs, jax_funs, seed=2
                ),
                "time": cur_params["time"],
            }
            for i, cur_params in kl_hist
        ]
        kl_estimates = pd.DataFrame(kl_estimates)
    else:
        kl_estimates = None

    flat_params_over_time = {
        x[0]: x[1]["flat_params"] for x in pymc_advi_history_callback.kl_history
    }

    base_dir = join(target_dir, f"s{method}_results")

    makedirs(join(base_dir, "draw_dicts"), exist_ok=True)
    makedirs(join(base_dir, "info"), exist_ok=True)

    np.savez(
        join(base_dir, "draw_dicts", model_name + ".npz"),
        **fit_result_sadvi["draw_dict"],
    )

    with open(join(base_dir, "info", model_name + ".pkl"), "wb") as f:
        pickle.dump(
            {
                "steps": fit_result_sadvi["fit_res"].hist.shape[0],
                "runtime": runtime,
                "unconstrained_param_names": get_unconstrained_variable_names(model),
                "kl_history": kl_estimates,
                # TODO: Maybe compute the convergence criterion over time here
                "flat_params_over_time": flat_params_over_time,
                **get_run_datetime_and_hostname(),
            },
            f,
        )
