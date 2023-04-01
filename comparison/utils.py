import numpy as np
import pandas as pd
from config import (
    ARM_CONFIG_CSV_PATH,
    ARM_JSON_PATH,
    MICROCREDIT_JSON_PATH,
    OCC_DET_PICKLE_PATH,
    POTUS_JSON_PATH,
    SACKMANN_DIR,
)
from dadvi.pymc.models.arm import load_arm_model, add_bambi_family
from dadvi.pymc.models.microcredit import load_microcredit_model
from dadvi.pymc.models.occ_det import get_occ_det_model_from_pickle
from dadvi.pymc.models.potus import get_potus_model
from dadvi.pymc.models.tennis import fetch_tennis_model
# from jqas_tennis_model import load_model_and_data
import pymc as pm
from jax.flatten_util import ravel_pytree
from datetime import datetime
import socket
from time import time


NON_ARM_MODELS = ["microcredit", "occ_det", "potus", "tennis", "tennis-jqas"]


def load_model_by_name(model_name):

    if model_name == "microcredit":
        model = load_microcredit_model(MICROCREDIT_JSON_PATH)

    elif model_name == "occ_det":
        model = get_occ_det_model_from_pickle(OCC_DET_PICKLE_PATH)

    elif model_name == "potus":
        model = get_potus_model(POTUS_JSON_PATH)

    elif model_name == "tennis":
        model = fetch_tennis_model(1969, sackmann_dir=SACKMANN_DIR)["model"]

 #    elif model_name == "tennis-jqas":
 #        model = load_model_and_data(SACKMANN_DIR)

    else:
        df = pd.read_csv(ARM_CONFIG_CSV_PATH)
        df = add_bambi_family(df)

        # Should check for duplicates here
        rel_row = df[df["model_name"] == model_name].iloc[0]

        model = load_arm_model(rel_row, ARM_JSON_PATH)["pymc_model"]

    return model


def estimate_kl_fresh_draws(dadvi_funs, var_params, n_draws=1000, seed=None):

    n_params = len(var_params) // 2

    if seed is not None:
        np.random.seed(seed)

    cur_z = np.random.randn(n_draws, n_params)
    return dadvi_funs.kl_est_and_grad_fun(var_params, cur_z)[0]


def estimate_kl_stderr_fresh_draws(dadvi_funs, var_params, n_draws=1000, seed=None):

    n_params = len(var_params) // 2

    if seed is not None:
        np.random.seed(seed)

    cur_z = np.random.randn(n_draws, n_params)

    individual_results = [
        dadvi_funs.kl_est_and_grad_fun(var_params, cur_z[[i]])[0]
        for i in range(n_draws)
    ]

    stdev = np.std(individual_results)

    return stdev / np.sqrt(n_draws)


def arviz_to_draw_dict(az_trace):
    # Converts an arviz trace to a dict of str -> np.ndarray

    dict_version = dict(az_trace.posterior.data_vars.variables)

    return {x: y.values for x, y in dict_version.items()}


def flat_results_to_dict(flat_data, advi_fit_result):
    # Turns a flat vector of PyMC ADVI results into a dictionary.
    # Adapted from: https://github.com/pymc-devs/pymc/pull/6387

    result = dict()
    for name, s, shape, dtype in advi_fit_result.ordering.values():
        values = np.array(flat_data[s]).reshape(shape).astype(dtype)
        result[name] = values

    return result


def flatten_shared(shared_list):
    # From https://github.com/pymc-devs/pymc/blob/main/pymc/variational/callbacks.py
    return np.concatenate([sh.get_value().flatten() for sh in shared_list])


def pymc_advi_history_callback(Approximation, losses, i, record_every=100):
    # A callback for PyMC3, recording the KL over time, estimated with n_draws.
    # logp_fn_dict_vmap must evaluate the log posterior, and must be vectorised,
    # i.e. must be able to take batches of parameters.

    if i == 1:
        # Initialise to empty list
        pymc_advi_history_callback.kl_history = list()

    if i % record_every != 1:
        return

    means = Approximation.mean.eval()
    sds = Approximation.std.eval()
    cur_time = time()

    mean_dict = flat_results_to_dict(means, Approximation)
    sd_dict = flat_results_to_dict(sds, Approximation)

    # Also store what's needed to compute the convergence criterion
    flat_params = flatten_shared(Approximation.params)

    # TODO: Use these results to compute the ELBO
    # Slight problem is that I need to make sure the ordering agrees.
    pymc_advi_history_callback.kl_history.append(
        (
            i,
            {
                "means": mean_dict,
                "sds": sd_dict,
                "time": cur_time,
                "flat_params": flat_params,
            },
        )
    )


def flatten_and_check_consistent(param_dict, jax_funs):
    """
    Flattens the dictionary and checks that the flattening can be undone properly by
    the "unflatten_fun" in the jax_funs dictionary.
    """

    flat_vec, fun = ravel_pytree(param_dict)

    v1 = jax_funs["unflatten_fun"](flat_vec)
    v2 = fun(flat_vec)

    check = np.all([(v1[x] == v2[x]).all() for x in v1])

    assert check

    return flat_vec


def estimate_kl_pymc_advi(
    mean_dict, sd_dict, dadvi_funs, jax_funs, seed=None, n_draws=1000
):
    """
    Estimates the ELBO for a PyMC ADVI set of means and standard deviations.
    """

    flat_means = flatten_and_check_consistent(mean_dict, jax_funs)
    flat_sds = flatten_and_check_consistent(sd_dict, jax_funs)

    var_params = np.concatenate([flat_means, np.log(flat_sds)])

    kl = estimate_kl_fresh_draws(dadvi_funs, var_params, seed=seed, n_draws=n_draws)

    return kl


def fit_pymc_sadvi(
    m,
    n_draws=1000,
    n_steps=100000,
    method="advi",
    convergence_crit="default",
    extra_callbacks=[],
):

    assert method in ["advi", "fullrank_advi"]

    if convergence_crit is None:
        extra_args = {"callbacks": extra_callbacks}
    elif convergence_crit == "default":
        extra_args = {
            "callbacks": [pm.callbacks.CheckParametersConvergence()] + extra_callbacks
        }
    elif convergence_crit == "absdiff":
        extra_args = {
            "callbacks": [pm.callbacks.CheckParametersConvergence(diff="absolute")]
            + extra_callbacks
        }
    else:
        assert False, "Unknown convergence criterion."

    with m as model:

        fit_res = pm.fit(method=method, n=n_steps, **extra_args)
        draws = fit_res.sample(n_draws)

    return {"draw_dict": arviz_to_draw_dict(draws), "fit_res": fit_res}


def get_run_datetime_and_hostname():

    run_datetime = datetime.now()
    hostname = socket.gethostname()

    return {"datetime": run_datetime, "hostname": hostname}
