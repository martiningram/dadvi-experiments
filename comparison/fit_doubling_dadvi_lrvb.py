from jax.config import config

config.update("jax_enable_x64", True)
# Fit using DADVI. This is the verbose version; we'll want a higher-level API down the road.
# It's not hard to write one, but hopefully this makes sense to you.
import sys
from utils import load_model_by_name, estimate_kl_fresh_draws
from dadvi.core import get_lrvb_draws
from dadvi.jax import build_dadvi_funs
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
import numpy as np
import time
from dadvi.utils import opt_callback_fun
from dadvi.doubling_dadvi import optimise_dadvi_by_doubling
from dadvi.pymc.pymc_to_jax import transform_dadvi_draws
from dadvi.pymc.utils import get_unconstrained_variable_names
from os import makedirs
from os.path import join
import pickle
from utils import get_run_datetime_and_hostname


def extract_metadata_dict(opt_result):

    opt = opt_result["dadvi_result"]["optimisation_result"]
    zs = opt_result["zs"]
    lrvb_cov = opt_result["dadvi_result"]["lrvb_covariance"]
    dadvi_opt_sequence = opt_result["opt_sequence"]

    kl_hist_dadvi = [
        estimate_kl_fresh_draws(dadvi_funs, cur_hist["theta"], seed=2)
        for cur_hist in dadvi_opt_sequence
    ]

    metadata = {
        "opt_result": opt,
        "fixed_draws": zs,
        "M": zs.shape[0],
        "kl_hist": kl_hist_dadvi,
        "opt_sequence": dadvi_opt_sequence,
        "lrvb_cov": lrvb_cov,
        "newton_step_norm": opt["newton_step_norm"],
        "newton_step": opt["newton_step"],
        "unconstrained_param_names": get_unconstrained_variable_names(m),
        "ratio": opt_result["ratio"],
        "ratio_is_ok": opt_result["ratio_is_ok"],
        "lrvb_hvp_count": opt_result["dadvi_result"]["lrvb_hvp_calls"],
        "lrvb_freq_cov_grad_count": opt_result["dadvi_result"][
            "lrvb_freq_cov_grad_calls"
        ],
    }

    return metadata


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("fork")

    model_name = sys.argv[1]
    target_dir = sys.argv[2]
    max_freq_to_post_ratio = float(sys.argv[3])
    m = load_model_by_name(model_name)

    # This will store the sequence of parameters
    opt_callback_fun.opt_sequence = []

    start_time = time.time()
    jax_funs = get_jax_functions_from_pymc(m)
    dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
    init_means = np.zeros(jax_funs["n_params"])
    init_log_vars = np.zeros(jax_funs["n_params"]) - 3
    init_var_params = np.concatenate([init_means, init_log_vars])
    all_opt_results = optimise_dadvi_by_doubling(
        init_var_params,
        dadvi_funs,
        seed=2,
        verbose=True,
        start_m=20,
        max_m=160,
        max_freq_to_posterior_ratio=max_freq_to_post_ratio,
        callback_fun=opt_callback_fun,
    )

    opt_result = all_opt_results[max(all_opt_results.keys())]
    opt = opt_result["dadvi_result"]["optimisation_result"]
    dadvi_res = opt["opt_result"].x
    zs = opt_result["zs"]
    lrvb_cov = opt_result["dadvi_result"]["lrvb_covariance"]

    print(opt_result["M"], opt_result["ratio"])

    finish_time = time.time()

    runtime_dadvi = finish_time - start_time
    dadvi_opt_sequence = opt_callback_fun.opt_sequence

    z = np.random.randn(1000, jax_funs["n_params"])

    dadvi_draws_flat = get_lrvb_draws(np.split(dadvi_res, 2)[0], lrvb_cov, z)

    dadvi_dict = transform_dadvi_draws(
        m,
        dadvi_draws_flat,
        jax_funs["unflatten_fun"],
        add_chain_dim=True,
        keep_untransformed=True,
    )

    target_dir = join(target_dir, "lrvb_doubling_results")

    makedirs(join(target_dir, "draw_dicts"), exist_ok=True)
    makedirs(join(target_dir, "lrvb_info"), exist_ok=True)
    np.savez(join(target_dir, "draw_dicts", model_name + ".npz"), **dadvi_dict)

    kl_hist_dadvi = [
        estimate_kl_fresh_draws(dadvi_funs, cur_hist["theta"], seed=2)
        for cur_hist in dadvi_opt_sequence
    ]

    metadata_final = extract_metadata_dict(opt_result)
    all_metadata = {
        cur_M: extract_metadata_dict(cur_result)
        for cur_M, cur_result in all_opt_results.items()
    }

    with open(join(target_dir, "lrvb_info", model_name + ".pkl"), "wb") as f:
        pickle.dump(
            {
                "last_step_info": metadata_final,
                "all_doubling_step_info": all_metadata,
                "runtime": finish_time - start_time,
                **get_run_datetime_and_hostname(),
            },
            f,
        )
