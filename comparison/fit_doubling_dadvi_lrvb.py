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
from os import makedirs
from os.path import join
import pickle


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.set_start_method("fork")

    model_name = sys.argv[1]
    target_dir = sys.argv[2]
    m = load_model_by_name(model_name)

    # This will store the sequence of parameters
    opt_callback_fun.opt_sequence = []

    start_time = time.time()
    jax_funs = get_jax_functions_from_pymc(m)
    dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
    init_means = np.zeros(jax_funs["n_params"])
    init_log_vars = np.zeros(jax_funs["n_params"]) - 3
    init_var_params = np.concatenate([init_means, init_log_vars])
    opt_result = optimise_dadvi_by_doubling(
        init_var_params, dadvi_funs, seed=2, verbose=True, start_m_power=3
    )

    opt = opt_result["dadvi_result"]["optimisation_result"]
    dadvi_res = opt["opt_result"].x
    zs = opt_result["zs"]
    lrvb_cov = opt_result["dadvi_result"]["lrvb_covariance"]

    print(opt_result['M'], opt_result['ratio'])

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
        estimate_kl_fresh_draws(dadvi_funs, cur_hist["theta"])
        for cur_hist in dadvi_opt_sequence
    ]

    with open(join(target_dir, "lrvb_info", model_name + ".pkl"), "wb") as f:
        pickle.dump(
            {
                "opt_result": opt,
                "fixed_draws": zs,
                "M": zs.shape[0],
                "kl_hist": kl_hist_dadvi,
                "opt_sequence": dadvi_opt_sequence,
                "runtime": runtime_dadvi,
                "lrvb_cov": lrvb_cov,
            },
            f,
        )
