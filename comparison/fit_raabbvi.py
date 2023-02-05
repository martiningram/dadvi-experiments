from jax.config import config

config.update("jax_enable_x64", True)
import multiprocessing

multiprocessing.set_start_method("fork")
import sys
from utils import load_model_by_name, estimate_kl_fresh_draws
from dadvi.viabel.fit_with_viabel import fit_pymc_model_with_viabel
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.jax import build_dadvi_funs
import time
import numpy as np
import pandas as pd
from dadvi.core import get_dadvi_draws
from dadvi.pymc.pymc_to_jax import transform_dadvi_draws
from os import makedirs
from os.path import join
import pickle
from dadvi.pymc.utils import get_unconstrained_variable_names
from utils import get_run_datetime_and_hostname


model_name = sys.argv[1]
target_dir = sys.argv[2]
m = load_model_by_name(model_name)
seed = 2

np.random.seed(seed)

jax_funs = get_jax_functions_from_pymc(m)
dadvi_funs = build_dadvi_funs(jax_funs["log_posterior_fun"])
init_means = np.zeros(jax_funs["n_params"])
init_log_vars = np.zeros(jax_funs["n_params"]) - 3
init_var_params = np.concatenate([init_means, init_log_vars])

num_mc_samples = 50

start_time = time.time()
viabel_result = fit_pymc_model_with_viabel(
    m, num_mc_samples=num_mc_samples, init_var_param=init_var_params, n_iters=20000
)
end_time = time.time()
runtime_viabel = end_time - start_time

viabel_opt_params = viabel_result["opt_param"]
viabel_timings = pd.Series(viabel_result["call_times"])

z = np.random.randn(1000, jax_funs["n_params"])
viabel_draws_flat = get_dadvi_draws(viabel_opt_params, z)

viabel_dict = transform_dadvi_draws(
    m,
    viabel_draws_flat,
    jax_funs["unflatten_fun"],
    add_chain_dim=True,
    keep_untransformed=True,
)

compute_kl_every = 10

viabel_i = (
    np.arange(viabel_result["variational_param_history"].shape[0])[::compute_kl_every]
    * num_mc_samples
)
kl_hist_viabel = [
    estimate_kl_fresh_draws(dadvi_funs, cur_params, seed=2)
    for cur_params in viabel_result["variational_param_history"][::compute_kl_every]
]
rel_timings = viabel_timings.loc[viabel_i]

target_dir = join(target_dir, "raabbvi_results")

makedirs(join(target_dir, "draw_dicts"), exist_ok=True)
makedirs(join(target_dir, "info"), exist_ok=True)
np.savez(join(target_dir, "draw_dicts", model_name + ".npz"), **viabel_dict)

# TODO: Work out what else I can pickle
with open(join(target_dir, "info", model_name + ".pkl"), "wb") as f:
    pickle.dump(
        {
            "opt_result": viabel_opt_params,
            "kl_hist": kl_hist_viabel,
            "kl_hist_i": viabel_i,
            "kl_hist_times": rel_timings.values,
            "runtime": runtime_viabel,
            "unconstrained_param_names": get_unconstrained_variable_names(m),
            **get_run_datetime_and_hostname(),
        },
        f,
    )
