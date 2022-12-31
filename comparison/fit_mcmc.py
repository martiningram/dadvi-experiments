import sys
from os import makedirs
from os.path import join
import numpy as np
import time
import pandas as pd
from dadvi.pymc.utils import get_unconstrained_variable_names
import pickle
import arviz as az
from utils import get_run_datetime_and_hostname

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.set_start_method("fork")

    import pymc as pm
    from utils import load_model_by_name, arviz_to_draw_dict

    model_name = sys.argv[1]
    target_dir = sys.argv[2]
    model = load_model_by_name(model_name)

    print("Fitting")
    start_time = time.time()

    with model as m:
        # Running in parallel gets stuck for some models. Fall back to sequential.
        # Microcredit strangely is killed on its third chain, so run only two.
        if model_name == "microcredit":
            num_chains = 2
            fit_result_nuts = pm.sample(cores=1, chains=num_chains)
        else:
            num_chains = 4
            fit_result_nuts = pm.sample(cores=1, chains=num_chains)
        # if model_name in ['potus', 'occ_det']:
        # else:
        #     # Use the defaults
        #     fit_result_nuts = pm.sample()

    end_time = time.time()
    runtime = end_time - start_time
    print("Done")

    target_dir = join(target_dir, "nuts_results")

    makedirs(join(target_dir, "netcdfs"), exist_ok=True)
    makedirs(join(target_dir, "nuts_info"), exist_ok=True)
    makedirs(join(target_dir, "draw_dicts"), exist_ok=True)

    try:
        fit_result_nuts.to_netcdf(join(target_dir, "netcdfs", model_name + ".netcdf"))
    except RuntimeError as e:
        print("NetCDF saving failed with error:")
        print(e)
        print("Continuing.")

    draw_dict = arviz_to_draw_dict(fit_result_nuts)
    np.savez(join(target_dir, "draw_dicts", model_name + ".npz"), **draw_dict)

    unconstrained_param_names = get_unconstrained_variable_names(model)

    rhats = az.rhat(fit_result_nuts)
    ess = az.ess(fit_result_nuts)
    # rhat_dict = rhats.to_dict()["data_vars"]
    # ess_dict = ess.to_dict()["data_vars"]

    dims = fit_result_nuts.posterior.dims

    metadata = {
        "runtime": runtime,
        "unconstrained_param_names": unconstrained_param_names,
        "ess": ess,
        "rhat": rhats,
        "n_chains": dims["chain"],
        "n_draws": dims["draw"],
        **get_run_datetime_and_hostname(),
    }

    target_file = join(target_dir, "nuts_info", f"{model_name}.pkl")

    with open(target_file, "wb") as f:
        pickle.dump(metadata, f)
