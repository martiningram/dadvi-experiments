import sys
from os import makedirs
from os.path import join
import numpy as np
import time
import pandas as pd

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
        fit_result_nuts = pm.sample(cores=1, chains=4)
        # if model_name in ['potus', 'occ_det']:
        # else:
        #     # Use the defaults
        #     fit_result_nuts = pm.sample()

    end_time = time.time()
    runtime = end_time - start_time
    print("Done")

    target_dir = join(target_dir, "nuts_results")

    makedirs(join(target_dir, "netcdfs"), exist_ok=True)
    makedirs(join(target_dir, "runtimes"), exist_ok=True)
    makedirs(join(target_dir, "draw_dicts"), exist_ok=True)

    fit_result_nuts.to_netcdf(join(target_dir, "netcdfs", model_name + ".netcdf"))
    draw_dict = arviz_to_draw_dict(fit_result_nuts)
    np.savez(join(target_dir, "draw_dicts", model_name + ".npz"), **draw_dict)

    pd.Series({"runtime": runtime}).to_csv(
        join(target_dir, "runtimes", model_name + ".csv")
    )
