from glob import glob
import os
from os.path import join, split, splitext, dirname
import pandas as pd
from scipy.stats import norm
import numpy as np
from functools import partial
from scipy import stats
import pickle
from argparse import ArgumentParser


def RepList(x, n):
    return [x for _ in range(n)]


def GetCoverageDataframe(filename, M):

    print(filename)
    raw_model_df = pd.read_pickle(filename)

    # The final directory is the model name
    model_name = split(os.path.dirname(filename))[-1]

    num_runs = raw_model_df.shape[0]

    means = raw_model_df["means"].to_numpy()
    seeds = raw_model_df["seed"].to_numpy()
    freq_sds = raw_model_df["freq_sds"].to_numpy()
    assert len(means) == len(freq_sds)
    num_runs = len(means)
    param_dim = len(means[0])

    # A for loop is not the most efficient or elegant but it will let me make sure
    # everything lines up correctly
    model_dict = {"seed": [], "param": [], "mean": [], "freq_sd": []}
    param_dims = np.arange(param_dim)
    for i in range(num_runs):
        assert len(means[i]) == len(freq_sds[i])
        model_dict["seed"].append(RepList(seeds[i], param_dim))
        model_dict["param"].append(param_dims)
        model_dict["mean"].append(means[i])
        model_dict["freq_sd"].append(freq_sds[i])

    #     # Save the ''reference'' as well
    #     reference_means = raw_model_df['reference_means'].iloc[0]
    #     reference_freq_sds = raw_model_df['reference_freq_sds'].iloc[0]
    #     assert(len(reference_means) == param_dim)
    #     assert(len(reference_means) == len(reference_freq_sds))
    #     model_dict['seed'].append(RepList('reference', param_dim))
    #     model_dict['param'].append(param_dims)
    #     model_dict['mean'].append(reference_means)
    #     model_dict['freq_sd'].append(reference_freq_sds)

    model_df = pd.DataFrame()
    for k, v in model_dict.items():
        model_df[k] = np.hstack(v)
    model_df["model"] = model_name
    model_df["num_draws"] = M

    return model_df


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--input-folder", required=True, help="Path to the coverage runs"
    )
    args = parser.parse_args()

    input_path = args.input_folder
    output_path = input_path

    M_vals = [8, 16, 32, 64]

    model_dfs = []

    for M in M_vals:
        print(f"Loading for {M} draws")
        coverage_filenames = glob(
            join(input_path, f"M_{M}", "*", "coverage_results.pkl")
        )
        assert len(coverage_filenames) > 0
        model_dfs.append(
            pd.concat(
                [GetCoverageDataframe(filename, M) for filename in coverage_filenames]
            )
        )

    model_df = pd.concat(model_dfs)

    print(output_path)
    model_df.to_csv(join(output_path, "coverage_tidy.csv"), index=False)
