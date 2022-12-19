from os.path import join, split, splitext
from glob import glob
import pandas as pd
import numpy as np
import pickle

VALID_METHODS = \
    ['NUTS', 'RAABBVI', 'DADVI', 'LRVB', 'SADVI', 'SADVI_FR', 'LRVB_Doubling']


def load_pickle_safely(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)

def maybe_load_pickle(pickle_file):
    try:
        result = load_pickle_safely(pickle_file)
    except Exception:
        return None

# For the draws arrays, the first index is the chain and the second is
# the draw of that chaing, so we want to average over the first two
# indices when computing posterior moments.
def compute_means(draw_dict):
    return {x: y.mean(axis=(0, 1)) for x, y in draw_dict.items()}

def compute_sds(draw_dict):
    return {x: y.std(axis=(0, 1)) for x, y in draw_dict.items()}

def compute_z_score_mean(mean_dict, ref_mean_dict, ref_sd_dict):
    return {x: (mean_dict[x] - ref_mean_dict[x]) / ref_sd_dict[x]
            for x in mean_dict}

def compute_relative_error_sd(sd_dict, ref_sd_dict):
    return {x: (sd_dict[x] - ref_sd_dict[x]) / ref_sd_dict[x]
            for x in ref_sd_dict}

def flatten_dict(var_dict, names):
    return np.concatenate([var_dict[x].reshape(-1) for x in names])




# The experimental results are saved in method-specific folders.
# Load all the results in a folder into a pandas dataframe.  The output
# of each method is some draws from the posterior.
# The means and sds columns are dictionaries thereof, named by parameter.
def load_moment_df(draw_folder):
    model_dicts = glob(join(draw_folder, "draw_dicts", "*.npz"))
    data = pd.DataFrame({"draw_dict_path": model_dicts})

    data["draws"] = data["draw_dict_path"].apply(lambda x: dict(np.load(x)))
    data["means"] = data["draws"].apply(compute_means)
    data["sds"] = data["draws"].apply(compute_sds)

    # No need for the draws any more for now
    data = data.drop(columns="draws")

    # Fetch model name
    data['model_name'] = data['draw_dict_path'].apply(
        lambda x: splitext(split(x)[-1])[0])

    return data


def check_convergence_sadvi(metadata, max_iter=100000):
    return metadata['steps'] < max_iter


def check_convergence_raabbvi(metadata, max_iter=19900):
    n_steps = metadata['kl_hist_i'].max()
    return n_steps < max_iter


# Add some extra information to the moment_df dataframe
def add_metadata(moment_df, method):
    assert method in VALID_METHODS

    if method in ['RAABBVI', 'DADVI', 'LRVB', 'SADVI',
                  'SADVI_FR', 'LRVB_Doubling']:
        subdir_lookup = {
            'RAABBVI': 'info',
            'DADVI': 'dadvi_info',
            'LRVB': 'lrvb_info',
            'SADVI': 'info',
            'SADVI_FR': 'info',
            'LRVB_Doubling': 'lrvb_info'
        }
        subdir = subdir_lookup[method]
        moment_df["info_path"] = (
            moment_df["draw_dict_path"]
            .str.replace("draw_dicts", subdir)
            .str.replace(".npz", ".pkl", regex=False)
        )

        moment_df['metadata'] = moment_df['info_path'].apply(
            load_pickle_safely)
        moment_df['runtime'] = moment_df['metadata'].apply(
            lambda x: x['runtime'])

        if method.startswith('SADVI'):
            moment_df['converged'] = moment_df['metadata'].apply(
                check_convergence_sadvi)
        elif method == 'RAABBVI':
            moment_df['converged'] = moment_df['metadata'].apply(
                check_convergence_raabbvi)

    else:
        # It's NUTS; get runtime:
        assert method == "NUTS"
        moment_df['runtime_path'] = (
            moment_df['draw_dict_path']
            .str.replace('draw_dicts', 'runtimes')
            .str.replace('.npz', '.csv', regex=False)
        )
        moment_df['runtime'] = (
            moment_df['runtime_path']
            .apply(lambda x: pd.read_csv(x)['0'].iloc[0])
        )

        # TODO: get rhat

    return moment_df



# Compare one method's posterior to another, reference method (typically NUTS)
def add_deviation_stats(model_df, reference_df):
    together = model_df.merge(
        reference_df, on="model_name", suffixes=("_model", "_reference")
    )

    together["mean_deviations"] = together.apply(
        lambda x: compute_z_score_mean(
            x["means_model"], x["means_reference"], x["sds_reference"]
        ),
        axis=1,
    )

    together["sd_deviations"] = together.apply(
        lambda x: compute_relative_error_sd(
            x["sds_model"], x["sds_reference"]), axis=1
    )

    # Note: the sorting breaks the order relative to the *_deviations.
    # It doesn't seem to matter for the way this is used, though.
    together["var_names"] = together["means_reference"].apply(
        lambda x: sorted(list(x.keys()))
    )

    # Add these to the model stats
    cols_to_keep = [
        "model_name",
        "mean_deviations",
        "sd_deviations",
        "var_names",
    ]

    new_stats = together[cols_to_keep]

    return model_df.merge(new_stats, on='model_name', how='left')


# Compute some summary stats within a model of average errors relative
# to a reference model.  This is for the output of add_deviation_stats
def add_derived_stats(model_df):
    # These first two just flatten the dictionary in each row of a dataframe,
    # the apply is to each row: each row of mean_deviations is a dictionary,
    # and each row of mean_deviations_flat is an array.
    model_df["mean_deviations_flat"] = model_df.apply(
        lambda x: flatten_dict(x["mean_deviations"], x["var_names"]), axis=1)

    model_df["sd_deviations_flat"] = model_df.apply(
        lambda x: flatten_dict(x["sd_deviations"], x["var_names"]), axis=1)

    # So the RMS is within a model.  (Were you to aggregate you should
    # aggregate before the square root)
    model_df['mean_rms'] = model_df['mean_deviations_flat'].apply(
        lambda x: np.sqrt(np.mean(x**2)))

    model_df['sd_rms'] = model_df['sd_deviations_flat'].apply(
        lambda x: np.sqrt(np.mean(x**2)))

    return model_df
