from os.path import join, split, splitext
from glob import glob
import pandas as pd
import numpy as np
import pickle

VALID_METHODS = \
    ['NUTS', 'RAABBVI', 'DADVI', 'LRVB', 'SADVI', 'SADVI_FR', 'LRVB_Doubling']


def LoadPickleSafely(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


def RepList(x, n):
    return [x for _ in range(n)]


def GetDrawFilenames(folder):
    draw_folder = join(folder, 'draw_dicts')
    draw_filenames = glob(join(draw_folder, "*.npz"))
    if len(draw_filenames) == 0:
        raise ValueError(f'No npz files found in {draw_folder}')
    model_names = [ splitext(split(filename)[-1])[0]
                    for filename in draw_filenames ]
    return draw_filenames, model_names


def GetMethodDataframe(folder, method):
    draw_filenames, model_names = GetDrawFilenames(folder)

    draw_dict = {
        model: dict(np.load(filename))
        for filename, model in zip(draw_filenames, model_names) }

    method_dict = {
        'model': [],
        'param': [],
        'ind': [],
        'mean': [],
        'sd': []
    }

    for model in draw_dict.keys():
        for par, draws in draw_dict[model].items():
            par_mean = np.mean(draws, axis=(0,1)).flatten()
            par_sd = np.std(draws, axis=(0,1)).flatten()
            num_rows = len(par_mean)
            method_dict['model'].append(RepList(model, num_rows))
            method_dict['param'].append(RepList(par, num_rows))
            method_dict['ind'].append(np.arange(num_rows))
            method_dict['mean'].append(par_mean)
            method_dict['sd'].append(par_sd)

    method_df = pd.DataFrame()
    for k,v in method_dict.items():
        method_df[k] = np.hstack(v)
    method_df['method'] = method

    return method_df


def CheckConvergence(method, metadata):
    missing_value = float('NaN')
    if method == 'NUTS':
        return missing_value
    elif method == 'RAABBVI':
        n_steps = metadata['kl_hist_i'].max()
        return n_steps < 19900
    elif method == 'DADVI':
        return missing_value
    elif method == 'LRVB':
        return missing_value
    elif method == 'SADVI':
        return metadata['steps'] < 100000
    elif method == 'SADVI_FR':
        return missing_value
    elif method == 'LRVB_Doubling':
        return missing_value
    else:
        print(f'Invalid method {method}\n')
        assert(False)


def GetMetdataDataframe(folder, method):
    subdir_lookup = {
        'RAABBVI': 'info',
        'DADVI': 'dadvi_info',
        'LRVB': 'lrvb_info',
        'SADVI': 'info',
        'SADVI_FR': 'info',
        'LRVB_Doubling': 'lrvb_info' }

    draw_filenames, model_names = GetDrawFilenames(folder)

    if method == 'NUTS':
        runtime_filenames = [
            x.replace('draw_dicts', 'runtimes').replace('.npz', '.csv')
            for x in draw_filenames ]
        raw_metadata = [ { 'runtime': pd.read_csv(x)['0'].iloc[0] }
                         for x in runtime_filenames ]
    else:
        subdir = subdir_lookup[method]
        metadata_filenames = [
            x.replace('draw_dicts', subdir).replace('.npz', '.pkl')
            for x in draw_filenames ]
        raw_metadata = [ LoadPickleSafely(x) for x in metadata_filenames ]

    metadata_df = pd.DataFrame({
        'method': RepList(method, len(raw_metadata)),
        'model': model_names,
        'runtime': [ m['runtime'] for m in raw_metadata ],
        'converged': [ CheckConvergence(method, m) for m in raw_metadata ]
        } )
    return metadata_df
