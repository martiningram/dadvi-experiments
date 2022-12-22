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


def GetEvaluationCount(method, metadata):
    missing_value = float('NaN')
    if method == 'NUTS':
        # Need to save the number of draws in the metadata
        n_calls = missing_value
    elif method == 'RAABBVI':
        n_calls = metadata['kl_hist_i'].max()
    elif method == 'DADVI':
        evaluation_count = metadata['opt_result']['evaluation_count']
        n_calls = evaluation_count['n_hvp_calls'] + \
            evaluation_count['n_val_and_grad_calls']
    elif method == 'LRVB':
        # Need to save the extra LRVB iterations in the metadata
        n_calls = missing_value
    elif method == 'SADVI':
        n_calls = metadata['steps']
    elif method == 'SADVI_FR':
        # This is not apples-to-apples obviously
        n_calls = metadata['steps']
    elif method == 'LRVB_Doubling':
        # Need to save all the steps in the metadata
        n_calls = missing_value
    else:
        raise ValueError(f'Invalid method {method}\n')
    return n_calls


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


def GetMetadataDataframe(folder, method, return_raw_metadata=False):
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

    if return_raw_metadata:
        return raw_metadata

    metadata_df = pd.DataFrame({
        'method': RepList(method, len(raw_metadata)),
        'model': model_names,
        'runtime': [ m['runtime'] for m in raw_metadata ],
        'converged': [ CheckConvergence(method, m) for m in raw_metadata ],
        'op_count': [ GetEvaluationCount(method, m) for m in raw_metadata ]
        } )

    return metadata_df


def GetObjectiveTraces(method, metadata):
    missing_value = [ float('NaN') ]
    if method == 'NUTS':
        # Doesn't make sense for NUTS
        obj_hist = missing_value
        step_hist = missing_value
    elif method == 'RAABBVI':
        obj_hist = np.array(metadata['kl_hist'])
        step_hist = np.array(metadata['kl_hist_i'])
    elif method == 'DADVI':
        obj_hist = np.array(metadata['kl_hist'])
        opt_sequence = metadata['opt_sequence']
        step_hist = np.array([ o['val_and_grad_calls'] +
                               o['hvp_calls'] for o in opt_sequence ])
        if len(step_hist) != len(obj_hist):
            raise ValueError(
                f'Different lengths for histories: '+
                f'{len(step_hist)} != {len(obj_hist)}')
    elif method == 'LRVB':
        # Doesn't make sense for LRVB
        obj_hist = missing_value
        step_hist = missing_value
    elif method == 'SADVI':
        # TODO: Save KL traces for SADVI
        obj_hist = missing_value
        step_hist = missing_value
    elif method == 'SADVI_FR':
        # TODO: Save KL traces for SADVI
        obj_hist = missing_value
        step_hist = missing_value
    elif method == 'LRVB_Doubling':
        # TODO: make sure this makese sense
        obj_hist = missing_value
        step_hist = missing_value
    else:
        print(f'Invalid method {method}\n')
        assert(False)

    return step_hist, obj_hist


def GetTraceDataframe(folder, method):
    draw_filenames, model_names = GetDrawFilenames(folder)
    raw_metadata = GetMetadataDataframe(folder, method, return_raw_metadata=True)
    traces = [ GetObjectiveTraces(method, m) for m in raw_metadata ]

    trace_dict = {
        'model': [],
        'n_calls': [],
        'obj_value': []
    }

    assert(len(traces) == len(model_names))
    for model_ind in range(len(traces)):
        model = model_names[model_ind]
        step_hist, obj_hist = traces[model_ind]
        assert(len(step_hist) == len(obj_hist))
        num_rows = len(step_hist)
        trace_dict['model'].append(RepList(model, num_rows))
        trace_dict['n_calls'].append(step_hist)
        trace_dict['obj_value'].append(obj_hist)

    trace_df = pd.DataFrame()
    for k,v in trace_dict.items():
        trace_df[k] = np.hstack(v)
    trace_df['method'] = method

    return trace_df
