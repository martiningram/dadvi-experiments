import os
import numpy as np

def add_columns(df):
    
    df['means_with_names'] = df.apply(lambda x: {x['names'][0]: np.array(x['means'])}, axis=1)
    df['freq_sds_with_names'] = df.apply(lambda x: {x['names'][0]: np.array(x['freq_sds'])}, axis=1)
    
    return df

def save_dfs_by_M(model_df, model_name, coverage_base_dir):

    for cur_m in model_df['M'].unique():

        cur_data = model_df[model_df['M'] == cur_m]

        target_dir = os.path.join(coverage_base_dir, f'M_{cur_m}', model_name)

        os.makedirs(target_dir, exist_ok=True)

        cur_data.to_pickle(os.path.join(target_dir, 'coverage_results.pkl'))

