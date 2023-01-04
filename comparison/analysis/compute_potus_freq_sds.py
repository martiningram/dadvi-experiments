# TODO: Maybe abstract so that we can do this for general functions...
from jax.scipy.special import expit
from jax import grad
from dadvi.utils import cg_using_fun_scipy
from dadvi.core import compute_score_matrix
from dadvi.pymc.models.potus import get_potus_model
import numpy as np
import json
from dadvi.pymc.pymc_to_jax import get_jax_functions_from_pymc
from dadvi.jax import build_dadvi_funs
import pickle
from glob import glob
from tqdm import tqdm
import pandas as pd
import sys
from datetime import datetime
from scipy.sparse import diags

M = int(sys.argv[1])
base_dir = f'/media/martin/External Drive/projects/lrvb_paper/potus_coverage_warm_starts/100_reruns/M_{M}/'

all_runs = glob(base_dir + '*/*.pkl')
reference = [x for x in all_runs if 'reference' in x][0]
other_runs = [x for x in all_runs if 'reference' not in x]

ref_results = pickle.load(open(reference, 'rb'))

data = json.load(open('../data/potus_data.json'))
np_data = {x: np.squeeze(np.array(y)) for x, y in data.items()}
potus_model = get_potus_model('../data/potus_data.json')
jax_funs = get_jax_functions_from_pymc(potus_model)

national_cov_matrix_error_sd = np.sqrt(
        np.squeeze(
            np_data["state_weights"].reshape(1, -1)
            @ (np_data["state_covariance_0"] @ np_data["state_weights"].reshape(-1, 1))
        )
    )
ss_cov_mu_b_T = (
    np_data["state_covariance_0"]
    * (np_data["mu_b_T_scale"] / national_cov_matrix_error_sd) ** 2
)
cholesky_ss_cov_mu_b_T = np.linalg.cholesky(ss_cov_mu_b_T)

ref_opt_params = ref_results['opt_result']['opt_result'].x
ref_z = ref_results['z']
dadvi_funs = build_dadvi_funs(jax_funs['log_posterior_fun'])

# Check whether this is the correct way to get the national pct!!
def compute_final_vote_share(full_var_params):

    # I think we only need the means:
    opt_means = full_var_params[:full_var_params.shape[0] // 2]

    mean_dict = jax_funs['unflatten_fun'](opt_means)

    rel_means = mean_dict['raw_mu_b_T']

    mean_shares = cholesky_ss_cov_mu_b_T @ rel_means + np_data["mu_b_prior"]

    return expit(mean_shares) @ np_data['state_weights']


def estimate_final_vote_share_and_freq_sd(opt_params, dadvi_funs, z, use_preconditioner=False):
    
    final_share = compute_final_vote_share(opt_params)
    
    rel_grad = grad(compute_final_vote_share)(opt_params)

    if use_preconditioner:
        opt_variances = np.exp(np.split(opt_params, 2)[1])**2
        diag_prec = np.concatenate([opt_variances, np.ones(z.shape[1])])
        preconditioner = diags(diag_prec)
    else:
        preconditioner = None
        
    rel_hvp = lambda x: dadvi_funs.kl_est_hvp_fun(opt_params, z, x)
    cg_result = cg_using_fun_scipy(rel_hvp, rel_grad, preconditioner=preconditioner, maxiter=1000)
    h_inv_g = cg_result[0]

    cg_success = cg_result[1]
    print(cg_success)
        
    score_mat = compute_score_matrix(opt_params, dadvi_funs.kl_est_and_grad_fun, z)
    score_mat_means = score_mat.mean(axis=0, keepdims=True)
    centred_score_mat = score_mat - score_mat_means
    
    vec = centred_score_mat @ h_inv_g
    M = score_mat.shape[0]
    freq_sd = np.sqrt((vec.T @ vec) / (M * (M - 1)))
    
    return final_share, freq_sd, cg_success

use_preconditioner = False

ref_vote_share, ref_sd, cg_success = estimate_final_vote_share_and_freq_sd(ref_opt_params, dadvi_funs, ref_z, use_preconditioner=use_preconditioner)

others_loaded = [pickle.load(open(x, 'rb')) for x in other_runs]

rerun_results = list()

rerun_results.append({'vote_share': ref_vote_share, 'sd': ref_sd,
                      'seed': ref_results['seed'], 'is_reference': True,
                      'M': M, 'filename': reference, 'cg_converged_after_1k': cg_success})

for cur_filename, cur_loaded in tqdm(zip(other_runs, others_loaded)):

    print(datetime.now(), cur_filename)
    
    other_z = cur_loaded['z']
    new_opt_params = cur_loaded['opt_result']['opt_result'].x
    vote_share, sd, cg_success = estimate_final_vote_share_and_freq_sd(new_opt_params, dadvi_funs, other_z)
   
    rerun_results.append({'vote_share': vote_share, 'sd': sd,
                          'seed': cur_loaded['seed'], 'is_reference': False,
                          'M': M, 'filename': cur_filename,
                          'cg_converged_after_1k': cg_success})

pd.DataFrame(rerun_results).to_csv(base_dir + f'rerun_sds_precondition_{use_preconditioner}.csv')
