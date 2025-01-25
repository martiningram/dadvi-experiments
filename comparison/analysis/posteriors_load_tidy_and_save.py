from os.path import join
import pandas as pd
import numpy as np
from argparse import ArgumentParser

from load_and_tidy_posteriors_lib import (
    VALID_METHODS,
    GetMetadataDataframe,
    GetMethodDataframe,
    GetTraceDataframe,
    GetUnconstraintedParamsDataframe,
    GetMCMCDiagnosticsDataframe,
)

parser = ArgumentParser()
parser.add_argument("--input-folder", required=True, help="Path to the model runs")
parser.add_argument(
    "--dry-run", action="store_true", help="If passed, does not save the output"
)
args = parser.parse_args()

# Set to false for a dry run; doing so will not overwrite the csv
save_output = not args.dry_run

input_folder = args.input_folder
output_folder = input_folder

folder_method_list = (
    (join(input_folder, "nuts_results/"), "NUTS"),
    (join(input_folder, "dadvi_results/"), "DADVI"),
    (join(input_folder, "raabbvi_results/"), "RAABBVI"),
    (join(input_folder, "sadvi_results/"), "SADVI"),
    (join(input_folder, "sfullrank_advi_results/"), "SADVI_FR"),
    (join(input_folder, "lrvb_Direct_results/"), "LRVB"),
    (join(input_folder, "lrvb_doubling_results"), "LRVB_Doubling"),
    (join(input_folder, "lrvb_cg_results"), "LRVB_CG"),
)

posterior_dfs = []
for folder, method in folder_method_list:
    print(f"Loading {method}")
    posterior_dfs.append(GetMethodDataframe(folder, method))
posterior_df = pd.concat(posterior_dfs)

# xarray, scipy, jax, jaxlib are required to load the pickle files
# Also, pandas has to be < 2.0.0 (e.g. 1.5.3)
metadata_dfs = []
for folder, method in folder_method_list:
    print(f"Loading {method}")
    metadata_dfs.append(GetMetadataDataframe(folder, method))

metadata_df = pd.concat(metadata_dfs)

trace_dfs = []
for folder, method in folder_method_list:
    print(f"Loading {method}")
    trace_dfs.append(GetTraceDataframe(folder, method))

trace_df = pd.concat(trace_dfs)

if save_output:
    print("Saving output")
    posterior_df.to_csv(join(output_folder, "posteriors_tidy.csv"), index=False)
    metadata_df.to_csv(join(output_folder, "metadata_tidy.csv"), index=False)
    trace_df.to_csv(join(output_folder, "trace_tidy.csv"), index=False)

# Save the names of unconstrained parameters
if save_output:
    folder, method = folder_method_list[1]
    assert method == "DADVI"
    param_df = GetUnconstraintedParamsDataframe(folder, method)
    param_df.to_csv(join(output_folder, "params_tidy.csv"), index=False)


# Save the full MCMC diagnostic information
folder, method = folder_method_list[0]
assert method == "NUTS"
mcmc_df = GetMCMCDiagnosticsDataframe(folder, method)
if save_output:
    print("Saving MCMC output")
    mcmc_df.to_csv(join(output_folder, "mcmc_diagnostics_tidy.csv"), index=False)

raw_metadata = {}
model_names = {}
for folder, method in folder_method_list:
    print(f"Loading {method}")
    raw_metadata[method], model_names[method] = GetMetadataDataframe(
        folder, method, return_raw_metadata=True
    )
