library(gridExtra)
library(tidyverse)
library(shiny)



base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/dadvi-experiments"
paper_base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/fd-advi-paper"
analysis_folder <- file.path(base_folder, "comparison/analysis")

source(file.path(analysis_folder, "load_tidy_lib.R"))

input_folder <- file.path(base_folder, "comparison/blade_runs/")
output_folder <- file.path(paper_base_folder, "experiments_data")

models_to_remove <- GetModelsToRemove()
non_arm_models <- GetNonARMModels()


# A list of stuff to be saved for the paper
save_list <- list()

# Load data

raw_posteriors_df <- read.csv(file.path(input_folder, "posteriors_tidy.csv"), as.is=TRUE)
raw_metadata_df <- read.csv(file.path(input_folder, "metadata_tidy.csv"), as.is=TRUE)
raw_trace_df <- read.csv(file.path(input_folder, "trace_tidy.csv"), as.is=TRUE)
raw_param_df <- read.csv(file.path(input_folder, "params_tidy.csv"), as.is=TRUE)
mcmc_diagnostics_df <- read.csv(file.path(input_folder, "mcmc_diagnostics_tidy.csv"), as.is=TRUE)


stopifnot(length(unique(raw_param_df$method)) == 1)

model_dims <-
    raw_posteriors_df %>%
    filter(method == "DADVI") %>%
    group_by(model, param, method) %>%
    summarize(dims=n(), .groups="drop") %>%
    inner_join(raw_param_df %>% rename(param=unconstrained_params),
               by=c("model", "method", "param")) %>%
    group_by(model) %>%
    summarize(dim=sum(dims))
save_list[["model_dims"]] <- model_dims



# Compute M * (2 * hvp + grad calss for op count)
filter(raw_metadata_df, method == "LRVB_Doubling")
num_methods <- length(unique(raw_posteriors_df$method))


# LRVB_CG didn't save the number of draws, so we need to join with DADVI
dadvi_num_draws <-
    raw_metadata_df %>%
    filter(method == "DADVI") %>%
    select(model, num_draws)

metadata_df <-
    raw_metadata_df %>%
    left_join(dadvi_num_draws, by="model", suffix=c("", "_dadvi")) %>%
    filter(!(model %in% models_to_remove)) %>%
    mutate(op_count=case_when(
        method == "DADVI" ~ num_draws * (2 * hvp_count + grad_count),
        method == "LRVB" ~ num_draws * (2 * hvp_count + grad_count),
        method == "LRVB_CG" ~ num_draws_dadvi * (2 * hvp_count + grad_count),
        TRUE ~ grad_count
    )) %>%
    mutate(is_arm = IsARM(model),
           time_per_op=runtime / op_count,
           converged=as.logical(converged)) %>%
    inner_join(select(model_dims, model, dim), by="model")

# Sanity check
metadata_df %>%
    filter(method %in% c("DADVI", "LRVB")) %>%
    mutate(check=num_draws == num_draws_dadvi) %>%
    pull(check) %>%
    all() %>%
    stopifnot()

save_list[["metadata_df"]] <- metadata_df


non_arm_models <-
    metadata_df %>%
    filter(!is_arm) %>%
    pull(model) %>%
    unique()
save_list[["non_arm_models"]] <- non_arm_models


# You can inner join with this
# dataframe to get LRVB results from a dataframe.
# The logic is that if LRVB_CG is available, that should count as our
# LRVB method.
lrvb_methods_df <-
    metadata_df %>%
    filter(method %in% c("LRVB", "LRVB_CG")) %>%
    select(method, model, op_count) %>%
    pivot_wider(id_cols="model", names_from=method, values_from=op_count) %>%
    mutate(method=case_when(!is.na(LRVB_CG) ~ "LRVB_CG",
                            TRUE ~ "LRVB")) %>%
    select(model, method)


#########################################################
# Check for models that didn't run with all methods

unique(raw_posteriors_df$model) %>% sort()

unique(model_methods$method)

# For now a model is allowed to be missing LRVB_Doubling,
# and must have at least one of LRVB or LRVB_CG
model_methods <-
    raw_posteriors_df %>%
    filter(method != "LRVB_Doubling") %>%
    mutate(method=str_replace(method, "LRVB_CG", "LRVB")) %>%
    group_by(model, method) %>%
    summarise(.groups="drop")

incomplete_models <-
    model_methods %>%
    group_by(model) %>%
    summarize(n=n(), .groups="drop") %>%
    filter(n <  max(n)) %>%
    pull(model)

# See which ones are missing.
print("Methods which we /do/ have for incomplete models:")
filter(model_methods, model %in% incomplete_models) %>%
    mutate(value=TRUE) %>%
    pivot_wider(id_cols=model, names_from=method)

filter(model_methods, model == "potus")

posteriors_df <-
    raw_posteriors_df %>%
    filter(!(model %in% bad_models)) %>%
    inner_join(metadata_df, by=c("model", "method")) %>%
    mutate(is_arm = IsARM(model))


# Make sure all methods worked for every ARM model
arm_models <-
    posteriors_df %>%
    filter(is_arm) %>%
    pull(model) %>%
    unique()
stopifnot(length(intersect(arm_models, incomplete_models)) == 0)


head(posteriors_df)

# TODO: check carefully
posteriors_df %>%
    group_by(model, method) %>%
    summarize(n=n())

metadata_df %>%
    group_by(model, method) %>%
    summarize(n=n())


########################################
# Check convergence


posteriors_df %>%
    filter(is.na(converged)) %>%
    pull(method) %>% unique()

# Basically SADVI and RAABVI rarely converge?
posteriors_df %>%
    group_by(method) %>%
    summarize(prop_converged=mean(converged))

mcmc_nonconverged_models <-
    posteriors_df %>%
    filter(method == 'NUTS', !converged) %>%
    pull(model) %>%
    unique()

filter(mcmc_diagnostics_df, model %in% mcmc_nonconverged_models) %>%
    group_by(model) %>%
    summarize(min_rhat=min(rhat, na.rm=TRUE),
              max_rhat=max(rhat, na.rm=TRUE),
              min_ess=min(ess),
              median_ess=median(ess),
              q10_ess=quantile(ess, 0.1))

# This is no good
filter(mcmc_diagnostics_df, model == "election88_full")

setdiff(mcmc_nonconverged_models, mcmc_bad_models)

filter(mcmc_diagnostics_df, model == "microcredit")


posteriors_df %>%
    group_by(method) %>%
    summarize(prop_converged=mean(converged))




########################################
# Look at the number of draws

metadata_df %>%
    filter(method %in% c("DADVI", "LRVB", "LRVB_Doubling")) %>%
    group_by(method, num_draws) %>%
    summarize(n=n())



########################################
# Inspect and categorize the parameters

# See the parameter dimensions and filter reporting.
# We will use this dataframe later to select which parameters
# get reported in the output.

# Manually label random effects for non-ARM models.  Potus is a bit arbitrary
# Some ARM random effect scale parameters have a | but they all have dimension 1.
IsRE <- function(model, is_arm, param, dimension) {
    case_when(
        model == "microcredit" & str_detect(param, "_k") ~ TRUE,
        model == "occ_det" & param %in% c("w_obs", "w_env") ~ TRUE,
        model == "tennis" & param == "player_skills" ~ TRUE,
        model == "potus" & (dimension > 51) ~ TRUE,
        is_arm & str_detect(param, fixed("|")) & (dimension > 1) ~ TRUE,
        TRUE ~ FALSE
    )
}

param_df <-
    posteriors_df %>%
    filter(method == "DADVI") %>%
    left_join(raw_param_df %>%
                  rename(param=unconstrained_params) %>%
                  mutate(status="unconstrainted") %>%
                  select(model, param, status),
              by=c("model", "param")) %>%
    mutate(is_unconstrained=!is.na(status)) %>%
    select(model, is_arm, param, is_unconstrained, -status) %>%
    group_by(model, is_arm, param, is_unconstrained) %>%
    summarize(dimension=n(), .groups="drop") %>%
    mutate(report_param=case_when(
        !is_arm & grepl("raw", param) ~ FALSE,
        TRUE ~ TRUE)) %>%
    mutate(is_re=IsRE(model, is_arm, param, dimension))

if (FALSE) {
    # Manual check
    View(filter(param_df, !is_arm, report_param))
    View(filter(param_df, is_arm))
}



#######################################################
# Save

save(posteriors_df, metadata_df, param_df, lrvb_methods_df, file="/tmp/foo.Rdata")

