library(tidyverse)

base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/dadvi-experiments"
input_folder <- file.path(base_folder, "comparison/blade_runs/") 

raw_posteriors_df <- read.csv(file.path(input_folder, "posteriors_tidy.csv"), as.is=TRUE)
metadata_df <- read.csv(file.path(input_folder, "metadata_tidy.csv"), as.is=TRUE)

num_methods <- length(unique(posteriors_df$method))


#########################################################
# Check for models that didn't run with all methods

unique(raw_posteriors_df$model) %>% sort()

model_methods <-
    group_by(raw_posteriors_df, model, method) %>%
    summarise(.groups="drop") 

incomplete_models <- 
    model_methods %>%
    group_by(model) %>%
    summarize(n=n(), .groups="drop") %>%
    filter(n <  num_methods) %>%
    pull(model)

# See which ones are missing.
filter(model_methods, model %in% incomplete_models) %>%
    mutate(value=TRUE) %>%
    pivot_wider(id_cols=model, names_from=method)

# These test models shouldn't really be in there
bad_models <- c("test", "test_rstanarm")
non_arm_models <- c("potus", "tennis", "microcredit", "occ_det")

# This function is more convenient than always grouping and merging on is_arm
IsARM <- function(model) { !(model %in% non_arm_models) }

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


########################################
# Inspect and categorize the parameters

# See how many parameters are in each model
posteriors_df %>%
    filter(method == "DADVI") %>%
    group_by(model, is_arm) %>%
    summarize(n_pars=n(), .groups="drop") %>%
    arrange(n_pars) %>%
    View()


# See the parameter dimensions and filter reporting
param_df <-
    posteriors_df %>%
    filter(method == "DADVI") %>%
    group_by(model, is_arm, param) %>%
    summarize(dimension=n(), .groups="drop") %>%
    arrange(model, dimension)

# Don't report "raw" parameters
param_df <- 
    param_df %>%
    mutate(report_param=case_when(
        !is_arm & grepl("raw", param) ~ FALSE,
        TRUE ~ TRUE))

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
param_df <- mutate(param_df, is_re=IsRE(model, is_arm, param, dimension))

if (FALSE) {
    # Manual check
    View(filter(param_df, !is_arm, report_param))
    View(filter(param_df, is_arm))
}


########################################
# Explore a little


# Basically SADVI and RAABVI rarely converge?
posteriors_df %>%
    group_by(method) %>%
    summarize(prop_converged=mean(converged))


# Hmm, something's up with microcredit
posteriors_df %>%
    filter(model=="microcredit", param == "beta_full", ind %in% c(3, 4, 5)) %>%
    select(model, param, ind, mean, sd, method)



########################################
# Compare to a reference method

reference_method <- "NUTS"
stopifnot(sum(posteriors_df$method == reference_method) > 0)

results_df <-
    inner_join(posteriors_df %>% filter(method != reference_method),
               posteriors_df %>% filter(method == reference_method),
               by=c("model", "param", "ind"),
               suffix=c("", "_ref")) %>%
    mutate(mean_z_err=(mean - mean_ref) / sd_ref,
           sd_rel_err=(sd - sd_ref) / sd_ref) %>%
    mutate(is_arm=IsARM(model)) %>%
    inner_join(param_df, by=c("model", "param")) %>%
    filter(report_param)

# Sanity check for bad reference values
filter(results_df, sd_ref < 1e-6) %>% 
    select(method, model, param, ind, mean, sd, mean_ref, sd_ref)


########################################
# Aggregate and compare results

agg_results_df <-
    group_by(results_df, model, method, is_re) %>%
    summarise(mean_z_rmse=sqrt(mean(mean_z_err^2)),
              sd_rel_rmse=sqrt(mean(sd_rel_err^2)),
              .groups="drop")


method1 <- "LRVB_Doubling"
method2 <- "RAABBVI"
stopifnot(method1 %in% unique(posteriors_df$method))
stopifnot(method2 %in% unique(posteriors_df$method))

comp_df <-
    inner_join(filter(agg_results_df, method == !!method1),
               filter(agg_results_df, method == !!method2),
               suffix=c("_1", "_2"),
               by=c("model", "is_re")) %>%
    mutate(is_arm=IsARM(model))

comp_df %>%
    filter(is_arm) %>%
    ggplot(aes(x=mean_z_rmse_1, y=mean_z_rmse_2)) +
        geom_point() +
        geom_text(aes(label=model), hjust="left", nudge_x=0.01) +
        geom_abline(aes(slope=1, intercept=0)) +
        xlab(method1) + ylab(method2) +
        facet_grid(~ is_re, scales="fixed") +
        scale_x_log10() + scale_y_log10()
    
