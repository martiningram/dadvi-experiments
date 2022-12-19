library(tidyverse)

base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/dadvi-experiments"
input_folder <- file.path(base_folder, "comparison/blade_runs/") 

raw_posteriors_df <- read.csv(file.path(input_folder, "posteriors_tidy.csv"), as.is=TRUE)
metadata_df <- read.csv(file.path(input_folder, "metadata_tidy.csv"), as.is=TRUE)

num_methods <- length(unique(posteriors_df$method))


#########################################################
# Check for models that didn't run with all methods
model_methods <-
    group_by(raw_posteriors_df, model, method) %>%
    summarise(.groups="drop") 

bad_models <- 
    model_methods %>%
    group_by(model) %>%
    summarize(n=n(), .groups="drop") %>%
    filter(n <  num_methods) %>%
    pull(model)

# See which ones are missing and filter them out
filter(model_methods, model %in% bad_models) %>%
    mutate(value=TRUE) %>%
    pivot_wider(id_cols=model, names_from=method)

posteriors_df <-
    raw_posteriors_df %>%
    filter(!(model %in% bad_models)) %>%
    inner_join(metadata_df, by=c("model", "method"))

# Basically SADVI and RAABVI never converge?
posteriors_df %>%
    group_by(method) %>%
    summarize(prop_converged=mean(converged))


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
           sd_rel_err=(sd - sd_ref) / sd_ref)

# Hmm, something's up with microcredit NUTS
filter(results_df, sd_ref < 1e-6)
filter(posteriors_df, method == "NUTS", model=="microcredit", param == "beta_full")

# NB we might be able to indentify random effects by the vertical bar
unique(results_df$param)

agg_results_df <-
    group_by(results_df, model, method) %>%
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
               by=c("model"))

ggplot(comp_df, aes(x=mean_z_rmse_1, y=mean_z_rmse_2)) +
    geom_point() +
    geom_text(aes(label=model), hjust="left", nudge_x=0.01) +
    geom_abline(aes(slope=1, intercept=0)) +
    xlab(method1) + ylab(method2) +
    scale_x_log10() + scale_y_log10()

