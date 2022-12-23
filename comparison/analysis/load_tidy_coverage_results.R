library(gridExtra)

library(tidyverse)

base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/dadvi-experiments"
input_folder <- file.path(base_folder, "comparison/analysis/coverage_warm_starts_rerun/") 


raw_coverage_df <- read.csv(file.path(input_folder, "coverage_tidy.csv"), as.is=TRUE)

# We don't really postprocess the metadata
metadata_df <- raw_metadata_df

raw_coverage_df$model %>% unique()

# These test models shouldn't really be in there
bad_models <- c("test", "test_rstanarm")
non_arm_models <- c("potus", "tennis", "microcredit", "occ_det")

# This function is more convenient than always grouping and merging on is_arm
IsARM <- function(model) { !(model %in% non_arm_models) }

REF_SEED <- "reference"
stopifnot(sum(raw_coverage_df$seed == REF_SEED) > 0)

coverage_df <- 
    raw_coverage_df %>%
    filter(!(model %in% bad_models)) %>%
    mutate(is_arm=IsARM(model)) %>%
    filter(seed != REF_SEED)

non_arm_models %in% unique(coverage_df$model)


#########################################
# Compute a ground truth and p values

head(coverage_df)

coverage_df <-
    coverage_df %>%
    group_by(param, model, num_draws) %>%
    mutate(mean_all=mean(mean),
              n_runs=n(),
              freq_sd_all=sqrt(mean(freq_sd^2) / n_runs),
              .groups="drop") %>%
    mutate(z_score=(mean - mean_all) / freq_sd,
           p_val=pnorm(z_score))

coverage_df$n_runs %>% unique()

n_bins <- 30
p_breaks <- seq(0, 1, 1/n_bins)
stopifnot(length(p_breaks) == n_bins + 1)
plot_df <-
    coverage_df %>%
    group_by(model, num_draws) %>%
    mutate(num_z_vals=n()) %>%
    mutate(p_bucket=cut(p_val, p_breaks)) %>%
    group_by(model, num_draws, num_z_vals, p_bucket) %>%
    summarize(count=n(),
              p_dens=n_bins * n() / num_z_vals, .groups="drop")

plot_df %>% arrange(desc(p_dens))

ggplot(plot_df) +
    geom_line(aes(x=p_bucket, y=p_dens, group=model), alpha=0.1) +
    facet_grid(num_draws ~ .) +
    expand_limits(y=0)

