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


# Bin and look for p-value uniformity
n_bins <- 100
p_breaks <- seq(0, 1, 1/n_bins)
stopifnot(length(p_breaks) == n_bins + 1)
# Time consuming for some reason
coverage_df <-
    coverage_df %>%
    mutate(p_bucket=cut(p_val, p_breaks))

coverage_df <-
    coverage_df %>%
    mutate(group_col=paste(model))

plot_df <-
    coverage_df %>%
    group_by(num_draws, group_col, p_bucket) %>%
    summarize(bucket_n=n(), .groups="drop") %>%
    inner_join(
        coverage_df %>%
            group_by(num_draws, group_col) %>%
            summarize(group_n=n(), .groups="drop"),
        by=c("num_draws", "group_col")) %>%
    mutate(p_dens=bucket_n  / group_n)

# Sanity check.  It seems the use of grouping with multiple n() calls
# does not work the way I'd expect
group_by(plot_df) %>%
    group_by(num_draws, group_col, group_n) %>%
    summarize(s=sum(p_dens), .groups="drop") %>%
    pull(s) %>%
    unique()

ggplot(plot_df) +
    geom_line(aes(x=p_bucket, y=n_bins * p_dens, group=group_col)) +
    facet_grid(num_draws ~ .) +
    expand_limits(y=0)

