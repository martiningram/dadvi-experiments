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

non_arm_models %in% unique(coverage_df$model)


#########################################
# Compute a ground truth and p values



if (TRUE) {
    # Compare the the average within the set of runs
    tmp_df <- 
        raw_coverage_df %>%
        filter(!(model %in% bad_models)) %>%
        mutate(is_arm=IsARM(model))

    truth_df <-     
        tmp_df %>%
        filter(num_draws==max(num_draws)) %>%
        group_by(param, model, num_draws) %>%
        summarize(mean_all=mean(mean),
                  n_runs=n(),
                  freq_sd_all=sqrt(mean(freq_sd^2) / n_runs),
                  .groups="drop")
    coverage_df <-
        inner_join(tmp_df,
                   truth_df %>% select(param, model, mean_all, freq_sd_all, n_runs),
                   by=c("param", "model")) %>%
        mutate(z_score=(mean - mean_all) / freq_sd,
               p_val=pnorm(z_score),
               sd_ratio=freq_sd / freq_sd_all)

    # Check whether the sampling variability of the average is negligible.
    summary(coverage_df %>% filter(num_draws < max(num_draws)) %>% pull(sd_ratio))

} else {
    # Alternatively use a reference value
    reference_seed <- REF_SEED
    tmp_df <-
        raw_coverage_df %>%
        filter(!(model %in% bad_models)) %>%
        select(seed, param, model, num_draws, mean, freq_sd)
    coverage_df <-
        inner_join(filter(tmp_df, seed != reference_seed),
                   filter(tmp_df, seed == reference_seed),
                   by=c("param", "model", "num_draws"),
                   suffix=c("", "_ref")) %>%
        mutate(freq_diff_sd=sqrt(freq_sd^2 + freq_sd_ref^2),
               z_score=(mean - mean_ref) / freq_diff_sd,
               p_val=pnorm(z_score),
               is_arm=IsARM(model))
    rm(tmp_df)
}

# TODO: probably you should use the 64-draw runs as reference values for
# the other ones.  Once we get to 32 everything is okay anyway.


stopifnot(nrow(coverage_df) > 0)




#######################################################
# Check for p-value uniformity in a variety of ways


# Bin and look for p-value uniformity
n_bins <- 100
p_breaks <- seq(0, 1, 1/n_bins)
stopifnot(length(p_breaks) == n_bins + 1)
# Time consuming for some reason
coverage_df <-
    coverage_df %>%
    mutate(p_bucket=cut(p_val, p_breaks))



GetKSPval <- function(x) {
    ks.test(x, "punif")$p.value
}


ks_test_arm_df <-
    coverage_df %>%
    filter(is_arm) %>%
    mutate(group_col=paste(model)) %>%
    group_by(num_draws, group_col) %>%
    summarize(ks_test=GetKSPval(p_val), .groups="drop") %>%
    mutate(reject=ks_test < 0.01) %>%
    arrange(num_draws, ks_test)
if (FALSE) {
    View(ks_test_arm_df)
}


# For the non-ARM models, we can't reject any of the individual parameters
ks_test_nonarm_df <-
    coverage_df %>%
    filter(!is_arm) %>%
    group_by(num_draws, model, param) %>%
    summarize(ks_test=GetKSPval(p_val), .groups="drop") %>%
    mutate(reject=ks_test < 0.01) %>%
    arrange(num_draws, ks_test)

ks_test_nonarm_df %>%
    group_by(num_draws, model) %>%
    summarize(num_reject=sum(reject)) 


# But some of the models can be rejected
ks_test_models_df <-
    coverage_df %>%
    group_by(num_draws, model, is_arm) %>%
    summarize(ks_test=GetKSPval(p_val), .groups="drop") %>%
    mutate(reject=ks_test < 0.01) %>%
    arrange(num_draws, ks_test)

ks_test_models_df %>%
    mutate(model_short=ifelse(is_arm, "ARM", model)) %>%
    group_by(num_draws, model_short) %>%
    summarize(num_reject=sum(reject)) 

if (FALSE) {
    View(ks_test_nonarm_df)
}


ks_test_df <-
    coverage_df %>%
    mutate(group_col=paste(model)) %>%
    group_by(num_draws, group_col) %>%
    summarize(ks_test=GetKSPval(p_val), .groups="drop") %>%
    mutate(reject=ks_test < 0.01,
           is_arm=IsARM(group_col)) %>%
    arrange(num_draws, ks_test)
if (FALSE) {
    View(ks_test_df %>% filter(is_arm))
    View(ks_test_df %>% filter(!is_arm))
}



if (TRUE) {
    # Plot the non-ARM results
    plot_df <-
        coverage_df %>%
        mutate(group_col=paste(model)) %>%
        filter(!is_arm)
    
} else {
    # Plot the ARM results
    plot_df <-
        coverage_df %>%
        mutate(group_col=paste("")) %>%
        filter(is_arm)
}


plot_df <-
    plot_df %>%
    group_by(num_draws, group_col, p_bucket) %>%
    summarize(bucket_n=n(), .groups="drop") %>%
    inner_join(
        plot_df %>%
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


if (FALSE) {
    ggplot(plot_df) +
        geom_line(aes(x=p_bucket, y=n_bins * p_dens, group=group_col, color=group_col)) +
        facet_grid(num_draws ~ .) +
        expand_limits(y=0) +
        theme(axis.text.x=element_blank())
}
