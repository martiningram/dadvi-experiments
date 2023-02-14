library(gridExtra)
library(tidyverse)

base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/dadvi-experiments"
paper_base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/fd-advi-paper"

input_folder <- file.path(base_folder, "comparison/analysis/coverage_warm_starts_rerun/") 
output_folder <- file.path(paper_base_folder, "experiments_data") 


raw_coverage_df <- read.csv(file.path(input_folder, "coverage_tidy.csv"), as.is=TRUE)

# We don't really postprocess the metadata
metadata_df <- raw_metadata_df

raw_coverage_df$model %>% unique()

# These test models shouldn't really be in there
bad_models <- c("test", "test_rstanarm")
non_arm_models <- c("potus", "tennis", "microcredit", "occ_det")

# These models are just modified versions of another model.  We should
# probably sort these out systematically.
repeated_models <- c(
    "radon_group_chr", "radon_intercept_chr", "radon_no_pool_chr",
    "wells_predicted", "mesquite_va")

# These models didn't work with NUTS well enough to use here.
mcmc_bad_models <- c("earnings_latin_square", "earnings_vary_si", "election88_full")


# This function is more convenient than always grouping and merging on is_arm
IsARM <- function(model) { !(model %in% non_arm_models) }

REF_SEED <- "reference"
stopifnot(sum(raw_coverage_df$seed == REF_SEED) > 0)

non_arm_models %in% unique(coverage_df$model)


# A list of stuff to be saved for the paper
save_list <- list()


#########################################
# Compute a ground truth and p values



if (TRUE) {
    # Compare the the average within the set of runs
    tmp_df <- 
        raw_coverage_df %>%
        filter(!(model %in% bad_models)) %>%
        filter(!(model %in% repeated_models)) %>%
        filter(!(model %in% mcmc_bad_models)) %>%
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
                   truth_df %>% 
                       select(param, model, mean_all, freq_sd_all, n_runs),
                   by=c("param", "model")) %>%
        mutate(z_score=(mean - mean_all) / freq_sd,
               p_val=pnorm(z_score),
               sd_ratio=freq_sd / freq_sd_all)
    
    unique(coverage_df$model) %>% sort()
    coverage_df$p_val %>% length() / coverage_df$p_val %>% unique() %>% length()
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

# A single model column that groups the ARM models will be convenient
coverage_df <-
    coverage_df %>%
    mutate(model_grouping=ifelse(is_arm, "ARM", model))

stopifnot(nrow(coverage_df) > 0)




# There were some repeated p-values due to repeated models
if (FALSE) {
    repeated_p_vals <-
        coverage_df$p_val %>%
        table()
    repeated_p_vals <- repeated_p_vals[repeated_p_vals > 1]
    repeated_p_vals <- names(repeated_p_vals) %>% as.numeric()
    
    length(repeated_p_vals) / nrow(coverage_df)
    coverage_df %>%
        mutate(p_match=p_val - repeated_p_vals[1]) %>%
        filter(abs(p_match) < 1e-5) %>%
        arrange(mean)
    
    filter(coverage_df, abs(p_val - repeated_p_vals[1]) < 1e-8)
}


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

ks_test_param_df <-
    coverage_df %>%
    group_by(num_draws, model, param) %>%
    summarize(ks_test=GetKSPval(p_val), .groups="drop") %>%
    mutate(reject=ks_test < 0.01) %>%
    arrange(num_draws, ks_test)


if (FALSE) {
    filter(ks_test_param_df, reject) %>%
        arrange(model, param, num_draws) %>%
        select(model, param, num_draws, ks_test, reject) %>%
        View()
}



ks_test_df <-
    coverage_df %>%
    group_by(num_draws, model_grouping) %>%
    summarize(ks_test=GetKSPval(p_val), .groups="drop") %>%
    mutate(reject=ks_test < 0.01) %>%
    arrange(num_draws, ks_test)

save_list[["ks_test_param_df"]] <- ks_test_param_df
save_list[["ks_test_df"]] <- ks_test_df
    


############################################################
# Aggregate within a bucket for visualization

bucketed_df <-
    inner_join(
        coverage_df %>%
            group_by(num_draws, model_grouping, p_bucket) %>%
            summarize(bucket_n=n(), .groups="drop"),
        coverage_df %>%
            group_by(num_draws, model_grouping) %>%
            summarize(group_n=n(), .groups="drop"),
        by=c("num_draws", "model_grouping")) %>%
    mutate(p_dens=bucket_n / group_n)
head(bucketed_df)
save_list[["bucketed_df"]] <- bucketed_df

# Sanity check.
p_dens_total <-
    group_by(bucketed_df) %>%
    group_by(num_draws, model_grouping, group_n) %>%
    summarize(s=sum(p_dens), .groups="drop") %>%
    pull(s) %>%
    unique()
stopifnot(p_dens_total == 1)

if (FALSE) {
    ggplot(bucketed_df) +
        geom_line(aes(x=p_bucket, y=n_bins * p_dens, 
                      group=model_grouping, color=model_grouping)) +
        facet_grid(num_draws ~ .) +
        expand_limits(y=0) +
        theme(axis.text.x=element_blank())
}


save_list[["bucketed_df"]] <- bucketed_df

#############################
# Save

output_file <- file.path(output_folder, "coverage_summary.Rdata")
print(output_file)
save(save_list, file=output_file)
