library(gridExtra)
library(tidyverse)

base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/dadvi-experiments"
paper_base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/fd-advi-paper"
analysis_folder <- file.path(base_folder, "comparison/analysis")

source(file.path(analysis_folder, "load_tidy_lib.R"))

input_folder <- file.path(base_folder, "comparison/analysis/coverage_warm_starts_rerun/") 
output_folder <- file.path(paper_base_folder, "experiments_data") 

models_to_remove <- GetModelsToRemove()
non_arm_models <- GetNonARMModels()

raw_coverage_df <-
    read.csv(file.path(input_folder, "coverage_tidy.csv"), as.is=TRUE) %>%
    filter(!(model %in% models_to_remove)) %>%
    mutate(is_arm=IsARM(model)) %>%
    mutate(method="inverse")

raw_coverage_cg_df <-
    read.csv(file.path(input_folder, "coverage_tidy_cg.csv"), as.is=TRUE) %>%
    mutate(is_arm=FALSE) %>%
    mutate(model=recode(model, occu="occ_det")) %>%
    mutate(method="CG")

cg_models <- raw_coverage_cg_df$model %>% unique()

coverage_comb_df <-
    bind_rows(
        raw_coverage_cg_df,
        raw_coverage_df %>% filter(!(model %in% cg_models))
    )
head(coverage_comb_df)

# A list of stuff to be saved for the paper
save_list <- list()


#########################################
# Compute a ground truth and p values



# Compare to the average within the runs with the max number of draws.
truth_df <-     
    coverage_comb_df %>%
    filter(num_draws==max(num_draws)) %>%
    group_by(param, model, num_draws) %>%
    summarize(mean_all=mean(mean),
              n_runs=n(),
              freq_sd_all=sqrt(mean(freq_sd^2) / n_runs),
              .groups="drop")

filter(truth_df, model == "tennis")

coverage_df <-
    inner_join(coverage_comb_df,
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
# Check for p-value uniformity


# Bin and look for p-value uniformity
n_bins <- 100
p_breaks <- seq(0, 1, 1/n_bins)
stopifnot(length(p_breaks) == n_bins + 1)
# Time consuming for some reason
coverage_df <-
    coverage_df %>%
    mutate(p_bucket=cut(p_val, p_breaks))


bucketed_df <-
    inner_join(
        coverage_df %>%
            group_by(num_draws, model_grouping, p_bucket) %>%
            summarize(bucket_n=n(), .groups="drop") %>%
            group_by(model_grouping, num_draws) %>%
            complete(p_bucket, fill=list(bucket_n=0)),
        coverage_df %>%
            group_by(num_draws, model_grouping) %>%
            summarize(group_n=n(), .groups="drop"),
        by=c("num_draws", "model_grouping")) %>%
    mutate(p_dens=bucket_n / group_n)
head(bucketed_df)

bucketed_df %>%
  group_by(model_grouping, num_draws) %>%
  summarize(n=n())

save_list[["bucketed_df"]] <- bucketed_df

# Sanity check.
p_dens_total <-
    group_by(bucketed_df) %>%
    group_by(num_draws, model_grouping) %>%
    summarize(s=sum(p_dens), .groups="drop") %>%
    pull(s) %>%
    unique()
stopifnot(p_dens_total == 1)

if (FALSE) {
  num_draws <- unique(bucketed_df$num_draws)
  draw_labels <- sprintf("%d draws", num_draws)
  names(draw_labels) <- num_draws

  bucketed_df %>%
    mutate(model=model_grouping) %>%
    mutate(p_bucket_num=as.numeric(p_bucket)) %>%
    mutate(p_bucket_num=p_bucket_num / max(p_bucket_num)) %>%
    ggplot() +
    geom_hline(aes(yintercept=1), alpha=0.6) +
    geom_line(aes(x=p_bucket_num, y=n_bins * p_dens,
                  group=model_grouping, color=model_grouping)) +
    facet_grid(model_grouping ~ num_draws,
               scales="free") +
    expand_limits(y=0) +
    ylab("Proportion of p-values in bucket times # of buckets") +
    xlab("P-value bucket")
}


save_list[["bucketed_df"]] <- bucketed_df


################################
# Hmm

coverage_df %>%
    filter(method == "CG") %>%
    #select(mean, mean_all, freq_sd, freq_sd_all, z_score, p_val, p_bucket, model) %>%
    group_by(model) %>%
    arrange(p_val) %>%
    mutate(n=1:n() / n()) %>%
    ggplot() +
        geom_point(aes(x=n, y=p_val, group=model, color=model)) +
    ggtitle(sprintf("CG methods, Num draws = %d", num_draws)) +
    facet_grid(~ num_draws)


# Sanity check
if (FALSE) {
  bind_cols(
    coverage_df %>% filter(model == "occ_det", num_draws == 32) %>% 
      arrange(p_val) %>% select(p_val) %>% rename(p1=p_val),
    coverage_df %>% filter(model == "occ_det", num_draws == 64) %>% 
      arrange(p_val) %>% select(p_val) %>% rename(p2=p_val)) %>%
    ggplot() + geom_point(aes(x=p1, y=p2))
}


#############################
# Save

output_file <- file.path(output_folder, "coverage_summary.Rdata")
print(output_file)
save(save_list, file=output_file)




############################################################
# KS test is not really valid since within a model results
# are correlated

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

# save_list[["ks_test_param_df"]] <- ks_test_param_df
# save_list[["ks_test_df"]] <- ks_test_df
