library(gridExtra)
library(tidyverse)

# These test models shouldn't really be in there
bad_models <- c("test", "test_rstanarm")
non_arm_models <- c("potus", "tennis", "microcredit", "occ_det")

# This function is more convenient than always grouping and merging on is_arm
IsARM <- function(model) { !(model %in% non_arm_models) }

base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/dadvi-experiments"
paper_base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/fd-advi-paper"

input_folder <- file.path(base_folder, "comparison/blade_runs/") 
output_folder <- file.path(paper_base_folder, "experiments_data") 

# A list of stuff to be saved for the paper
save_list <- list()

# Load data

raw_posteriors_df <- read.csv(file.path(input_folder, "posteriors_tidy.csv"), as.is=TRUE)
raw_metadata_df <- read.csv(file.path(input_folder, "metadata_tidy.csv"), as.is=TRUE)
raw_trace_df <- read.csv(file.path(input_folder, "trace_tidy.csv"), as.is=TRUE)

num_methods <- length(unique(raw_posteriors_df$method))

metadata_df <-
    raw_metadata_df %>%
    filter(!(model %in% bad_models)) %>%
    mutate(is_arm = IsARM(model),
           time_per_op=runtime / op_count)
save_list[["metadata_df"]] <- metadata_df


non_arm_models <-
    metadata_df %>%
    filter(!is_arm) %>%
    pull(model) %>%
    unique()
save_list[["non_arm_models"]] <- non_arm_models


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


########################################
# Look at the number of draws

metadata_df %>%
    filter(method %in% c("DADVI", "LRVB", "LRVB_Doubling")) %>%
    group_by(method, num_draws) %>%
    summarize(n=n())

# metadata_df %>%
#     filter(method %in% c("DADVI", "LRVB", "LRVB_Doubling")) %>%
#     arrange(model, method, num_draws) %>%
#     select(model, method, num_draws)



########################################
# Save parameter dimensions

param_dims <-
    filter(posteriors_df, method == "DADVI") %>%
    group_by(model, is_arm) %>%
    summarize(param_dim=n(), .groups="drop")
save_list[["param_dims"]] <- param_dims



########################################
# Inspect and categorize the parameters

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


############################################
# Get optimization traces

valid_trace_methods <-
    raw_trace_df %>%
    filter(!is.na(n_calls)) %>%
    pull(method) %>%
    unique()
print(valid_trace_methods)

head(raw_trace_df)

trace_df <-
    filter(raw_trace_df, method %in% valid_trace_methods) %>%
    # group_by(model) %>%
    # mutate(n_call_prop = n_calls / max(n_calls),
    #        obj_value_prop = obj_value - min(obj_value) + 1) %>%
    # ungroup() %>%
    mutate(is_arm=IsARM(model))
save_list[["trace_df"]] <- trace_df


if (FALSE) {
    length(non_arm_models)
    ind <- 4
    ggplot(trace_df %>% filter(model == non_arm_models[ind])) +
        geom_line(aes(x=n_call_prop, y=obj_value_prop, color=method, 
                      group=paste0(method, model))) +
        scale_y_log10() +
        scale_x_log10() +
        geom_hline(aes(yintercept=1)) +
        ggtitle(non_arm_models[ind])

}

final_trace_df <-
    trace_df %>%
    group_by(model, method) %>%
    filter(n_calls == max(n_calls))


final_trace_comp_df <-
    inner_join(filter(final_trace_df, method != "DADVI"),
               filter(final_trace_df, method == "DADVI"),
               by=c("model", "is_arm"), suffix=c("", "_dadvi")) %>%
    mutate(n_calls_vs_dadvi=n_calls / n_calls_dadvi,
           obj_value_vs_dadvi=obj_value - obj_value_dadvi)
save_list[["final_trace_comp_df"]] <- final_trace_comp_df

if (FALSE) {
    View(final_trace_comp_df %>% filter(!is_arm))

    ggplot(final_trace_comp_df) +
        geom_point(aes(x=n_calls_vs_dadvi, y=obj_value_vs_dadvi, color=method)) +
        facet_grid(~ method) +
        scale_x_log10()
}

# Compare the timem to termination (if not convergence)
runtime_comp_df <-
    inner_join(filter(metadata_df, method != "DADVI") %>% 
                   select(method, model, runtime, op_count, time_per_op, converged),
               filter(metadata_df, method == "DADVI") %>% 
                   select(method, model, runtime, op_count, time_per_op, converged),
               by=c("model"), suffix=c("", "_dadvi")) %>%
    mutate(runtime_vs_dadvi=runtime / runtime_dadvi,
           op_count_vs_dadvi=op_count / op_count_dadvi,
           is_arm=IsARM(model))

if (FALSE) {
    metadata_df %>%
        filter(method %in% c("DADVI", "RAABBVI", "SADVI", "SADVI_FR"),
               is_arm) %>%
        ggplot() +
            geom_histogram(aes(x=time_per_op, fill=method)) +
            facet_grid(method ~ .) +
            scale_x_log10() +
            ggtitle("Runtime per model evaluation (ARM only)")

    metadata_df %>%
        filter(method %in% c("DADVI", "RAABBVI", "SADVI", "SADVI_FR"),
               !is_arm) %>%
        select(model, method, time_per_op, runtime, op_count) %>%
        arrange(model,  method)
}

head(runtime_comp_df)
runtime_comp_df$method %>% unique()

if (FALSE) {
    comp_methods <- c("NUTS", "RAABBVI", "SADVI", "SADVI_FR")
    runtime_plot <-
        runtime_comp_df %>%
            filter(method %in% comp_methods, is_arm) %>%
            ggplot() + 
                geom_histogram(aes(x=runtime_vs_dadvi, fill=method)) +
                facet_grid(method ~ .) +
                scale_x_log10() +
                xlab("Runtime / DADVI runtime") +
                expand_limits(x=1)

    op_plot <-
        runtime_comp_df %>%
            filter(method %in% comp_methods, is_arm) %>%
            ggplot() + 
                geom_histogram(aes(x=op_count_vs_dadvi, fill=method)) +
                facet_grid(method ~ .) +
                scale_x_log10() +
                xlab("Model evaluations / DADVI model evluations") +
                expand_limits(x=1)
    grid.arrange(runtime_plot, op_plot, ncol=2)

    
    ggplot(runtime_comp_df %>% filter(is_arm)) +
        geom_point(aes(x=op_count, y=runtime, color=method)) +
        scale_x_log10() + scale_y_log10()
}

runtime_comp_df %>%
    filter(method %in% c("NUTS", "RAABBVI", "SADVI"), !is_arm) %>%
    select(model, method, runtime, runtime_dadvi, runtime_vs_dadvi) %>%
    arrange(model, method)


if (FALSE) {
    ComputationComparisonGraph <- function(comp_df, col) {
        plt <- ggplot(comp_df) +
            geom_bar(aes(x=method, group=model, fill=method,
                         y={{col}}), stat="Identity") +
            scale_y_log10() +
            theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
            facet_grid( ~ model)
        return(plt)
    }
    grid.arrange(
        runtime_comp_df %>%
            filter(method %in% c("RAABBVI", "SADVI", "SADVI_FR"), !is_arm) %>%
            ComputationComparisonGraph(runtime_vs_dadvi),
        runtime_comp_df %>%
            filter(method %in% c("RAABBVI", "SADVI", "SADVI_FR"), !is_arm) %>%
            ComputationComparisonGraph(op_count_vs_dadvi),
        ncol=1
    )
}



save_list[["runtime_comp_df"]] <- runtime_comp_df



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
               by=c("model", "param", "ind", "is_arm"),
               suffix=c("", "_ref")) %>%
    mutate(mean_z_err=(mean - mean_ref) / sd_ref,
           sd_rel_err=(sd - sd_ref) / sd_ref) %>%
    mutate(is_arm=IsARM(model),
           param_ind=paste0(param, ind)) %>% # Easier to count distinct parameters  
    inner_join(param_df, by=c("model", "param", "is_arm")) %>%
    filter(report_param)
save_list[["results_df"]] <- results_df

# Sanity check for bad reference values
filter(results_df, sd_ref < 1e-6) %>% 
    select(method, model, param, ind, mean, sd, mean_ref, sd_ref)


########################################
# Aggregate and compare results

GetMethodComparisonDf <- function(results_df, method1, method2, group_cols) {
    # group_cols should be a string vector (for inner_join)

    stopifnot(method1 %in% unique(results_df$method))
    stopifnot(method2 %in% unique(results_df$method))

    # Note that we remove zero sd_ref here, but we should not assume
    # that a datapoint is bad just because sd_ref is zero.
    # Rather, we should make sure above that we're not
    # accidentally hiding real mistakes.
    agg_results_df <-
        results_df %>%
        filter(sd_ref > 1e-6) %>%
        group_by(across({{group_cols}}), method) %>%
        summarise(mean_z_rmse=sqrt(mean(mean_z_err^2)),
                  sd_rel_rmse=sqrt(mean(sd_rel_err^2)),
                  .groups="drop")
    stopifnot(!any(c(
        any(is.na(agg_results_df$mean_z_rmse)),
        any(is.na(agg_results_df$sd_rel_rmse))
    )))
    
    comp_df <-
        inner_join(filter(agg_results_df, method == !!method1),
                   filter(agg_results_df, method == !!method2),
                   suffix=c("_1", "_2"),
                   by=group_cols) %>%
        mutate(comparison=paste0(method1, " vs ", method2))
    return(comp_df)    
}

arm_group_cols <- c("model", "param")
arm_df <-
    bind_rows(
        results_df %>%
            filter(is_arm) %>%
            GetMethodComparisonDf("LRVB", "SADVI", 
                                  group_cols=arm_group_cols),
        results_df %>%
            filter(is_arm) %>%
            GetMethodComparisonDf("LRVB", "RAABBVI",
                                  group_cols=arm_group_cols),
        results_df %>%
            filter(is_arm) %>%
            GetMethodComparisonDf("LRVB", "SADVI_FR",
                                  group_cols=arm_group_cols)
    ) 
#%>%    mutate(re_label=ifelse(is_re, "Random effect", "Fixed effect"))
save_list[["arm_df"]] <- arm_df

nonarm_group_cols <- c("model", "param")
nonarm_df <-
    bind_rows(
        results_df %>%
            filter(!is_arm) %>%
            GetMethodComparisonDf("LRVB", "SADVI", 
                                  group_cols=nonarm_group_cols),
        results_df %>%
            filter(!is_arm) %>%
            GetMethodComparisonDf("LRVB", "RAABBVI",
                                  group_cols=nonarm_group_cols)
    )
save_list[["nonarm_df"]] <- nonarm_df

if (FALSE) {
    # What models does DADVI do badly on?
    threshold <- 0.4
    dadvi_bad_models <-
        # filter(arm_df, mean_z_rmse_1 > threshold & 
        #                mean_z_rmse_2 < threshold) %>%
        filter(arm_df, mean_z_rmse_1 > threshold) %>%
        pull(model) %>%
        unique()
    
    print(dadvi_bad_models)

    filter(results_df, model %in% dadvi_bad_models) %>% 
        select(model, param, ind, method, mean, mean_ref, mean_z_err) %>%
        arrange(model, param, ind, method) %>%
        View()

    metadata_df %>% 
        filter(model %in% dadvi_bad_models) %>% 
        arrange(model, method) %>% View()
}


if (FALSE) {
    # Confirm that the SD relative errors cluster at 1 because
    # MFVB tends to under-estimate variances
    ggplot(results_df) +
        geom_point(aes(x=sd_ref, y=sd)) +
        geom_abline(aes(slope=1, intercept=0)) +
        scale_x_log10() + scale_y_log10() +
        facet_grid(~ method)
}


if (FALSE) {
    # The ARM graph we want
    arm_mean_plot <-
        arm_df %>% 
        ggplot(aes(x=mean_z_rmse_1, y=mean_z_rmse_2)) +
        geom_density2d(size=1.5) +
        geom_point() +
        geom_abline(aes(slope=1, intercept=0)) +
        xlab("DADVI") + ylab("Stochastic VI") +
        facet_grid(comparison ~ ., scales="fixed") +
        scale_x_log10() + scale_y_log10() +
        ggtitle("Mean relative error (ARM)")

    arm_sd_plot <-
        arm_df %>%
        ggplot(aes(x=sd_rel_rmse_1, y=sd_rel_rmse_2)) +
        geom_density2d(size=1.5) +
        geom_point() +
        geom_abline(aes(slope=1, intercept=0)) +
        xlab("DADVI") + ylab("Stochastic VI") +
        facet_grid(comparison ~ ., scales="fixed") +
        scale_x_log10() + scale_y_log10() +
        ggtitle("SD relative error (ARM)")
    
    grid.arrange(
        arm_mean_plot, arm_sd_plot,
        ncol=2
    )
    
    nonarm_mean_plot <-
        nonarm_df %>% 
        ggplot(aes(x=mean_z_rmse_1, y=mean_z_rmse_2)) +
        #geom_density2d() +
        geom_point(aes(shape=model, color=model), size=4) +
        scale_shape(solid=TRUE) +
        geom_abline(aes(slope=1, intercept=0)) +
        xlab("DADVI") + ylab("Stochastic VI") +
        facet_grid(comparison ~ ., scales="fixed") +
        scale_x_log10() + scale_y_log10() +
        ggtitle("Mean relative error (non-ARM)")
    
    nonarm_sd_plot <-
        nonarm_df %>% 
        ggplot(aes(x=sd_rel_rmse_1, y=sd_rel_rmse_2)) +
        #geom_density2d() +
        geom_point(aes(shape=model, color=model), size=4) +
        scale_shape(solid=TRUE) +
        geom_abline(aes(slope=1, intercept=0)) +
        xlab("DADVI") + ylab("Stochastic VI") +
        facet_grid(comparison ~ ., scales="fixed") +
        scale_x_log10() + scale_y_log10() +
        ggtitle("SD relative error (non-ARM)")

    grid.arrange(
        nonarm_mean_plot, nonarm_sd_plot,
        ncol=2
    )

}


#############################
# Save

output_file <- file.path(output_folder, "posterior_summary.Rdata")
print(output_file)
save(save_list, file=output_file)
