library(gridExtra)
library(tidyverse)
library(shiny)

base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/dadvi-experiments"
paper_base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/fd-advi-paper"
analysis_folder <- file.path(base_folder, "comparison/analysis")

output_folder <- file.path(paper_base_folder, "experiments_data")

source(file.path(analysis_folder, "load_tidy_lib.R"))

models_to_remove <- GetModelsToRemove()
non_arm_models <- GetNonARMModels()

load(file.path(output_folder, "cleaned_experimental_results.Rdata"))


########################################
# Posteriors compared to a reference method

reference_method <- "NUTS"
stopifnot(sum(posteriors_df$method == reference_method) > 0)

# Note that we only have CG results for the match_predictions and presence_prediction.
any(is.na(posteriors_df$is_unconstrained))
results_all_df <-
    inner_join(posteriors_df %>% filter(method != reference_method),
               posteriors_df %>% filter(method == reference_method),
               by=c("model", "param", "ind", "is_arm", "is_unconstrained"),
               suffix=c("", "_ref")) %>%
    mutate(param=case_when(
        (model == "tennis") & (param == "match_predictions") ~ paste(param, ind, sep="_"),
        (model == "occ_det") & (param == "presence_prediction") ~ paste(param, ind, sep="_"),
        TRUE ~ param
    )) %>% # For certain parameters, treat each index as a "parameter" for purposes of reporting accuracy
    mutate(mean_z_err=(mean - mean_ref) / sd_ref,
           sd_rel_err=(sd - sd_ref) / sd_ref) %>%
    mutate(is_arm=IsARM(model),
           param_ind=paste0(param, ind)) # This makes it easier to count distinct parameters
    # filter(report_param) %>%
    # filter(is_unconstrained) # Can't only report unconstrained and also have the LRVB_CG results

# Use LRVB_CG or LRVB as the "LR" method according to lrvb_methods_df
lr_methods <- c("LRVB", "LRVB_CG")
results_lr_df <-
    results_all_df %>%
    filter(method %in% lr_methods) %>%
    inner_join(lrvb_methods_df, by=c("model", "method")) %>%
    mutate(old_method=method) %>%
    mutate(method="LR")

results_df <-
    bind_rows(results_lr_df,
              filter(results_all_df, !(method %in% lr_methods)))

# Sanity check for bad reference values
filter(results_df, sd_ref < 1e-6) %>%
    select(method, model, param, ind, mean, sd, mean_ref, sd_ref)

# # Everything needs an LR method
# filter(results_df, method == "LR") %>% filter(!is_arm) %>% pull(model) %>% unique()
# 
# filter(posteriors_df) %>% filter(!is_arm) %>% select(model, method) %>% unique() %>% filter(method %in% lr_methods)
# filter(results_df) %>% filter(!is_arm) %>% select(model, method) %>% unique()
# 
# filter(lrvb_methods_df, model %in% non_arm_models) %>%  select(model, method) %>% unique()
# filter(posteriors_df, model %in% non_arm_models) %>%  select(model, method) %>% unique() %>% filter(method %in% lr_methods)
# filter(results_lr_df, model %in% non_arm_models) %>%  select(model, method) %>% unique()
# filter(results_all_df, model %in% non_arm_models) %>%  select(model, method) %>% unique() %>% filter(method %in% lr_methods)

########################################
# Aggregate and compare posterior results

# Compute aggregated errors between two methods
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



# Call GetMethodComparisonDf for multiple methods and rbind the dataframes.
GetMethodComparisonsDf <- function(results_df, dadvi_methods, comp_methods, group_cols) {
    result_list <- list()
    for (dadvi_method in dadvi_methods) {
        for (comp_method in comp_methods) {
            result_list[[length(result_list) + 1]] <-
                GetMethodComparisonDf(
                    results_df, dadvi_method, comp_method, group_cols=all_of(group_cols))
        }
    }
    do.call(bind_rows, result_list)
}    


posterior_comp_df <- results_df %>%
    GetMethodComparisonsDf(c("DADVI", "LR"), 
                           c("SADVI", "RAABBVI", "SADVI_FR"), 
                           c("model", "param", "is_arm"))


PlotPostComparison <- function(comp_df, col1, col2, model_label=FALSE, same_lims=TRUE, plot_dens=TRUE) {
    lims <- max(pull(comp_df, {{col1}}), pull(comp_df, {{col2}}))
    
    if (model_label) {
        plt <-
            comp_df %>%
            ggplot(aes(x={{col1}}, y={{col2}})) +
            geom_point(aes(color=model, shape=model), size=4) +
            scale_shape(solid=TRUE)
    } else {
        plt <-
            comp_df %>%
            ggplot(aes(x={{col1}}, y={{col2}})) +
            geom_point()
    }
    plt <- plt +
        geom_abline(aes(slope=1, intercept=0)) +
        facet_grid(comparison ~ ., scales="fixed") +
        scale_x_log10() + scale_y_log10()
    
    if (plot_dens) {
        plt <- plt + geom_density2d(size=1.5)
    }
    
    if (same_lims) {
        plt <- plt + expand_limits(x=lims, y=lims)
    }
    return(plt)
}


arm_mean_plot <-
    posterior_comp_df %>%
    filter(is_arm, method_1 == "DADVI") %>%
    PlotPostComparison(mean_z_rmse_1, mean_z_rmse_2) +
    xlab("DADVI") + ylab("Stochastic VI") +
    ggtitle("Mean relative error (ARM)")

arm_sd_plot <-
    posterior_comp_df %>%
    filter(is_arm, method_1 == "LR") %>%
    PlotPostComparison(sd_rel_rmse_1, sd_rel_rmse_2) +
    xlab("LR") + ylab("Stochastic VI") +
    ggtitle("SD relative error (ARM)")

if (FALSE) {
    grid.arrange(
        arm_mean_plot, arm_sd_plot,
        ncol=2
    )
}

nonarm_mean_plot <-
    posterior_comp_df %>%
    filter(!is_arm, method_1 == "DADVI") %>%
    PlotPostComparison(mean_z_rmse_1, mean_z_rmse_2, plot_dens=FALSE, model_label=TRUE) +
    xlab("DADVI") + ylab("Stochastic VI") +
    ggtitle("Mean relative error (non-ARM)")

nonarm_sd_plot <-
    posterior_comp_df %>%
    filter(!is_arm, method_1 == "LR") %>%
    PlotPostComparison(sd_rel_rmse_1, sd_rel_rmse_2, plot_dens=FALSE, model_label=TRUE) +
    xlab("LR") + ylab("Stochastic VI") +
    ggtitle("SD relative error (non-ARM)")

if (FALSE) {
    grid.arrange(
        nonarm_mean_plot, nonarm_sd_plot,
        ncol=2
    )
}

save(posterior_comp_df, file=file.path(output_folder, "posteriors.Rdata"))




##################################################################################
##################################################################################
##################################################################################
##################################################################################
# Some extra sanity checking

stop()









########################################
# What's going on with tennis?

tennis_results_df <-
    filter(results_df, model == "tennis") %>%
    select(method, param, ind, mean, sd, mean_ref, sd_ref)

tennis_results_df %>%
    group_by(method, param) %>%
    summarize(n=n())

if (FALSE) {
    grid.arrange(
        ggplot(tennis_results_df %>% filter(param == "player_skills")) +
            geom_point(aes(x=mean_ref, y=mean, color=method)) +
            geom_abline(aes(slope=1, intercept=0)) +
            xlab("NUTS mean")
        ,
        ggplot(tennis_results_df %>% filter(param == "player_skills")) +
            geom_point(aes(x=sd_ref, y=sd, color=method)) +
            geom_abline(aes(slope=1, intercept=0)) +
            xlab("NUTS sd")
        , ncol=2
    )
}

print(tennis_results_df)

tennis_results_df %>%
    group_by(param, method) %>%
    summarize(mean_mse=sqrt(mean(mean / sd_ref - mean_ref / sd_ref)^2),
              sd_mse=sqrt(mean(sd / sd_ref - 1)^2))



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




#####################################################################

# Separately report each special LR quantity of interest for occ_det and tennis
posteriors_df %>%
    filter(model %in% c("tennis", "occ_det"), method == "DADVI") %>%
    group_by(model, param) %>%
    summarize(n=n())

#####################################################################

if (FALSE) {
    # Visually inspect which methods are missing for which models
    results_df %>%
        group_by(model, method) %>%
        summarize(n=1) %>%
        pivot_wider(id_cols=model, names_from=method, values_from=n) %>%
        View()
}




#########################################################################################
# Hmm, something's up with microcredit.  The model was just defined a little strangely.


posteriors_df %>%
    filter(model=="microcredit", param == "beta_full", ind %in% c(3, 4, 5)) %>%
    select(model, param, ind, mean, sd, method)




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

