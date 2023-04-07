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
#######################################################
#######################################################
#######################################################
save(posteriors_df, metadata_df, param_df, lrvb_methods_df, file="/tmp/foo.Rdata")
















#######################################################
#######################################################
#######################################################
#######################################################
# Compare final runtimes (not the optimization traces)



if (FALSE) {
    # Sanity check: look at the average runtime per operation.
    # If runtime is dominated by model evaluation, as we expect it
    # to be, this should not vary radically by method.

    metadata_df %>%
        filter(method %in% c("DADVI", "RAABBVI", "SADVI", "SADVI_FR", "LRVB"),
               is_arm) %>%
        ggplot() +
            geom_histogram(aes(x=time_per_op, fill=method)) +
            facet_grid(method ~ .) +
            scale_x_log10() +
            ggtitle("Runtime per model evaluation (ARM only)")

    metadata_df %>%
        filter(method %in% c("DADVI", "RAABBVI", "SADVI", "SADVI_FR", "LRVB_CG", "LRVB"),
               !is_arm) %>%
        select(model, method, time_per_op, runtime, op_count) %>%
        arrange(model,  method)

    metadata_df %>%
        filter(method %in% c("DADVI", "RAABBVI", "SADVI", "SADVI_FR", "LRVB_CG", "LRVB"),
               !is_arm) %>%
        select(model, method, time_per_op, runtime, op_count) %>%
        ggplot() +
            geom_bar(aes(x=model, y=time_per_op, fill=method, group=method),
                     position=position_dodge(), stat="Identity")

}



# The way we compute LRVB is use HVP evaluations once per dimension.
# But we see four.  Why?  Probably because there are four blocks of the VB
# parameters, and an HVP is on the model, not the VB objective.
metadata_df %>%
    filter(method == "LRVB") %>%
    select(model, method, op_count, dim, num_draws) %>%
    mutate(ops_per_dim_per_draw = op_count / (dim * num_draws)) %>%
    arrange(desc(ops_per_dim_per_draw))

# Compare to reference methods: DADVI and DADVI + LRVB

dadvi_runtimes_df <-
    metadata_df %>%
    select(method, model, runtime, op_count) %>%
    filter(method == "DADVI")



lrvb_runtimes_df <-
    metadata_df %>%
    select(method, model, runtime, op_count) %>%
    inner_join(lrvb_methods_df, by=c("model", "method")) %>%
    mutate(method="LR") %>%
    inner_join(dadvi_runtimes_df, suffix=c("", "_DADVI"), by=c("model")) %>%
    mutate(runtime=runtime + runtime_DADVI,
           op_count=op_count + op_count_DADVI) %>%
    select(-runtime_DADVI, -op_count_DADVI, -method_DADVI)
# %>%
#     pivot_wider(id_cols="model", names_from=method, values_from=c(op_count, runtime))


# Compare the time to termination (if not convergence)
# NUTS op count isn't counted correctly
comp_methods <- c("NUTS", "RAABBVI", "SADVI", "SADVI_FR")
runtime_comp_df <-
    filter(metadata_df, method %in% comp_methods) %>%
    select(method, model, runtime, op_count, time_per_op, converged) %>%
    inner_join(dadvi_runtimes_df, by=c("model"), suffix=c("", "_dadvi")) %>%
    mutate(runtime=case_when(method == "LRVB" ~ runtime + runtime_dadvi,
                             method == "LRVB_CG" ~ runtime + runtime_dadvi,
                             TRUE ~ runtime),
           op_count=case_when(method == "LRVB" ~ op_count + op_count_dadvi,
                              method == "LRVB_CG" ~ op_count + op_count_dadvi,
                              method == "NUTS" ~ as.numeric(NA),
                              TRUE ~ op_count)) %>%
    inner_join(lrvb_runtimes_df, by=c("model"), suffix=c("", "_lr")) %>%
    mutate(runtime_vs_dadvi=runtime / runtime_dadvi,
           op_count_vs_dadvi=op_count / op_count_dadvi,
           runtime_vs_lrvb=runtime / runtime_lr,
           op_count_vs_lrvb=op_count / op_count_lr,
           is_arm=IsARM(model))

head(runtime_comp_df)
runtime_comp_df$method %>% unique()

if (FALSE) {
    # Compare computational effort to a DADVI baseline
    ComputationComparisonHistogramGraph <- function(comp_df, col) {
        plt <- ggplot(comp_df) +
            geom_histogram(aes(x={{col}}, fill=method), bins=30) +
            facet_grid(method ~ .) +
            geom_vline(aes(xintercept=1)) +
            scale_x_log10() +
            expand_limits(x=1)
        return(plt)
    }

    comp_methods <- c("NUTS", "RAABBVI", "SADVI", "SADVI_FR")
    runtime_arm_df <-runtime_comp_df %>%
        filter(method %in% comp_methods, is_arm)

    runtime_dadvi_plot <-
        ComputationComparisonHistogramGraph(runtime_arm_df, runtime_vs_dadvi) +
        xlab("Runtime / DADVI runtime")

    op_dadvi_plot <-
        ComputationComparisonHistogramGraph(runtime_arm_df, op_count_vs_dadvi) +
        xlab("Model evaluations / DADVI model evaluations")

    runtime_lrvb_plot <-
        ComputationComparisonHistogramGraph(runtime_arm_df, runtime_vs_lrvb) +
        xlab("Runtime / LRVB runtime")

    op_lrvb_plot <-
        ComputationComparisonHistogramGraph(runtime_arm_df, op_count_vs_lrvb) +
        xlab("Model evaluations / LRVB model evaluations")

    grid.arrange(runtime_dadvi_plot, op_dadvi_plot, runtime_lrvb_plot, op_lrvb_plot, ncol=2)

    # Look at the big models using a different visualization
    runtime_comp_df %>%
        filter(method %in% c("NUTS", "RAABBVI", "SADVI", "LRVB", "LRVB_CG"), !is_arm) %>%
        select(model, method, runtime, runtime_dadvi, runtime_vs_dadvi, runtime_lr, runtime_vs_lrvb) %>%
        arrange(model, method)

    ComputationComparisonGraph <- function(comp_df, col) {
        plt <- ggplot(comp_df) +
            geom_bar(aes(x=method, group=model, fill=method,
                         y={{col}}), stat="Identity") +
            scale_y_log10() +
            theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
            facet_grid( ~ model)
        return(plt)
    }
    comp_methods <- c("NUTS", "RAABBVI", "SADVI", "SADVI_FR")
    grid.arrange(
        runtime_comp_df %>%
            filter(method %in% comp_methods, !is_arm) %>%
            ComputationComparisonGraph(runtime_vs_dadvi),
        runtime_comp_df %>%
            filter(method %in% comp_methods, !is_arm) %>%
            ComputationComparisonGraph(op_count_vs_dadvi),
        runtime_comp_df %>%
            filter(method %in% comp_methods, !is_arm) %>%
            ComputationComparisonGraph(runtime_vs_lrvb),
        runtime_comp_df %>%
            filter(method %in% comp_methods, !is_arm) %>%
            ComputationComparisonGraph(op_count_vs_lrvb),
        ncol=2
    )
}



save_list[["runtime_comp_df"]] <- runtime_comp_df



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
    filter(!(model %in% models_to_remove)) %>%
    mutate(is_arm=IsARM(model))
save_list[["trace_df"]] <- trace_df

# DADVI doesn't start counting at one :(
trace_df %>%
    group_by(method) %>%
    summarize(min_n_calls=min(n_calls))


trace_scales_df <-
    filter(metadata_df, method == "DADVI") %>%
    select(model, kl_sd) %>%
    rename(obj_value_sd=kl_sd)


# Get a "location" by looking at termination of the DADVI algorithm
trace_offset_df <-
    trace_df %>%
    filter(method == "DADVI") %>%
    group_by(model) %>%
    filter(n_calls == max(n_calls)) %>%
    rename(n_calls_dadvi=n_calls, obj_value_dadvi=obj_value) %>%
    select(model, n_calls_dadvi, obj_value_dadvi)



Cap <- function(x, min=-1e3, max=1e3) {
    return(case_when(x < min ~ min,
                     x > max ~ max,
                     TRUE ~ x))
}

# Compute "normed" objective values for common plotting
# Note!  Here I fix the fact that RAABBVI starts at zero rather than one.
# when this is fixed elsewhere remember to change it here too.
trace_norm_df <-
    trace_df %>%
    mutate(n_calls=case_when(method == "RAABBVI" ~ n_calls + 1, TRUE ~ n_calls)) %>%
    inner_join(trace_scales_df, by="model") %>%
    inner_join(trace_offset_df, by="model") %>%
    mutate(n_calls_norm=n_calls / n_calls_dadvi,
           obj_value_norm=(obj_value - obj_value_dadvi) / obj_value_sd,
           obj_value_norm_cap=Cap(obj_value_norm, min=-Inf, max=1e5))

# Get the termination point of each method
trace_norm_termination_df <-
    trace_norm_df %>%
    group_by(model, method) %>%
    filter(n_calls == max(n_calls))

if (FALSE) {
    View(trace_norm_termination_df)
}

SignedLog10 <- function(x) {
    case_when(x == 0 ~ 0,
              TRUE ~ sign(x) * log10(abs(x)))
}

SignedLog10Transform <- scales::trans_new(
    "signed_log10",
    SignedLog10,
    function(x) { sign(x) * 10^(abs(x)) }
    # breaks=function(b) { 10^round(SignedLog10(b)) },
    # minor_breaks=function(b, limits, n) { c() }
    )

RightLog10 <- function(x) {
    case_when(x < 1 ~ x,
              TRUE ~ log10(x) + 1)
}
RightExp10 <- function(x) {
    case_when(x < 1 ~ x,
              TRUE ~ 10^(x - 1))
}

RightLog10Transform <- scales::trans_new(
    "right_log10",
    RightLog10, RightExp10
)

save_list[["trace_norm_df"]] <- trace_norm_df

if (FALSE) {
    # This is the one!

    PlotTraces <- function(df) {
        trace_norm_termination_df <-
            df %>%
            group_by(model, method) %>%
            filter(n_calls == max(n_calls))
        break_steps <- 10 ^ seq(0, 9)
        breaks <- sort(c(-break_steps, break_steps))
        ggplot(df) +
            geom_line(aes(x=n_calls_norm, y=obj_value_norm, color=method, group=paste0(method, model))) +
            geom_point(aes(x=n_calls_norm, y=obj_value_norm, group=paste0(method, model)),
                       data=trace_norm_termination_df) +
            scale_y_continuous(trans=SignedLog10Transform, breaks=breaks) +
            scale_x_continuous(trans=RightLog10Transform) +
            #scale_x_log10() +
            geom_hline(aes(yintercept=0)) +
            #geom_hline(aes(yintercept=-2), color="dark gray") + # Two DADVI KL standard deviations
            geom_vline(aes(xintercept=1)) +
            xlab("Number of function calls / number of DADVI function calls\n(Values > 1 are log10 transformed)") +
            ylab("(ELBO - DADVI optimal ELBO) / DADVI optimal ELBO standard deviation \n(signed log10 transformed)")
    }

    PlotTraces(trace_norm_df %>% filter(is_arm)) +
        ggtitle("Standardized optimization traces for ARM") +
        facet_grid(method ~ .)

    trace_norm_df %>% filter(!is_arm) %>% mutate(model=as.character(model)) %>% pull(model) %>% class()
    PlotTraces(trace_norm_df %>% filter(!is_arm))  +
        ggtitle("Standardized optimization traces for non-ARM") +
        facet_grid(method ~ model)

}


if (FALSE) {
    # Look at single model / method traces
    trace_models <- unique(trace_norm_df$model)
    ui <- fluidPage(
        numericInput("model_index", "Model index", 1, min=1, max=length(trace_models), step=1),
        plotOutput("plot")
    )

    server <- function(input, output, session) {
        selected_model <- reactive({
            trace_models[input$model_index]
        })
        dataset <- reactive({
            trace_norm_df %>%
                filter(model == selected_model()) %>%
                filter(method == "RAABBVI")
        })
        output$plot <- renderPlot({
            ggplot(dataset()) +
                geom_line(aes(x=n_calls_norm, y=obj_value_norm, group=model)) +
                ggtitle(selected_model()) +
                scale_x_log10() +
                geom_hline(aes(yintercept=0)) +
                geom_vline(aes(xintercept=1)) +
                scale_y_continuous(trans=SignedLog10Transform, breaks=breaks)
        }, res = 96)
    }

    shinyApp(ui, server)
}



########################################
########################################
# Explore a little


# Hmm, something's up with microcredit
posteriors_df %>%
    filter(model=="microcredit", param == "beta_full", ind %in% c(3, 4, 5)) %>%
    select(model, param, ind, mean, sd, method)


########################################
########################################
########################################
########################################
########################################
# Posteriors

posteriors_df %>% pull(method) %>% unique()


########################################
# Compare to a reference method

reference_method <- "NUTS"
stopifnot(sum(posteriors_df$method == reference_method) > 0)

# Separately report each special LR quantity of interest for occ_det and tennis
posteriors_df %>%
    filter(model %in% c("tennis", "occ_det"), method == "DADVI") %>%
    group_by(model, param) %>%
    summarize(n=n())

results_all_df <-
    inner_join(posteriors_df %>% filter(method != reference_method),
               posteriors_df %>% filter(method == reference_method),
               by=c("model", "param", "ind", "is_arm"),
               suffix=c("", "_ref")) %>%
    mutate(mean_z_err=(mean - mean_ref) / sd_ref,
           sd_rel_err=(sd - sd_ref) / sd_ref) %>%
    mutate(is_arm=IsARM(model),
           param_ind=paste0(param, ind)) %>% # Easier to count distinct parameters
    inner_join(param_df, by=c("model", "param", "is_arm")) %>%
    filter(report_param) %>%
    mutate(param=case_when(
        (model == "tennis") & (param == "match_predictions") ~ paste(param, ind, sep="_"),
        (model == "occ_det") & (param == "presence_prediction") ~ paste(param, ind, sep="_"),
        TRUE ~ param
    ))

# Use LRVB_CG or LRVB as the "LR" method
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

if (FALSE) {
    # Visually inspect which methods are missing for which models
    results_df %>%
        group_by(model, method) %>%
        summarize(n=1) %>%
        pivot_wider(id_cols=model, names_from=method, values_from=n) %>%
        View()
}

save_list[["results_df"]] <- results_df

# Sanity check for bad reference values
filter(results_df, sd_ref < 1e-6) %>%
    select(method, model, param, ind, mean, sd, mean_ref, sd_ref)



########################################
# Look at LR-CG alone

results_df$method %>% unique()

results_df %>%
    filter(old_method=="LRVB_CG") %>%
    select(old_method, param, ind, mean, sd, mean_ref, sd_ref)


########################################
# Aggregate and compare posterior results

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
            GetMethodComparisonDf("DADVI", "SADVI",
                                  group_cols=all_of(arm_group_cols)),
        results_df %>%
            filter(is_arm) %>%
            GetMethodComparisonDf("DADVI", "RAABBVI",
                                  group_cols=all_of(arm_group_cols)),
        results_df %>%
            filter(is_arm) %>%
            GetMethodComparisonDf("DADVI", "SADVI_FR",
                                  group_cols=all_of(arm_group_cols))
    )

arm_lr_df <-
    bind_rows(
        results_df %>%
            filter(is_arm) %>%
            GetMethodComparisonDf("LR", "SADVI",
                                  group_cols=all_of(arm_group_cols)),
        results_df %>%
            filter(is_arm) %>%
            GetMethodComparisonDf("LR", "RAABBVI",
                                  group_cols=all_of(arm_group_cols)),
        results_df %>%
            filter(is_arm) %>%
            GetMethodComparisonDf("LR", "SADVI_FR",
                                  group_cols=all_of(arm_group_cols))
    )
save_list[["arm_df"]] <- arm_df
save_list[["arm_lr_df"]] <- arm_lr_df

nonarm_group_cols <- c("model", "param")
nonarm_df <-
    bind_rows(
        results_df %>%
            filter(!is_arm) %>%
            GetMethodComparisonDf("DADVI", "SADVI",
                                  group_cols=all_of(nonarm_group_cols)),
        results_df %>%
            filter(!is_arm) %>%
            GetMethodComparisonDf("DADVI", "RAABBVI",
                                  group_cols=all_of(nonarm_group_cols)),
        results_df %>%
            filter(!is_arm) %>%
            GetMethodComparisonDf("DADVI", "SADVI_FR",
                                  group_cols=all_of(nonarm_group_cols))
    )

nonarm_lr_df <-
    bind_rows(
        results_df %>%
            filter(!is_arm) %>%
            GetMethodComparisonDf("LR", "SADVI",
                                  group_cols=all_of(nonarm_group_cols)),
        results_df %>%
            filter(!is_arm) %>%
            GetMethodComparisonDf("LR", "RAABBVI",
                                  group_cols=all_of(nonarm_group_cols)),
        results_df %>%
            filter(!is_arm) %>%
            GetMethodComparisonDf("LR", "SADVI_FR",
                                  group_cols=all_of(nonarm_group_cols))
    )


results_df %>%
    filter(model == "occ_det") %>%
    pull(param) %>% unique()

save_list[["nonarm_df"]] <- nonarm_lr_df
save_list[["nonarm_lr_df"]] <- nonarm_lr_df


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
        arm_lr_df %>%
        ggplot(aes(x=sd_rel_rmse_1, y=sd_rel_rmse_2)) +
        geom_density2d(size=1.5) +
        geom_point() +
        geom_abline(aes(slope=1, intercept=0)) +
        xlab("LR") + ylab("Stochastic VI") +
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
        nonarm_lr_df %>%
        ggplot(aes(x=sd_rel_rmse_1, y=sd_rel_rmse_2)) +
        #geom_density2d() +
        geom_point(aes(shape=model, color=model), size=4) +
        scale_shape(solid=TRUE) +
        geom_abline(aes(slope=1, intercept=0)) +
        xlab("LR") + ylab("Stochastic VI") +
        facet_grid(comparison ~ ., scales="fixed") +
        scale_x_log10() + scale_y_log10() +
        ggtitle("SD relative error (non-ARM)")

    grid.arrange(
        nonarm_mean_plot, nonarm_sd_plot,
        ncol=2
    )


}


##################################################################################
# Save.  For now let's only do this manually to avoid overwriting accidentally

if (FALSE) {
    output_file <- file.path(output_folder, "posterior_summary.Rdata")
    print(output_file)
    save(save_list, file=output_file)
}



# Shiny?



datasets <- c("eurodist", "faithful", "sleep")
ui <- fluidPage(
    selectInput("dataset", "Dataset", choices = datasets),
    verbatimTextOutput("summary"),
    plotOutput("plot")
)

server <- function(input, output, session) {
    dataset <- reactive({
        get(input$dataset, "package:datasets")
    })
    output$summary <- renderPrint({
        summary(dataset())
    })
    output$plot <- renderPlot({
        plot(dataset())
    }, res = 96)
}

shinyApp(ui, server)



#########################


models <- unique(trace_df$model)
ui <- fluidPage(
    numericInput("model_index", "Model index", 1, min=1, max=length(models), step=1),
    plotOutput("plot")
)

server <- function(input, output, session) {
    selected_model <- reactive({
        models[input$model_index]
    })
    dataset <- reactive({
        trace_df %>% filter(model == selected_model())
    })
    output$plot <- renderPlot({
        ggplot(dataset()) +
            geom_line(aes(x=n_calls, y=obj_value, color=method)) +
            ggtitle(selected_model())
    }, res = 96)
}

shinyApp(ui, server)













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

