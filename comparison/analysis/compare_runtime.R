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


#######################################################
# Compare final runtimes (not the optimization traces)


# Compare to reference methods: DADVI and DADVI + LRVB
dadvi_runtimes_df <-
    metadata_df %>%
    select(method, model, runtime, op_count) %>%
    filter(method == "DADVI")

# The method "LR" will be "LRVB" or "LRVB_CG" according to lrvb_methods_df
lrvb_runtimes_df <-
    metadata_df %>%
    select(method, model, runtime, op_count) %>%
    inner_join(lrvb_methods_df, by=c("model", "method")) %>%
    mutate(method="LR") %>%
    inner_join(dadvi_runtimes_df, suffix=c("", "_DADVI"), by=c("model")) %>%
    mutate(runtime=runtime + runtime_DADVI,
           op_count=op_count + op_count_DADVI) %>%
    select(-runtime_DADVI, -op_count_DADVI, -method_DADVI)


# Compare the time to termination (if not convergence)
# Note: NUTS op count isn't counted correctly
comp_methods <- c("NUTS", "RAABBVI", "SADVI", "SADVI_FR")
runtime_comp_df <-
    filter(metadata_df, method %in% comp_methods) %>%
    select(method, model, runtime, op_count, time_per_op, converged) %>%
    mutate(op_count=case_when(method == "NUTS" ~ as.numeric(NA),
                              TRUE ~ op_count)) %>%
    inner_join(dadvi_runtimes_df, by=c("model"), suffix=c("", "_dadvi")) %>%
    inner_join(lrvb_runtimes_df, by=c("model"), suffix=c("", "_lr")) %>%
    mutate(runtime_vs_dadvi=runtime / runtime_dadvi,
           op_count_vs_dadvi=op_count / op_count_dadvi,
           runtime_vs_lrvb=runtime / runtime_lr,
           op_count_vs_lrvb=op_count / op_count_lr,
           is_arm=IsARM(model))

head(runtime_comp_df)
runtime_comp_df$method %>% unique()

# Compare computational effort to a DADVI or LR baseline
ComputationComparisonHistogramGraph <- function(comp_df, col) {
    plt <- ggplot(comp_df) +
        geom_histogram(aes(x={{col}}, fill=method), bins=30) +
        facet_grid(method ~ .) +
        geom_vline(aes(xintercept=1)) +
        scale_x_log10() +
        expand_limits(x=1)
    return(plt)
}


runtime_dadvi_plot <-
    runtime_comp_df %>% filter(is_arm) %>%
    ComputationComparisonHistogramGraph(runtime_vs_dadvi) +
    xlab("Runtime / DADVI runtime")

op_dadvi_plot <-
    runtime_comp_df %>% filter(is_arm) %>%
    ComputationComparisonHistogramGraph(op_count_vs_dadvi) +
    xlab("Model evaluations / DADVI model evaluations")

runtime_lrvb_plot <-
    runtime_comp_df %>% filter(is_arm) %>%
    ComputationComparisonHistogramGraph(runtime_vs_lrvb) +
    xlab("Runtime / LRVB runtime")

op_lrvb_plot <-
    runtime_comp_df %>% filter(is_arm) %>%
    ComputationComparisonHistogramGraph(op_count_vs_lrvb) +
    xlab("Model evaluations / LRVB model evaluations")

if (FALSE) {
    grid.arrange(
    runtime_dadvi_plot, 
    op_dadvi_plot, 
    runtime_lrvb_plot, 
    op_lrvb_plot, ncol=2)
}

# Look at the big models using a different visualization
ComputationComparisonBarGraph <- function(comp_df, col) {
    plt <- ggplot(comp_df) +
        geom_bar(aes(x=method, group=model, fill=method,
                     y={{col}}), stat="Identity") +
        scale_y_log10() +
        theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1)) +
        facet_grid( ~ model)
    return(plt)
}

if (FALSE) {
    grid.arrange(
        runtime_comp_df %>% filter(!is_arm) %>%
            ComputationComparisonBarGraph(runtime_vs_dadvi),
        runtime_comp_df %>% filter(!is_arm) %>%
            ComputationComparisonBarGraph(op_count_vs_dadvi),
        runtime_comp_df %>% filter(!is_arm) %>%
            ComputationComparisonBarGraph(runtime_vs_lrvb),
        runtime_comp_df %>% filter(!is_arm) %>%
            ComputationComparisonBarGraph(op_count_vs_lrvb),
        ncol=2)
}



save(runtime_comp_df, file=file.path(output_folder, "runtime.Rdata"))


stop()

##############################################################
# Some sanity checks


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
