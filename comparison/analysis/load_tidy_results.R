library(tidyverse)

base_folder <- "/home/rgiordan/Documents/git_repos/DADVI/dadvi-experiments"
input_folder <- file.path(base_folder, "comparison/blade_runs/") 

posteriors_df <- read.csv(file.path(input_folder, "posteriors_tidy.csv"), as.is=TRUE)
metadata_df <- read.csv(file.path(input_folder, "metadata_tidy.csv"), as.is=TRUE)

num_methods <- length(unique(posteriors_df$method))

# Check for models that didn't run with all methods
model_methods <-
    group_by(posteriors_df, model, method) %>%
    summarise(.groups="drop") 

bad_models <- 
    model_methods %>%
    group_by(model) %>%
    summarize(n=n(), .groups="drop") %>%
    filter(n <  num_methods) %>%
    pull(model)

# See which ones are missing
filter(model_methods, model %in% bad_models) %>%
    mutate(value=TRUE) %>%
    pivot_wider(id_cols=model, names_from=method)


results_df <-
    filter(posteriors_df, !(model %in% bad_models)) %>%
    inner_join(metadata_df, by=c("method", "model"))

head(results_df)

# We might be able to indentify random effects by the vertical bar
unique(results_df$param)


