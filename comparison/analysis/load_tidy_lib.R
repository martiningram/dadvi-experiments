GetModelsToRemove <- function() {
    # These test models shouldn't really be in there
    bad_models <- c("test", "test_rstanarm")

    # I think these models are just modified versions of another model.  We should
    # remove them systematically.
    repeated_models <- c(
        "radon_group_chr", "radon_intercept_chr", "radon_no_pool_chr",
        "wells_predicted", "mesquite_va")
    
    # These models didn't work with NUTS well enough to use here.
    mcmc_bad_models <- c("earnings_latin_square", "earnings_vary_si", "election88_full")
    
    models_to_remove <- c(bad_models, repeated_models, mcmc_bad_models) %>% unique()
    
    return(models_to_remove)
}


GetNonARMModels <- function() {
    non_arm_models <- c("potus", "tennis", "microcredit", "occ_det")
    return(non_arm_models)
}

# This function is more convenient than always grouping and merging on is_arm
IsARM <- function(model) { !(model %in% non_arm_models) }
