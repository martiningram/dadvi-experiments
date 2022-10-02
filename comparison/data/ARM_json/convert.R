library(jsonlite)
library(rstanarm)
library(optparse)

# Converts .data.R to JSON for easy use with python
parser <- OptionParser()
parser <- add_option(parser, c('-i', '--input-file'))
parser <- add_option(parser, c('-o', '--output-file'))
parser <- add_option(parser, c('--allow-overwrite'), default = FALSE,
                     action = 'store_true')
args <- parse_args(parser)

data_file <- args[['input-file']]
output_file <- args[['output-file']]
allow_overwrite <- args[['allow-overwrite']]

# Make sure required arguments have been passed
stopifnot(!is.null(data_file) & !is.null(output_file))

source(data_file)

loaded_vars <- ls()
all_vars <- list()

for (cur_var_name in loaded_vars) {
    if (grepl('config', cur_var_name) | grepl('parser', cur_var_name)) {
        print(paste0('Not fetching variable with name ', cur_var_name,
                     ' since it is suspected to be config, not data.'))
        next
    }
  all_vars[[cur_var_name]] <- get(cur_var_name)
}

exists <- file.exists(output_file)

if (exists) {
    stopifnot(allow_overwrite)
}

write(toJSON(all_vars), output_file)
