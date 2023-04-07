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

load("/tmp/foo.Rdata")



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






