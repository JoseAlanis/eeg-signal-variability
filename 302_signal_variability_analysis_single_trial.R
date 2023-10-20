# Title:  More than noise?: Signal variability measures, single trial analysis
# Author: José C. García Alanis
# Date:   Fri Oct 20 11:08:12 2023
# Notes:  R script can be ran as job on HPC. Takes args `sensor_n`, `task_i`,
#         and `jobs`

# get utility functions --------------------------------------------------------
source('utils.R')

load.package(
  c('foreach', 'optparse',
    'rjson', 'dplyr', 'tidyr', 'stringr',
    'afex', 'optimx',
    'performance', 'emmeans', 'effectsize')
)

# parse arguments --------------------------------------------------------------
option_list <- list(
  make_option("--sensor_n",
              type = "integer",
              help = "Which sensor should be analysed? (can be 0 to 31)",
              action = "store",
              default = 1
  ),
  make_option("--task_i",
              type = "character",
              help = "Which task should be analysed? (e.g., 'Odd/Even')",
              action = "store",
              default = "Odd/Even"
  ),
  make_option("--jobs",
              type = "integer",
              help = "How many cores should be used in parallel?",
              action = "store",
              default = 1
  )
)

# save all the options variables in a list -------------------------------------
opt <- parse_args(
  OptionParser(option_list = option_list)
)
sensor_n <- opt$sensor_n
task_i <- opt$task_i
jobs <- opt$jobs

# parallel settings ------------------------------------------------------------
n.cores <- jobs
my.cluster <- parallel::makeCluster(
  n.cores,
  type = "PSOCK"
)

# register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)

# check if it is registered (optional)
# foreach::getDoParRegistered()

# function for combining output of foreach
comb <- function(List1, List2) {
  output_a <- c(List1[[1]], List2[[1]])
  output_b <- c(List1[[2]], List2[[2]])
  return(list(contrasts = output_a, effects = output_b))
}

# get paths to signal variability measures -------------------------------------
paths <- fromJSON(file = "paths.json")
fpath_sigle_trial_var <- paste(
  paths$bids,
  'derivatives',
  'analysis_dataframes',
  'single_trial_variability.rds',
  sep = '/'
)

# get measures dataframe
signal_variability_st <- readRDS(fpath_sigle_trial_var)

# measures to be analysed
measures <- c("permutation_entropy", "weighted_permutation_entropy",
              "multiscale_entropy", "ms_1", "ms_2", "ms_3", "ms_4", "ms_slope",
              "activity", "mobility", "complexity",
              "1f_offset", "1f_exponent",
              "spectral_entropy")

# get paths to signal variability measures -------------------------------------

# get sensor data
sensor <- signal_variability_st %>%
  ungroup() %>%
  filter(sensor == sensor_n) %>%
  pivot_longer(cols = permutation_entropy:spectral_entropy,
               names_to = "var_meas", values_to = "var") %>%
  filter(!is.na(var)) %>%
  mutate(tw = paste(stringr::str_to_title(window),
                    stringr::str_to_title(stimulus))) %>%
  mutate(tw = factor(tw, levels = c('Pre Cue', 'Post Cue', 'Post Target')),
         task = factor(task, levels = c('Odd/Even', 'Number/Letter')),
         condition = factor(condition, labels = c('Repeat', 'Switch'))
  )

# fit models for each measure --------------------------------------------------

message(
  paste0(
    '\n',
    'Running signal variability analysis for sensor: ', sensor_n,
    '. Task: ', task_i)
)

sensor_results <- foreach(
  meas = measures,
  .combine = 'comb',
  .init = list(list(), list())
) %dopar% {

  require(dplyr)
  require(afex)
  require(emmeans)

  # get measure of interest
  dat <- sensor %>%
    filter(var_meas == meas,
           task == task_i)

  # fit LMM
  var_mxd <- mixed(
    var ~ tw + (tw | subject),
    dat,
    method = 'S',
    check_contrasts = TRUE,
    all_fit = TRUE
  )

  o2 <- effectsize::F_to_omega2(f = var_mxd$anova_table$F,
                                df = var_mxd$anova_table$`num Df`,
                                df_error = var_mxd$anova_table$`den Df`,
                                ci = 0.99)
  o2 <- data.frame(o2)

  # model performance
  df_model_fit <- performance::model_performance(var_mxd$full_model) %>%
    data.frame() %>%
    mutate(predictor = row.names(var_mxd$anova_table),
           F = var_mxd$anova_table$F,
           p = var_mxd$anova_table$`Pr(>F)`,
           o_sq = o2$Omega2_partial,
           o_sq_ci = o2$CI,
           o_sq_CI_low = o2$CI_low,
           o_sq_CI_high = o2$CI_high,
           optimiser = attr(var_mxd$full_model, 'optinfo')$optimizer,
           sensor = sensor_n + 1,
           measure = meas,
           task = task_i)

  # compute contrasts
  var_eff <- emmeans(
    var_mxd$full_model, ~ tw,
    type = 'response',
    lmer.df = "satterthwaite",
    lmerTest.limit = nrow(dat),
    level = 0.99
  )
  contr_var_eff <- contrast(var_eff, method = 'pairwise', adjust = 'holm')
  contr_var_eff <- data.frame(contr_var_eff)

  # compute effect sizes
  contr_d <- effectsize::t_to_d(
    t = contr_var_eff$t.ratio,
    df = contr_var_eff$df,
    paired = TRUE,
    ci = 0.99
  )

  # make effect sizes table
  contr_d <- contr_d %>%
    data.frame() %>%
    mutate(contrast = contr_var_eff$contrast,
           sensor = sensor_n + 1,
           measure = meas)

  list(contrasts = list(contr_d), effects = list(df_model_fit))

}

# extract foreach results ------------------------------------------------------

# save model fits
fits <- sensor_results[2] %>%
  bind_rows()
# save effect sizes
eff_sizes <- sensor_results[1] %>%
  bind_rows()

# save tables ------------------------------------------------------------------
task_i <- stringr::str_remove(
  stringr::str_to_lower("Number/Letter"),
  '\\/'
)
# create paths for results storage
fpath_contrasts_var <- paste(
  paths$bids,
  'derivatives',
  'analysis_dataframes',
  paste0(task_i, '_sensor_', sensor_n, '_constrats_st.rds'),
  sep = '/')
dir.create(dirname(file.path(fpath_contrasts_var)), showWarnings = FALSE)
saveRDS(eff_sizes, file = fpath_contrasts_var)

fpath_fits_var <- paste(
  paths$bids,
  'derivatives',
  'analysis_dataframes',
  paste0(task_i, '_sensor_', sensor_n, '_fits_st.rds'),
  sep = '/')
dir.create(dirname(file.path(fpath_fits_var)), showWarnings = FALSE)
saveRDS(fits, file = fpath_fits_var)
