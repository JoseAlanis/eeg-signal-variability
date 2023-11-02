# Title:  More than noise?: Signal variability measures, single trial analysis
# Author: José C. García Alanis
# Date:   Fri Oct 20 11:08:12 2023
# Notes:  R script can be ran as job on HPC. Takes args `sensor_n`, `task_i`,
#         and `jobs`

require('optparse')

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
  make_option("--measure",
              type = "character",
              help = "Which measure should be modelled?",
              action = "store",
              default = "permutation_entropy"
  ),
  make_option("--lib",
              type = "character",
              help = "Path to R-library",
              action = "store",
              default = "default"
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
measure <- opt$measure
lib <- opt$lib
jobs <- opt$jobs

if (lib == "default") {
  lib <- .libPaths()[1]
}

# get utility functions --------------------------------------------------------
source('utils.R')

load.package(
  c('foreach',
    'rjson', 'dplyr', 'tidyr', 'stringr',
    'afex', 'optimx',
    'performance', 'emmeans', 'effectsize'),
  lib = lib
)

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
              "multiscale_entropy", "ms_1", "ms_2", "ms_3", "ms_4",
              "activity", "mobility", "complexity",
              "1f_offset", "1f_exponent",
              "spectral_entropy")

# get paths to signal variability measures -------------------------------------

# get sensor data
dat <- signal_variability_st %>%
  ungroup() %>%
  filter(sensor == sensor_n) %>%
  select(-ms_slope) %>%
  pivot_longer(cols = permutation_entropy:spectral_entropy,
               names_to = "var_meas", values_to = "var") %>%
  filter(!is.na(var)) %>%
  mutate(tw = paste(str_to_title(window),
                    str_to_title(stimulus))) %>%
  mutate(tw = factor(tw, levels = c('Pre Cue', 'Post Cue', 'Post Target')),
         task = factor(task, levels = c('Odd/Even', 'Number/Letter')),
         condition = factor(condition, labels = c('Repeat', 'Switch'))
  ) %>%
  filter(var_meas == measure,
         task == task_i)

# fit models for each measure --------------------------------------------------

message(
  paste0(
    '\n',
    'Running signal variability analysis for sensor: ', sensor_n,
    '. Task: ', task_i, '. Measure: ', measure)
)

# fit LMM
var_mxd <- mixed(
  var ~ tw * condition + (tw * condition || subject),
  dat,
  method = 'S',
  check_contrasts = TRUE,
  all_fit = TRUE,
  expand_re = TRUE,
  control = lmerControl(optCtrl = list(maxfun = 1e8))
)

# model fit
mod_perf <- performance::model_performance(var_mxd$full_model) %>%
  data.frame()

# effect sizes
o2 <- effectsize::F_to_omega2(f = var_mxd$anova_table$F,
                              df = var_mxd$anova_table$`num Df`,
                              df_error = var_mxd$anova_table$`den Df`,
                              ci = 0.99,
                              alternative = "two.sided")

# save model fit in data frame for export
df_model_fit <- o2 %>%
  data.frame() %>%
  mutate(predictor = row.names(var_mxd$anova_table),
         F = var_mxd$anova_table$F,
         p = var_mxd$anova_table$`Pr(>F)`,
         optimiser = attr(var_mxd$full_model, 'optinfo')$optimizer,
         sensor = sensor_n + 1,
         measure = measure,
         task = task_i,
         AICc = mod_perf$AIC,
         AICc = mod_perf$AICc,
         BIC = mod_perf$BIC,
         R2_conditional = mod_perf$R2_conditional,
         R2_marginal = mod_perf$R2_marginal)

# compute contrasts ** time window **
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
contr_d_tw <- effectsize::t_to_d(
  t = contr_var_eff$t.ratio,
  df = contr_var_eff$df,
  paired = TRUE,
  ci = 0.99
)

# make effect sizes table
contr_d_tw <- contr_d_tw %>%
  data.frame() %>%
  mutate(contrast = contr_var_eff$contrast,
         sensor = sensor_n + 1,
         measure = measure,
         task = task_i)

# compute contrasts ** condition **
var_eff <- emmeans(
  var_mxd$full_model, ~ condition,
  type = 'response',
  lmer.df = "satterthwaite",
  lmerTest.limit = nrow(dat),
  level = 0.99
)
contr_var_eff <- contrast(var_eff, method = 'pairwise', adjust = 'holm')
contr_var_eff <- data.frame(contr_var_eff)

# compute effect sizes
contr_d_cond <- effectsize::t_to_d(
  t = contr_var_eff$t.ratio,
  df = contr_var_eff$df,
  paired = TRUE,
  ci = 0.99
)

# make effect sizes table
contr_d_cond <- contr_d_cond %>%
  data.frame() %>%
  mutate(contrast = contr_var_eff$contrast,
         sensor = sensor_n + 1,
         measure = measure,
         task = task_i)

# compute contrasts ** interaction **
var_eff <- emmeans(
  var_mxd$full_model, ~ tw | condition,
  type = 'response',
  lmer.df = "satterthwaite",
  lmerTest.limit = nrow(dat),
  level = 0.99
)
contr_var_eff <- contrast(var_eff, method = 'pairwise', adjust = 'holm')
contr_var_eff <- data.frame(contr_var_eff)

# compute effect sizes
contr_d_int <- effectsize::t_to_d(
  t = contr_var_eff$t.ratio,
  df = contr_var_eff$df,
  paired = TRUE,
  ci = 0.99
)

# make effect sizes table
contr_d_int <- contr_d_int %>%
  data.frame() %>%
  mutate(contrast = paste0(contr_var_eff$contrast, ' (', contr_var_eff$condition, ')'),
         sensor = sensor_n + 1,
         measure = measure,
         task = task_i)

contr_d <- bind_rows(list(contr_d_tw, contr_d_cond, contr_d_int))

# save tables ------------------------------------------------------------------
task_i <- str_remove(
  str_to_lower(task_i),
  '\\/'
)

# create paths for results storage
fpath_contrasts_var <- paste(
  paths$bids,
  'derivatives',
  'analysis_dataframes',
  paste0(task_i, '_sensor_', sensor_n, '_', measure, '_constrats_st.rds'),
  sep = '/')
dir.create(dirname(file.path(fpath_contrasts_var)), showWarnings = FALSE)
saveRDS(contr_d, file = fpath_contrasts_var)

fpath_fits_var <- paste(
  paths$bids,
  'derivatives',
  'analysis_dataframes',
  paste0(task_i, '_sensor_', sensor_n,  '_', measure, '_fits_st.rds'),
  sep = '/')
dir.create(dirname(file.path(fpath_fits_var)), showWarnings = FALSE)
saveRDS(df_model_fit, file = fpath_fits_var)
