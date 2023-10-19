# get utility functions --------------------------------------------------------
source('utils.R')

load.package(
  c('parallel', 'foreach', 'optparse')
)

# parallel settings
n.cores <- 11
my.cluster <- parallel::makeCluster(
  n.cores,
  type = "PSOCK"
)
# print(my.cluster)

# register it to be used by %dopar%
doParallel::registerDoParallel(cl = my.cluster)

#c heck if it is registered (optional)
# foreach::getDoParRegistered()

# function for combining output of foreach
comb <- function(List1, List2) {
  output_a <- c(List1[[1]], List2[[1]])
  output_b <- c(List1[[2]], List2[[2]])
  return(list(contrasts = output_a, effects = output_b))
}

# parse arguments --------------------------------------------------------------
option_list <- list(
  make_option(c("--sensor_n"),
              type = "integer",
              help = "Which sensor should be analysed",
              action = "store",
              default = 1
  )
)

# save all the options variables in a list -------------------------------------
opt <- parse_args(
  OptionParser(option_list = option_list)
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
              "multiscale_entropy", "ms_1", "ms_2", "ms_3", "ms_4", "ms_slope",
              "activity", "mobility", "complexity",
              "1f_offset", "1f_exponent",
              "spectral_entropy")

# get paths to signal variability measures -------------------------------------

# place holder for results
sensor_effects <- data.frame()
sensor_contrasts <- data.frame()

# get sensor data
sensor <- signal_variability_st %>%
  ungroup() %>%
  filter(sensor == option_list$sensor_n) %>%
  pivot_longer(cols = permutation_entropy:spectral_entropy,
               names_to = "var_meas", values_to = "var") %>%
  filter(!is.na(var)) %>%
  mutate(tw = paste(stringr::str_to_title(window),
                    stringr::str_to_title(stimulus))) %>%
  mutate(tw = factor(tw, levels = c('Pre Cue', 'Post Cue', 'Post Target')),
         task = factor(task, levels = c('Odd/Even', 'Number/Letter')),
         condition = factor(condition, labels = c('Repeat', 'Switch'))
  )

# and euch measure
sensor_results <- foreach(
  meas = measures,
  .combine = 'comb',
  .init = list(list(), list())
) %dopar% {

  require(dplyr)
  require(afex)
  require(emmeans)

  # # * for debbuging purposes *
  # print(paste0('Running model for sensor: ', sensor_n, '. Measure: ', meas))

  # get measure of interest
  dat <- sensor %>%
    filter(var_meas == meas)

  # fit LMM
  var_mxd <- mixed(
    var ~ tw + (tw | subject),
    dat,
    method = 'S',
    check_contrasts = TRUE,
    all_fit = TRUE
  )

  # model performance
  df_model_fit <- performance::model_performance(var_mxd$full_model) %>%
    data.frame() %>%
    mutate(predictor = row.names(var_mxd$anova_table),
           F = var_mxd$anova_table$F,
           p = var_mxd$anova_table$`Pr(>F)`,
           o_sq = effectsize::F_to_omega2(f = var_mxd$anova_table$F,
                                          df = var_mxd$anova_table$`num Df`,
                                          df_error = var_mxd$anova_table$`den Df`),
           optimiser = attr(var_mxd$full_model, 'optinfo')$optimizer,
           sensor = sensor_n + 1,
           measure = meas)

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

# save model fits
fits <- sensor_results[2] %>%
  bind_rows()
sensor_effects <- rbind(sensor_effects, fits)

# save effect sizes
eff_sizes <- sensor_results[1] %>%
  bind_rows()
sensor_contrasts <- rbind(sensor_contrasts, eff_sizes)

# create paths for results storage
fpath_single_trial_var <- paste(paths$bids,
                                'derivatives',
                                'analysis_dataframes',
                                'sensor_constrats_st.rds',
                                sep = '/')
dir.create(dirname(file.path(fpath_single_trial_var)), showWarnings = FALSE)
saveRDS(eff_sizes, file = fpath_single_trial_var)

fpath_single_trial_var <- paste(paths$bids,
                                'derivatives',
                                'analysis_dataframes',
                                'sensor_effects_st.rds',
                                sep = '/')
dir.create(dirname(file.path(fpath_single_trial_var)), showWarnings = FALSE)
saveRDS(fits, file = fpath_single_trial_var)