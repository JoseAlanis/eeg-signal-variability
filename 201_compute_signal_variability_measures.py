"""
===================================
Compute neural variability measures
===================================

Extracts relevant data and removes artefacts.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
# %%
# imports
import sys
import os

import warnings

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import periodogram

from mne import read_epochs
from mne.utils import logger

from utils import parse_overwrite
from signal_variability import parallel_analysis

from config import (
    FPATH_DERIVATIVES,
    MISSING_FPATH_BIDS_MSG,
    SUBJECT_IDS,
    BAD_SUBJECTS_SES_01,
    BAD_SUBJECTS_SES_02
)

# %%
# default settings (use subject 1, don't overwrite output files)
subject = 1
session = 1
task = 'oddeven'
stimulus = 'cue'
window = 'pre'
overwrite = False
report = False
jobs = 1

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        subject=subject,
        session=session,
        task=task,
        stimulus=stimulus,
        window=window,
        overwrite=overwrite,
        report=report,
        jobs=jobs
    )

    defaults = parse_overwrite(defaults)

    subject = defaults["subject"]
    session = defaults["session"]
    task = defaults["task"]
    stimulus = defaults["stimulus"]
    window = defaults["window"]
    overwrite = defaults["overwrite"]
    report = defaults["report"]
    jobs = defaults["jobs"]

# %%
# paths and overwrite settings
if subject not in SUBJECT_IDS:
    raise ValueError(
        f"'{subject}' is not a valid subject ID.\nUse: {SUBJECT_IDS}"
    )

# skip bad subjects
if session == 1 and subject in BAD_SUBJECTS_SES_01:
    sys.exit()
if session == 2 and subject in BAD_SUBJECTS_SES_02:
    sys.exit()

# create path for preprocessed data
str_subj = str(subject).rjust(3, '0')
FPATH_EPOCHS = os.path.join(FPATH_DERIVATIVES,
                            'epochs',
                            'sub-%s' % str_subj)

if overwrite:
    logger.info("`overwrite` is set to ``True`` ")

# %%
#  create path for import
FNAME = os.path.join(
    FPATH_EPOCHS,
    'sub-%s_task-%s_%s%s-epo.fif' % (str_subj, task, window, stimulus)
)

if not os.path.exists(FNAME):
    warnings.warn(MISSING_FPATH_BIDS_MSG.format(FNAME))
    sys.exit()

# %%
# get the data
epochs = read_epochs(FNAME, preload=True)

# information needed for analyses
sfreq = epochs.info['sfreq']
n_epochs = len(epochs)
n_channels = len(epochs.ch_names)
metadata = epochs.metadata

# %%
psd_adds = np.array(
    ['subject', 'epoch', 'sensor',
     'condition', 'stimulus', 'window', 'accuracy', 'rt']
)

# %%
# run initial frequency analysis

# get EEG signal
signal = epochs.get_data()

# compute power spectral density (PSD)
freqs, psd = periodogram(signal,
                         detrend='constant',
                         fs=sfreq,
                         nfft=sfreq * 2,
                         window='hamming')

# normalize psd (epoch- and sensor-wise)
for epo in range(psd.shape[0]):
    for ch in range(psd[epo, :, :].shape[0]):
        psd[epo, ch, :] = psd[epo, ch, :] / psd[epo, ch, :].sum(keepdims=True)

# %%
# make psd results dataframe
e, c, f = psd.shape
psd_st = np.column_stack(
    (
        np.repeat(metadata.subject, c),
        np.repeat(epochs.selection, c),
        np.tile(np.arange(0, 32), e),
        np.repeat(metadata.behavior, c),
        np.repeat(stimulus, e*c),
        np.repeat(window, e*c),
        np.repeat(metadata.accuracy, c),
        np.repeat(metadata.rt, c),
        psd.reshape(e*c, -1)
    )
)
psd_results = pd.DataFrame(
    psd_st,
    columns=np.concatenate(
        (psd_adds, np.array(['f_' + str(f) for f in freqs])))
)

# only keep relevant frequencies
psd_results.drop(
    psd_results.columns[np.arange(209, 1009)],
    axis=1,
    inplace=True
)

# make column types for pandas dataframe
types_subj = {
    "subject": int,
    "epoch": int,
    "sensor": int,
    "accuracy": int,
    "rt": np.float64
}
types_fp = [np.float64 for i in np.arange(0, 201)]
types_fq = {'f_' + str(key): val
            for key, val
            in zip(freqs[freqs <= 100], types_fp)}
types_psd_results = types_subj | types_fq

# set column types
psd_results = psd_results.astype(types_psd_results)

# %%
# export PSD results to .tsv
FPATH_PSD = os.path.join(
    FPATH_DERIVATIVES,
    'power_spectral_density',
    'sub-%s_task-%s_%s%s_psd.tsv' % (str_subj, task, window, stimulus)
)

if not Path(FPATH_PSD).parent.exists():
    Path(FPATH_PSD).parent.mkdir(parents=True, exist_ok=True)

if os.path.exists(FPATH_PSD) and not overwrite:
    raise RuntimeError(
        f"'{FPATH_PSD}' already exists; consider setting 'overwrite' to True"
    )

psd_results.to_csv(FPATH_PSD, index=False, sep='\t', float_format='%.5f')

# tidy up
del psd_results, psd_st, e, c, f, epo, ch

# %%
# compute signal variability measures

# in the frequency domain
frequency_results, measures_fq = parallel_analysis(
    inst=psd, freqs=freqs, jobs=jobs)
# in the amplitude domain
amplitude_results, measures_amp = parallel_analysis(
    inst=signal, freqs=None, jobs=jobs)

# %%
# save frequency domain results

# make frequency results dataframe
meas, e, c = frequency_results.shape

fq = np.column_stack(
    (
        np.repeat(metadata.subject, c),
        np.repeat(epochs.selection, c),
        np.tile(np.arange(0, 32), e),
        np.repeat(metadata.behavior, c),
        np.repeat(stimulus, e*c),
        np.repeat(window, e*c),
        np.repeat(metadata.accuracy, c),
        np.repeat(metadata.rt, c),
        frequency_results[0].reshape(e*c),
        frequency_results[1].reshape(e*c),
        frequency_results[2].reshape(e*c),
    )
)
fq_res = pd.DataFrame(
    fq,
    columns=np.concatenate(
        (psd_adds, measures_fq)
    )
)

# export frequency results results to .tsv
FPATH_FQ_VAR = os.path.join(
    FPATH_DERIVATIVES,
    'signal_variability',
    'sub-%s_task-%s_%s%s_freq_var_single_trial.tsv'
    % (str_subj, task, window, stimulus)
)

if not Path(FPATH_FQ_VAR).parent.exists():
    Path(FPATH_FQ_VAR).parent.mkdir(parents=True, exist_ok=True)

fq_res = fq_res.astype(
    {"subject": int, "epoch": int, "sensor": int,
     "accuracy": int, "rt": np.float64,
     "1f_offset": np.float64,
     "1f_exponent": np.float64,
     "spectral_entropy": np.float64,
     }
)

if os.path.exists(FPATH_FQ_VAR) and not overwrite:
    raise RuntimeError(
        f"'{FPATH_FQ_VAR}' already exists; consider setting 'overwrite' to True"
    )

fq_res.to_csv(FPATH_FQ_VAR, index=False, sep='\t', float_format='%.4f')

# tidy up
del fq_res, fq, meas, e, c

# %%
# save amplitude domain results

# make frequency results dataframe
meas, e, c = amplitude_results.shape

amp = np.column_stack(
    (
        np.repeat(metadata.subject, c),
        np.repeat(epochs.selection, c),
        np.tile(np.arange(0, 32), e),
        np.repeat(metadata.behavior, c),
        np.repeat(stimulus, e*c),
        np.repeat(window, e*c),
        np.repeat(metadata.accuracy, c),
        np.repeat(metadata.rt, c),
        amplitude_results[0].reshape(e*c),
        amplitude_results[1].reshape(e*c),
        amplitude_results[2].reshape(e*c),
        amplitude_results[3].reshape(e*c),
        amplitude_results[4].reshape(e*c),
        amplitude_results[5].reshape(e*c),
        amplitude_results[6].reshape(e*c),
        amplitude_results[7].reshape(e*c),
        amplitude_results[8].reshape(e*c),
        amplitude_results[9].reshape(e*c),
        amplitude_results[10].reshape(e*c),
    )
)
amp_res = pd.DataFrame(
    amp,
    columns=np.concatenate(
        (psd_adds, measures_amp)
    )
)

amp_res = amp_res.astype(
    {"subject": int, "epoch": int, "sensor": int,
     "accuracy": int, "rt": np.float64,
     "permutation_entropy": np.float64,
     "weighted_permutation_entropy": np.float64,
     "multi-scale entropy": np.float64,
     "multi-scale entropy (1)": np.float64,
     "multi-scale entropy (2)": np.float64,
     "multi-scale entropy (3)": np.float64,
     "multi-scale entropy (4)": np.float64,
     "multi-scale entropy (slope)": np.float64,
     "activity": np.float64,
     "mobility": np.float64,
     "complexity": np.float64
     }
)

# export frequency results results to .tsv
FPATH_AM_VAR = os.path.join(
    FPATH_DERIVATIVES,
    'signal_variability',
    'sub-%s_task-%s_%s%s_amp_var_single_trial.tsv'
    % (str_subj, task, window, stimulus)
)

if not Path(FPATH_AM_VAR).parent.exists():
    Path(FPATH_AM_VAR).parent.mkdir(parents=True, exist_ok=True)

if os.path.exists(FPATH_AM_VAR) and not overwrite:
    raise RuntimeError(
        f"'{FPATH_AM_VAR}' already exists; consider setting 'overwrite' to True"
    )

amp_res.to_csv(FPATH_AM_VAR, index=False, sep='\t', float_format='%.4f')

# tidy up
del amp_res, amp, meas, e, c

# %%

# get correct responses
correct_repeat = (metadata.accuracy == 1) & (metadata.behavior == 'repeat')
correct_switch = (metadata.accuracy == 1) & (metadata.behavior == 'switch')

# get incorrect responses
incorrect_repeat = (metadata.accuracy == 0) & (metadata.behavior == 'repeat')
incorrect_switch = (metadata.accuracy == 0) & (metadata.behavior == 'switch')

conditions = {'correct_repeat': correct_repeat,
              'correct_switch': correct_switch,
              'incorrect_repeat': incorrect_repeat,
              'incorrect_switch': incorrect_switch}

# loop through conditions
for cond in conditions:
    if conditions[cond].sum():
        # compute condition
        mean_psd = psd[conditions[cond], :, :].mean(axis=0, keepdims=True)
        mean_rt = metadata[correct_repeat].rt.mean()
        acc = 1 if 'correct' in cond else 0

        # in the frequency domain
        frequency_results_cond, measures_fq_cond = parallel_analysis(
            inst=mean_psd, freqs=freqs, jobs=jobs)

        # make frequency results dataframe
        meas, e, c = frequency_results_cond.shape

        fq_cond = np.column_stack(
            (
                np.repeat(np.unique(metadata.subject), c),
                np.repeat('average', c),
                np.tile(np.arange(0, 32), e),
                np.repeat(cond.split('_')[1], c),
                np.repeat(stimulus, e*c),
                np.repeat(window, e*c),
                np.repeat(acc, c),
                np.repeat(mean_rt, c),
                frequency_results_cond[0].reshape(e*c),
                frequency_results_cond[1].reshape(e*c),
                frequency_results_cond[2].reshape(e*c),
            )
        )
        fq_res_cond = pd.DataFrame(
            fq_cond,
            columns=np.concatenate(
                (psd_adds, measures_fq_cond)
            )
        )

        # export frequency results results to .tsv
        FPATH_FQ_COND_VAR = os.path.join(
            FPATH_DERIVATIVES,
            'signal_variability',
            'sub-%s_task-%s_%s%s_freq_var_%s_average_psd.tsv'
            % (str_subj, task, window, stimulus, cond)
        )

        if not Path(FPATH_FQ_COND_VAR).parent.exists():
            Path(FPATH_FQ_COND_VAR).parent.mkdir(parents=True, exist_ok=True)

        fq_res_cond = fq_res_cond.astype(
            {"subject": int, "sensor": int,
             "accuracy": int, "rt": np.float64,
             "1f_offset": np.float64,
             "1f_exponent": np.float64,
             "spectral_entropy": np.float64,
             }
        )

        if os.path.exists(FPATH_FQ_COND_VAR) and not overwrite:
            raise RuntimeError(
                f"'{FPATH_FQ_COND_VAR}' already exists; "
                f"consider setting 'overwrite' to True"
            )

        fq_res_cond.to_csv(FPATH_FQ_COND_VAR,
                           index=False, sep='\t', float_format='%.4f')


# m, lwd, upp = mean_confidence_interval(psd_repeat)
# ms, lwds, upps = mean_confidence_interval(psd_switch)
#
# fig, ax = plt.subplots(1, 1)
# ax.plot(freqs, m[0, :], color='k')
# ax.plot(freqs, ms[0, :], color='crimson')
# ax.fill_between(freqs, lwd[0, :], upp[0, :], alpha=0.2, color='k')
# ax.fill_between(freqs, lwds[0, :], upps[0, :], alpha=0.2, color='crimson')
# ax.set_xlim(1, 20)
#
# # compute measures on a single trial level
# for epo in range(n_epochs):
#
#     signal = epochs.get_data()[epo, :, :]
#
#     delayed_funcs = [
#         delayed(compute_signal_variability)(signal[ch],
#                                             measures=measures,
#                                             sfreq=sfreq,
#                                             freq_lim=[2.0, 45.0])
#         for ch in range(signal.shape[0])
#     ]
#
#     parallel_pool = ProgressParallel(n_jobs=4)
#     out = np.array(parallel_pool(delayed_funcs))
#     vals[epo, ...] = out[:, 0, :]
#
# m, n, r = vals.shape
# vals = np.column_stack(
#     (np.repeat(epochs.selection, n), vals.reshape(m*n, -1))
# )
#
# var_meas = pd.DataFrame(vals,
#                         columns=np.concatenate(
#                             (['epoch'], out[0, 1, :])))
# var_meas['channel'] = epochs.ch_names * m
# var_meas['condition'] = np.repeat(
#     epochs.metadata.behavior.to_numpy(), n)
# var_meas['rt'] = np.repeat(
#     epochs.metadata.rt.to_numpy(), n)
# var_meas['accuracy'] = np.repeat(
#     epochs.metadata.accuracy.to_numpy(), n)
# var_meas = var_meas.assign(subject='%s' % subj)
# var_meas = var_meas.assign(session='%s' % session)
# var_meas = var_meas.assign(stage='%s' % stage)
# var_meas = var_meas.assign(stim='%s' % stim)
# var_meas = var_meas.assign(task='%s' % task)
#
# # export to .tsv
# FPATH_VAR_MEASURES = os.path.join(
#     FPATH_DATA_DERIVATIVES,
#     'neural_variability_measures',
#     'sub-%s_task-%s_%s%s_variability.tsv' % (
#         str_subj, task, stage, stim))
#
# if not Path(FPATH_VAR_MEASURES).parent.exists():
#     Path(FPATH_VAR_MEASURES).parent.mkdir(parents=True,
#                                           exist_ok=True)
#
# var_meas.to_csv(FPATH_VAR_MEASURES, index=False,
#                 sep='\t', float_format='%.4f')
#
# if stim == 'target':
#
#     stage = 'post'
#
#     #  create path for import
#     FNAME = os.path.join(FPATH_EPOCHS,
#                          'sub-%s_task-%s_%s%s-epo.fif' % (str_subj, task, stage, stim))
#
#     if not os.path.exists(FNAME):
#         warnings.warn(MISSING_FPATH_BIDS_MSG.format(FNAME))
#         sys.exit()
#
#     # get the data
#     epochs = read_epochs(FNAME, preload=True)
#     sfreq = epochs.info['sfreq']
#
#     n_epochs = len(epochs)
#     n_channels = len(epochs.ch_names)
#
#     # name of measures
#     measures = ['fooof', 's_entropy', 'p_entropy',
#                 'wp_entropy', 'mse', 'mse_bins', 'mse_slope',
#                 'mobility', 'complexity']
#
#     # placeholder for results
#     vals = np.empty((n_epochs, n_channels, len(measures)+4))
#
#     # compute measures on a single trial level
#     for epo in range(n_epochs):
#
#         signal = epochs.get_data()[epo, :, :]
#
#         # freqs, _, spg = spectrogram(signal[0, :] - np.mean(signal[0, :]),
#         #                             1000,
#         #                             'hamming', 800, None,
#         #                             nfft=1000 * 2,
#         #                             detrend=False)
#         #
#         # freqs_1, psd = periodogram(signal[0, :],
#         #                          detrend='constant',
#         #                          fs=1000,
#         #                          nfft=1000 * 2,
#         #                          window='hamming')
#         #
#         # fig, ax = plt.subplots(1,1)
#         # ax.plot(freqs, np.log10(spg), label='spec')
#         # ax.plot(freqs_1, np.log10(psd), label='period')
#         # ax.plot(freqs_short, np.log10(psd_short), label='period')
#         # ax.set_xlim(0, 50)
#         # ax.plot(off - np.log10(freqs_short**exp))
#         # ax.legend()
#
#         delayed_funcs = [
#             delayed(compute_signal_variability)(signal[ch],
#                                                 measures=measures,
#                                                 sfreq=sfreq,
#                                                 freq_lim=[2.0, 45.0])
#             for ch in range(signal.shape[0])
#         ]
#
#         parallel_pool = ProgressParallel(n_jobs=jobs)
#         out = np.array(parallel_pool(delayed_funcs))
#         vals[epo, ...] = out[:, 0, :]
#
#     m, n, r = vals.shape
#     vals = np.column_stack(
#         (np.repeat(epochs.selection, n), vals.reshape(m*n, -1))
#     )
#
#     var_meas = pd.DataFrame(vals,
#                             columns=np.concatenate(
#                                 (['epoch'], out[0, 1, :])))
#     var_meas['channel'] = epochs.ch_names * m
#     var_meas['condition'] = np.repeat(
#         epochs.metadata.behavior.to_numpy(), n)
#     var_meas['rt'] = np.repeat(
#         epochs.metadata.rt.to_numpy(), n)
#     var_meas['accuracy'] = np.repeat(
#         epochs.metadata.accuracy.to_numpy(), n)
#     var_meas = var_meas.assign(subject='%s' % subj)
#     var_meas = var_meas.assign(session='%s' % session)
#     var_meas = var_meas.assign(stage='%s' % stage)
#     var_meas = var_meas.assign(stim='%s' % stim)
#     var_meas = var_meas.assign(task='%s' % task)
#
#     # export to .tsv
#     FPATH_VAR_MEASURES = os.path.join(
#         FPATH_DATA_DERIVATIVES,
#         'neural_variability_measures',
#         'sub-%s_task-%s_%s%s_variability.tsv' % (
#             str_subj, task, stage, stim))
#
#     if not Path(FPATH_VAR_MEASURES).parent.exists():
#         Path(FPATH_VAR_MEASURES).parent.mkdir(parents=True,
#                                               exist_ok=True)
#
#     var_meas.to_csv(FPATH_VAR_MEASURES, index=False,
#                     sep='\t', float_format='%.4f')