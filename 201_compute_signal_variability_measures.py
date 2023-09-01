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

import matplotlib.pyplot as plt
from joblib import delayed

import numpy as np
import pandas as pd

from scipy.signal import periodogram

from mne import read_epochs
from mne.utils import logger

from stats import mean_confidence_interval

from utils import parse_overwrite, ProgressParallel
from signal_variability import frequency_variability

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


# name of measures that should be computed
measures = ['fooof',
            's_entropy', 'p_entropy', 'wp_entropy',
            'mse', 'mse_bins', 'mse_slope',
            'activity', 'mobility', 'complexity']

# placeholder for results
vals = np.empty((n_epochs, n_channels, len(measures)+4))

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
psd_reuslts = pd.DataFrame(
    psd_st,
    columns=np.concatenate(
        (psd_adds, np.array(['f_' + str(f) for f in freqs])))
)

# %%
# export PSD results to .tsv
FPATH_PSD = os.path.join(
    FPATH_DERIVATIVES,
    'power_spectral_density',
    'sub-%s_task-%s_%s%s_psd.tsv' % (str_subj, task, window, stimulus)
)

if not Path(FPATH_PSD).parent.exists():
    Path(FPATH_PSD).parent.mkdir(parents=True, exist_ok=True)

psd_reuslts.to_csv(FPATH_PSD, index=False, sep='\t', float_format='%.4f')

# tidy up
del psd_reuslts, psd_st

# %%
import time
starts_t = time.time()
for i in range(psd.shape[0]):
    for j in range(psd.shape[1]):
        meas = frequency_variability(freqs, psd[i, j, :], freq_lim=[2.0, 45.0])
        print(str(j) + ' ' + str(i))
starts_e = time.time()
print("total time taken this loop: ", starts_e - starts_t)

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # Import tqdm for the progress bar

# Parallelize the loop with a progress bar
def parallel_loop(freqs, psd, num_jobs):
    results = np.zeros((psd.shape[0], psd.shape[1]))

    def process_element(i, j):
        freq_var = frequency_variability(freqs, psd[i, j, :], freq_lim=[2.0, 45.0])
        return freq_var['spectral_entropy']

    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        total_tasks = psd.shape[0] * psd.shape[1]
        progress_bar = tqdm(total=total_tasks, desc="Processing", unit="task")

        for i in range(psd.shape[0]):
            for j in range(psd.shape[1]):
                results[i, j] = executor.submit(process_element, i, j).result()
                progress_bar.update(1)  # Update the progress bar

        progress_bar.close()

    return results


num_jobs = 2  # Specify the number of parallel jobs
parallel_results = parallel_loop(freqs, psd, num_jobs)

delayed_funcs = [
    delayed(frequency_variability)(freqs, psd[0, ch, :], freq_lim=[2.0, 45.0])
    for ch in range(psd.shape[1])
]

parallel_pool = ProgressParallel(n_jobs=4)
out = np.array(parallel_pool(delayed_funcs))


pd.DataFrame(vals,
             columns=np.concatenate((['epoch'], out[0, 1, :])))


correct_repeat = ((epochs.metadata.behavior == 'repeat') & (epochs.metadata.accuracy == 1))
correct_switch = (epochs.metadata.behavior == 'switch') & (epochs.metadata.accuracy == 1)

psd_repeat = psd[correct_repeat, :, :]
psd_switch = psd[correct_switch, :, :]



m, lwd, upp = mean_confidence_interval(psd_repeat)
ms, lwds, upps = mean_confidence_interval(psd_switch)

fig, ax = plt.subplots(1, 1)
ax.plot(freqs, m[0, :], color='k')
ax.plot(freqs, ms[0, :], color='crimson')
ax.fill_between(freqs, lwd[0, :], upp[0, :], alpha=0.2, color='k')
ax.fill_between(freqs, lwds[0, :], upps[0, :], alpha=0.2, color='crimson')
ax.set_xlim(1, 20)

# compute measures on a single trial level
for epo in range(n_epochs):

    signal = epochs.get_data()[epo, :, :]

    delayed_funcs = [
        delayed(compute_signal_variability)(signal[ch],
                                            measures=measures,
                                            sfreq=sfreq,
                                            freq_lim=[2.0, 45.0])
        for ch in range(signal.shape[0])
    ]

    parallel_pool = ProgressParallel(n_jobs=4)
    out = np.array(parallel_pool(delayed_funcs))
    vals[epo, ...] = out[:, 0, :]

m, n, r = vals.shape
vals = np.column_stack(
    (np.repeat(epochs.selection, n), vals.reshape(m*n, -1))
)

var_meas = pd.DataFrame(vals,
                        columns=np.concatenate(
                            (['epoch'], out[0, 1, :])))
var_meas['channel'] = epochs.ch_names * m
var_meas['condition'] = np.repeat(
    epochs.metadata.behavior.to_numpy(), n)
var_meas['rt'] = np.repeat(
    epochs.metadata.rt.to_numpy(), n)
var_meas['accuracy'] = np.repeat(
    epochs.metadata.accuracy.to_numpy(), n)
var_meas = var_meas.assign(subject='%s' % subj)
var_meas = var_meas.assign(session='%s' % session)
var_meas = var_meas.assign(stage='%s' % stage)
var_meas = var_meas.assign(stim='%s' % stim)
var_meas = var_meas.assign(task='%s' % task)

# export to .tsv
FPATH_VAR_MEASURES = os.path.join(
    FPATH_DATA_DERIVATIVES,
    'neural_variability_measures',
    'sub-%s_task-%s_%s%s_variability.tsv' % (
        str_subj, task, stage, stim))

if not Path(FPATH_VAR_MEASURES).parent.exists():
    Path(FPATH_VAR_MEASURES).parent.mkdir(parents=True,
                                          exist_ok=True)

var_meas.to_csv(FPATH_VAR_MEASURES, index=False,
                sep='\t', float_format='%.4f')

if stim == 'target':

    stage = 'post'

    #  create path for import
    FNAME = os.path.join(FPATH_EPOCHS,
                         'sub-%s_task-%s_%s%s-epo.fif' % (str_subj, task, stage, stim))

    if not os.path.exists(FNAME):
        warnings.warn(MISSING_FPATH_BIDS_MSG.format(FNAME))
        sys.exit()

    # get the data
    epochs = read_epochs(FNAME, preload=True)
    sfreq = epochs.info['sfreq']

    n_epochs = len(epochs)
    n_channels = len(epochs.ch_names)

    # name of measures
    measures = ['fooof', 's_entropy', 'p_entropy',
                'wp_entropy', 'mse', 'mse_bins', 'mse_slope',
                'mobility', 'complexity']

    # placeholder for results
    vals = np.empty((n_epochs, n_channels, len(measures)+4))

    # compute measures on a single trial level
    for epo in range(n_epochs):

        signal = epochs.get_data()[epo, :, :]

        # freqs, _, spg = spectrogram(signal[0, :] - np.mean(signal[0, :]),
        #                             1000,
        #                             'hamming', 800, None,
        #                             nfft=1000 * 2,
        #                             detrend=False)
        #
        # freqs_1, psd = periodogram(signal[0, :],
        #                          detrend='constant',
        #                          fs=1000,
        #                          nfft=1000 * 2,
        #                          window='hamming')
        #
        # fig, ax = plt.subplots(1,1)
        # ax.plot(freqs, np.log10(spg), label='spec')
        # ax.plot(freqs_1, np.log10(psd), label='period')
        # ax.plot(freqs_short, np.log10(psd_short), label='period')
        # ax.set_xlim(0, 50)
        # ax.plot(off - np.log10(freqs_short**exp))
        # ax.legend()

        delayed_funcs = [
            delayed(compute_signal_variability)(signal[ch],
                                                measures=measures,
                                                sfreq=sfreq,
                                                freq_lim=[2.0, 45.0])
            for ch in range(signal.shape[0])
        ]

        parallel_pool = ProgressParallel(n_jobs=jobs)
        out = np.array(parallel_pool(delayed_funcs))
        vals[epo, ...] = out[:, 0, :]

    m, n, r = vals.shape
    vals = np.column_stack(
        (np.repeat(epochs.selection, n), vals.reshape(m*n, -1))
    )

    var_meas = pd.DataFrame(vals,
                            columns=np.concatenate(
                                (['epoch'], out[0, 1, :])))
    var_meas['channel'] = epochs.ch_names * m
    var_meas['condition'] = np.repeat(
        epochs.metadata.behavior.to_numpy(), n)
    var_meas['rt'] = np.repeat(
        epochs.metadata.rt.to_numpy(), n)
    var_meas['accuracy'] = np.repeat(
        epochs.metadata.accuracy.to_numpy(), n)
    var_meas = var_meas.assign(subject='%s' % subj)
    var_meas = var_meas.assign(session='%s' % session)
    var_meas = var_meas.assign(stage='%s' % stage)
    var_meas = var_meas.assign(stim='%s' % stim)
    var_meas = var_meas.assign(task='%s' % task)

    # export to .tsv
    FPATH_VAR_MEASURES = os.path.join(
        FPATH_DATA_DERIVATIVES,
        'neural_variability_measures',
        'sub-%s_task-%s_%s%s_variability.tsv' % (
            str_subj, task, stage, stim))

    if not Path(FPATH_VAR_MEASURES).parent.exists():
        Path(FPATH_VAR_MEASURES).parent.mkdir(parents=True,
                                              exist_ok=True)

    var_meas.to_csv(FPATH_VAR_MEASURES, index=False,
                    sep='\t', float_format='%.4f')