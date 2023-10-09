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
# separate conditions

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

# %%
# get EEG signal
signal = epochs.get_data()

# %%
for cond in conditions:
    if conditions[cond].sum():

        mean_signal = signal[conditions[cond], :, :].mean(axis=0, keepdims=True)
        mean_rt = metadata[correct_repeat].rt.mean()
        acc = 1 if 'correct' in cond else 0

        # compute power spectral density (PSD)
        freqs, psd = periodogram(mean_signal,
                                 detrend='constant',
                                 fs=sfreq,
                                 nfft=sfreq * 2,
                                 window='hamming')

        # normalize psd (epoch- and sensor-wise)
        for ch in range(psd.shape[1]):
            psd[:, ch, :] = psd[:, ch, :] / psd[:, ch, :].sum(keepdims=True)

        # in the frequency domain
        frequency_results_cond, measures_fq_cond = parallel_analysis(
            inst=psd, freqs=freqs, jobs=jobs)

        # in the amplitude domain
        amplitude_results, measures_amp = parallel_analysis(
            inst=mean_signal, freqs=None, jobs=jobs)

        # make frequency results dataframe
        meas, e, c = frequency_results_cond.shape

        fq_cond = np.column_stack(
            (
                np.repeat(np.unique(metadata.subject), c),
                np.repeat('ERP', c),
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
            'sub-%s_task-%s_%s%s_freq_var_%s_ERP.tsv'
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

        # make frequency results dataframe
        meas, e, c = amplitude_results.shape

        amp = np.column_stack(
            (
                np.repeat(np.unique(metadata.subject), c),
                np.repeat('ERP', c),
                np.tile(np.arange(0, 32), e),
                np.repeat(cond.split('_')[1], c),
                np.repeat(stimulus, e*c),
                np.repeat(window, e*c),
                np.repeat(acc, c),
                np.repeat(mean_rt, c),
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

        # export frequency results results to .tsv
        FPATH_AM_VAR = os.path.join(
            FPATH_DERIVATIVES,
            'signal_variability',
            'sub-%s_task-%s_%s%s_amp_var_%s_ERP.tsv'
            % (str_subj, task, window, stimulus, cond)
        )

        amp_res = amp_res.astype(
            {"subject": int, "sensor": int,
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

        if not Path(FPATH_AM_VAR).parent.exists():
            Path(FPATH_AM_VAR).parent.mkdir(parents=True, exist_ok=True)

        if os.path.exists(FPATH_AM_VAR) and not overwrite:
            raise RuntimeError(
                f"'{FPATH_AM_VAR}' already exists; "
                f"consider setting 'overwrite' to True"
            )

        amp_res.to_csv(FPATH_AM_VAR, index=False, sep='\t', float_format='%.4f')

        # tidy up
        del amp_res, amp, meas, e, c
