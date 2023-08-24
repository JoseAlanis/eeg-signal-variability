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

from joblib import delayed

import numpy as np
import pandas as pd

from mne import read_epochs
from mne.utils import logger

from utils import parse_overwrite, ProgressParallel
from neural_variability import compute_signal_variability

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
        overwrite=overwrite,
        report=report,
        jobs=jobs
    )

    defaults = parse_overwrite(defaults)

    subject = defaults["subject"]
    session = defaults["session"]
    task = defaults["task"]
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
for stim in ['cue', 'target']:

    if stim == 'cue':

        for stage in ['pre', 'post']:

            #  create path for import
            FNAME = os.path.join(
                FPATH_EPOCHS,
                'sub-%s_task-%s_%s%s-epo.fif' % (str_subj, task, stage, stim)
            )

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

    if stim == 'target':

        stage = 'post'

        #  create path for import
        FNAME = os.path.join(FPATH_EPOCHS,
                             'sub-%s_task-%s_%s%s-epo.fif' % (str_subj, task, stage, stim))

        if not os.path.exists(FNAME):
            warnings.warn(FPATH_BIDSDATA_NOT_FOUND_MSG.format(FNAME))
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