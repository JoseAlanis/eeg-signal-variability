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

from mne import events_from_annotations, Epochs

from mne.io import read_raw_fif
from mne.utils import logger

from config import (
    FPATH_DERIVATIVES,
    MISSING_FPATH_BIDS_MSG,
    SUBJECT_IDS,
    BAD_SUBJECTS_SES_01,
    BAD_SUBJECTS_SES_02
)

from utils import parse_overwrite

# %%
# default settings (use subject 1, don't overwrite output files)
subject = 1
session = 1
task = 'oddeven'
overwrite = False

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        subject=subject,
        session=session,
        task=task,
        overwrite=overwrite
    )

    defaults = parse_overwrite(defaults)

    subject = defaults["subject"]
    session = defaults["session"]
    task = defaults["task"]
    overwrite = defaults["overwrite"]

# %%
# paths and overwrite settings
if subject not in SUBJECT_IDS:
    raise ValueError(f"'{subject}' is not a valid subject ID.\nUse: {SUBJECT_IDS}")

# skip bad subjects
if session == 1 and subject in BAD_SUBJECTS_SES_01:
    sys.exit()
if session == 2 and subject in BAD_SUBJECTS_SES_02:
    sys.exit()

if not os.path.exists(FPATH_DERIVATIVES):
    raise RuntimeError(
        MISSING_FPATH_BIDS_MSG.format(FPATH_DERIVATIVES)
    )

# create path for preprocessed data
FPATH_PREPROCESSED = os.path.join(FPATH_DERIVATIVES,
                                  'preprocessing',
                                  'sub-%s' % f'{subject:03}')

if overwrite:
    logger.info("`overwrite` is set to ``True`` ")

# %%
#  create path for import
FNAME = os.path.join(
    FPATH_PREPROCESSED,
    'eeg',
    'sub-%s_task-%s_preprocessed-raw.fif' % (f'{subject:03}', task)
)

if not os.path.exists(FNAME):
    warnings.warn(MISSING_FPATH_BIDS_MSG.format(FNAME))
    sys.exit()

# %%
# get the data
raw = read_raw_fif(FNAME)
raw.load_data()

# only keep eeg channels
raw.pick(['eeg'])
# sampling rate
sfreq = raw.info['sfreq']

# %%
# create a dictionary with event IDs for standardised handling
condition_ids = {'Stimulus/S 31': 1,
                 'Stimulus/S 51': 101,

                 'Stimulus/S 32': 2,
                 'Stimulus/S 52': 102,

                 'Stimulus/S 33': 3,
                 'Stimulus/S 53': 103,

                 'Stimulus/S 34': 4,
                 'Stimulus/S 54': 104,

                 'Stimulus/S150': 150,
                 'Stimulus/S160': 160,
                 'Stimulus/S250': 250,
                 'Stimulus/S260': 260,
                 'Stimulus/S251': 9,
                 'Stimulus/S222': 10}

# event codes for segmentation
fix_ids = {'RepeatCue/Correct': 20,
           'RepeatCue/Incorrect': 21,

           'SwitchCue/Correct': 30,
           'SwitchCue/Incorrect': 31,

           'RepeatTarget/Correct': 40,
           'RepeatTarget/Incorrect': 41,

           'SwitchTarget/Correct': 50,
           'SwitchTarget/Incorrect': 51,
           }

# extract events
events, ids = events_from_annotations(raw, event_id=condition_ids,
                                      regexp=None)

# %%
# recode repeat and switch events

# placeholders for epochs metadata
correct_false = []
repeat_switch = []
isi = []
rt = []
evs = np.append(events, np.array([[0, 0, 0]]), axis=0)

# loop through events and recode relevant events
for ev in range(evs.shape[0]):

    # if fix cross repeat trials
    if evs[ev, 2] in {1, 3}:

        # if correct reaction followed
        if evs[ev + 2, 2] in {150, 160}:
            # recode fix
            evs[ev, 2] = 20
            # recode target
            evs[ev + 1, 2] = 40
            # save accuracy
            correct_false.append(1)
            # save rt
            rt.append((evs[ev + 2, 0] - evs[ev + 1, 0]) / sfreq)

        # if incorrect reaction followed
        elif evs[ev + 2, 2] in {250, 260}:
            # recode fix
            evs[ev, 2] = 21
            # recode target
            evs[ev + 1, 2] = 41
            # save accuracy
            correct_false.append(0)
            # save rt
            rt.append((evs[ev + 2, 0] - evs[ev + 1, 0]) / sfreq)

        # else = missed reaction
        else:
            # accuracy and rt to nan
            correct_false.append(np.nan)
            rt.append(np.nan)

        # save condition as "repeat"
        repeat_switch.append('repeat')
        # save ISI
        isi.append((evs[ev + 1, 0] - evs[ev, 0]) / sfreq)

    # if fix-cross switch trials
    elif evs[ev, 2] in {2, 4}:

        # if correct reaction followed
        if evs[ev + 2, 2] in {150, 160}:
            # recode fix
            evs[ev, 2] = 30
            # recode target
            evs[ev + 1, 2] = 50
            # save accuracy
            correct_false.append(1)
            # save rt
            rt.append((evs[ev + 2, 0] - evs[ev + 1, 0]) / sfreq)

        # if incorrect reaction followed
        elif evs[ev + 2, 2] in {250, 260}:
            # recode fix
            evs[ev, 2] = 31
            # recode target
            evs[ev + 1, 2] = 51
            # save accuracy
            correct_false.append(0)
            # save rt
            rt.append((evs[ev + 2, 0] - evs[ev + 1, 0]) / sfreq)

        # else = missed reaction
        else:
            # accuracy and rt to nan
            correct_false.append(np.nan)
            rt.append(np.nan)

        # condition = switch
        repeat_switch.append('switch')
        # save ISI
        isi.append((evs[ev + 1, 0] - evs[ev, 0]) / sfreq)

# %%
# construct epochs metadata
metadata = {'subject': subject,
            'rt': rt,
            'accuracy': correct_false,
            'behavior': repeat_switch,
            }
metadata = pd.DataFrame(metadata)

# %%
# save RT measures for later analyses
rt_data = metadata.copy()
# create path for preprocessed data
FPATH_RT = os.path.join(FPATH_DERIVATIVES,
                        'rt',
                        'sub-%s' % f'{subject:03}',
                        'sub-%s_%s_rt.tsv' % (f'{subject:03}', task))

# check if directory exists
if not Path(FPATH_RT).exists():
    Path(FPATH_RT).parent.mkdir(parents=True, exist_ok=True)

# save rt data to disk
rt_data.to_csv(FPATH_RT,
               sep='\t',
               index=False)

# %%
# meta data for epochs
# drop missing values
metadata = metadata.dropna()

# relevant events (fix-cross)
cue_evs = evs[(evs[:, 2] == 20) |
              (evs[:, 2] == 30) |
              (evs[:, 2] == 21) |
              (evs[:, 2] == 31), :]

target_evs = evs[(evs[:, 2] == 40) |
                 (evs[:, 2] == 50) |
                 (evs[:, 2] == 41) |
                 (evs[:, 2] == 51), :]

# %%
# create path for preprocessed data
FPATH_EPOCHS = os.path.join(FPATH_DERIVATIVES,
                            'epochs',
                            'sub-%s' % f'{subject:03}')

# %%
# extract epochs
for period in ['baseline', 'stimulus']:

    if period == 'baseline':
        # cue events
        cue_keys = {key: val for key, val in fix_ids.items() if 'Cue' in key}

        # processing stage
        for stage in ['pre', 'post']:

            # set analysis time window
            if stage == 'pre':
                tmin, tmax = [-0.800, -0.001]
            else:
                tmin, tmax = [0.001, 0.800]

            # extract cue epochs
            epochs = Epochs(raw,
                            cue_evs,
                            cue_keys,
                            metadata=metadata,
                            on_missing='ignore',
                            tmin=tmin,
                            tmax=tmax,
                            baseline=None,
                            preload=True,
                            reject_by_annotation=True,
                            reject=dict(eeg=125e-6),
                            )

            # save cue epochs to disk
            FPATH = os.path.join(
                FPATH_EPOCHS,
                'sub-%s_task-%s_%scue-epo.fif' % (f'{subject:03}', task, stage)
            )
            # check if directory exists
            if not Path(FPATH).exists():
                Path(FPATH).parent.mkdir(parents=True, exist_ok=True)

            # save epochs
            epochs.save(FPATH,
                        overwrite=overwrite)

    else:
        # target events
        target_keys = {key: val for key, val in fix_ids.items()
                       if 'Target' in key}

        stage = 'post'
        # set analysis times
        tmin, tmax = [0.001, 0.800]

        # extract target epochs
        epochs = Epochs(raw,
                        target_evs,
                        target_keys,
                        metadata=metadata,
                        on_missing='ignore',
                        tmin=tmin,
                        tmax=tmax,
                        baseline=None,
                        preload=True,
                        reject_by_annotation=True,
                        reject=dict(eeg=125e-6),
                        )

        # save cue epochs to disk
        FPATH = os.path.join(
            FPATH_EPOCHS,
            'sub-%s_task-%s_%starget-epo.fif' % (f'{subject:03}', task, stage)
        )
        # check if directory exists
        if not Path(FPATH).exists():
            Path(FPATH).parent.mkdir(parents=True, exist_ok=True)

        # save epochs
        epochs.save(FPATH,
                    overwrite=overwrite)
