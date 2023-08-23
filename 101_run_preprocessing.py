"""
=============
Preprocessing
==============

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
import matplotlib.pyplot as plt

from mne import events_from_annotations, Report
from mne.preprocessing import ICA
from mne.utils import logger

from mne_bids import BIDSPath, read_raw_bids

from config import (
    FPATH_BIDS,
    FPATH_DERIVATIVES,
    MISSING_FPATH_BIDS_MSG,
    SUBJECT_IDS,
    CHECK_SUBJECTS_SES_01,
    CHECK_SUBJECTS_SES_02,
    BAD_SUBJECTS_SES_01,
    BAD_SUBJECTS_SES_02,
    eeg_markers
)

from utils import parse_overwrite

from pyprep.prep_pipeline import NoisyChannels
from mne_icalabel import label_components

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
    raise ValueError(f"'{subject}' is not a valid subject ID.\nUse: {SUBJECT_IDS}")

# skip bad subjects
if session == 1 and subject in BAD_SUBJECTS_SES_01:
    sys.exit()
if session == 2 and subject in BAD_SUBJECTS_SES_02:
    sys.exit()

if not os.path.exists(FPATH_BIDS):
    raise RuntimeError(
        MISSING_FPATH_BIDS_MSG.format(FPATH_BIDS)
    )

# %%
# create bids path for import

# run file id
run = 1
if subject in CHECK_SUBJECTS_SES_01 and session == 1:
    run = int(CHECK_SUBJECTS_SES_01[subject].split(',')[-1].split(' ')[-1])
elif subject in CHECK_SUBJECTS_SES_02 and session == 2:
    if subject == 43:
        run = 1
    elif subject == 52:
        run = 2

FNAME = BIDSPath(root=FPATH_BIDS,
                 subject=f'{subject:03}',
                 task='multiple',
                 session=str(session),
                 run=run,
                 datatype='eeg',
                 extension='.vhdr')
if not os.path.exists(FNAME):
    warnings.warn(MISSING_FPATH_BIDS_MSG.format(FNAME))
    sys.exit()

# %%
# get the data
raw = read_raw_bids(FNAME)
raw.load_data()

# get sampling rate
sfreq = raw.info['sfreq']

# %%
# extract task relevant events
ses = 'ses-%s' % session
session_ids = eeg_markers[ses]
markers = session_ids[task]['markers']

# standardise event codes for import
event_ids = {'Stimulus/S%s' % str(ev).rjust(3): ev for ev in markers.values()}

# search for desired events in the data
events, events_found = events_from_annotations(raw, event_id=event_ids)

# %%
# extract the desired section of recording (only odd-even task)

# set start and end markers (session 1)
start_end = ['Stimulus/S  6', 'Stimulus/S  7']
if subject == 2:
    start_end = ['Stimulus/S  6', 'Stimulus/S115']

# set start and end markers (session 2)
if session == 2:
    start_end = ['Stimulus/S 12', 'Stimulus/S 13']

# time relevant to those events
if subject == 121 and session == 2:
    tmin = (events[events[:, 2] == 13][0][0] / sfreq) - 1080
else:
    tmin = events[events[:, 2] == events_found[start_end[0]], 0] / sfreq - 20
if subject in {63, 73, 80} and session == 2:
    tmax = (events[events[:, 2] == 90][-1][0] / sfreq) - 0.5
else:
    tmax = events[events[:, 2] == events_found[start_end[1]], 0] / sfreq + 6

# extract data
raw_task = raw.copy().crop(tmin=float(tmin), tmax=float(tmax))

# clean up
del raw

# %%
# bad channel detection and interpolation

# first create a copy of the data
raw_copy = raw_task.copy()
# apply an 100Hz low-pass filter to data
raw_copy = raw_copy.filter(l_freq=None, h_freq=100.0,
                           picks=['eeg', 'eog'],
                           filter_length='auto',
                           l_trans_bandwidth='auto',
                           h_trans_bandwidth='auto',
                           method='fir',
                           phase='zero',
                           fir_window='hamming',
                           fir_design='firwin',
                           n_jobs=jobs)

# find bad channels
noisy_dectector = NoisyChannels(raw_copy, random_state=42, do_detrend=True)
noisy_dectector.find_all_bads(ransac=False)

# crate summary for PyPrep output
bad_channels = {'bads_by_deviation:': noisy_dectector.bad_by_deviation,
                'bads_by_hf_noise:': noisy_dectector.bad_by_hf_noise,
                'bads_by_correlation:': noisy_dectector.bad_by_correlation,
                'bads_by_SNR:': noisy_dectector.bad_by_SNR}

# %%
# interpolate the identified bad channels
raw_task.info['bads'] = noisy_dectector.get_bads()
raw_task.interpolate_bads(mode='accurate')

# %%

# set eeg reference
raw_task = raw_task.set_eeg_reference('average')
# raw_task.apply_proj()

# %%
# prepare ICA

# set ICA parameters
method = 'infomax'
reject = dict(eeg=250e-6)
ica = ICA(n_components=0.95,
          method=method,
          fit_params=dict(extended=True))

# make copy of raw with 1Hz high-pass filter
raw_4_ica = raw_task.copy().filter(l_freq=1.0, h_freq=100.0, n_jobs=jobs)

# run ICA
ica.fit(raw_4_ica,
        reject=reject,
        reject_by_annotation=True)

# %%
# find bad components using ICA label
ic_labels = label_components(raw_4_ica, ica, method="iclabel")

labels = ic_labels["labels"]
exclude_idx = [idx for idx, label in
               enumerate(labels) if label not in ["brain", "other"]]

logger.info(f"Excluding these ICA components: {exclude_idx}")

# exclude the identified components and reconstruct eeg signal
ica.exclude = exclude_idx
ica.apply(raw_task)

# clean up
del raw_4_ica

# %%
# apply filter to data
raw_task = raw_task.filter(l_freq=0.05, h_freq=40.0,
                           picks=['eeg', 'eog'],
                           filter_length='auto',
                           l_trans_bandwidth='auto',
                           h_trans_bandwidth='auto',
                           method='fir',
                           phase='zero',
                           fir_window='hamming',
                           fir_design='firwin',
                           n_jobs=jobs)

# %%
# create path for preprocessed data
FPATH_PREPROCESSED = os.path.join(
    FPATH_DERIVATIVES,
    'preprocessing',
    'sub-%s' % f'{subject:03}',
    'eeg',
    'sub-%s_task-%s_preprocessed-raw.fif' % (f'{subject:03}', task))

# check if directory exists
if not Path(FPATH_PREPROCESSED).exists():
    Path(FPATH_PREPROCESSED).parent.mkdir(parents=True, exist_ok=True)

# save file
if overwrite:
    logger.info("`overwrite` is set to ``True`` ")

raw_task.save(FPATH_PREPROCESSED, overwrite=overwrite)

# %%
if report:
    # make path
    FPATH_REPORT = os.path.join(
        FPATH_DERIVATIVES,
        'preprocessing',
        'sub-%s' % f'{subject:03}',
        'report')

    if not Path(FPATH_REPORT).exists():
        Path(FPATH_REPORT).mkdir(parents=True, exist_ok=True)

    # create data report
    bidsdata_report = Report(title='Subject %s' % f'{subject:03}')
    bidsdata_report.add_raw(raw=raw_task, title='Raw data',
                            butterfly=False,
                            replace=True,
                            psd=True)

    # add bad channels
    bads_html = """
    <p>Bad channels identified by PyPrep:</p>
    <p>%s</p> 
    """ % '<br> '.join([key + ' ' + str(val)
                        for key, val in bad_channels.items()])
    bidsdata_report.add_html(title='Bad channels',
                             tags='bads',
                             html=bads_html,
                             replace=True)
    # add ica
    fig = ica.plot_components(show=False, picks=np.arange(ica.n_components_))
    plt.close('all')

    bidsdata_report.add_figure(
        fig=fig,
        tags='ica',
        title='ICA cleaning',
        caption='Bad components identified by ICA Label: %s' % ', '.join(
            str(ix) + ': ' + labels[ix] for ix in exclude_idx),
        image_format='PNG',
        replace=True
    )

    for rep_ext in ['hdf5', 'html']:
        FPATH_REPORT_O = os.path.join(
            FPATH_REPORT,
            'sub-%s_task-%s_prep_report.%s' % (f'{subject:03}', task, rep_ext))

        if overwrite:
            logger.info("`overwrite` is set to ``True`` ")

            bidsdata_report.save(FPATH_REPORT_O,
                                 overwrite=overwrite,
                                 open_browser=False)
