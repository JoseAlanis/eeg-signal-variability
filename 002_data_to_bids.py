"""
===========================
Source data set to EEG BIDS
===========================

Put EEG data into a BIDS-compliant directory structure.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
# %%
# imports
import sys
import os

import re

from mne.io import read_raw_brainvision
from mne.utils import logger

from mne_bids import BIDSPath, write_raw_bids

from config import (
    FPATH_SOURCEDATA,
    FPATH_BIDS,
    MISSING_FPATH_SOURCEDATA_MSG,
    FNAME_SOURCEDATA_TEMPLATE,
    SUBJECT_IDS,
    CHECK_SUBJECTS_SES_01,
    CHECK_SUBJECTS_SES_02,
    # montage
)

from utils import parse_overwrite

# %%
# default settings (use subject 1, don't overwrite output files)
subject = 1
session = 1
overwrite = False
ext = '.vhdr'

# %%
# When not in an IPython session, get command line inputs
# https://docs.python.org/3/library/sys.html#sys.ps1
if not hasattr(sys, "ps1"):
    defaults = dict(
        subject=subject,
        session=session,
        overwrite=overwrite,
    )

    defaults = parse_overwrite(defaults)

    subject = defaults["subject"]
    session = defaults["session"]
    overwrite = defaults["overwrite"]

# %%
# paths and overwrite settings
if subject not in SUBJECT_IDS:
    raise ValueError(f"{subject} is not a valid subject ID.\n"
                     f"Use one of: {SUBJECT_IDS}")

if not os.path.exists(FPATH_SOURCEDATA):
    raise RuntimeError(
        MISSING_FPATH_SOURCEDATA_MSG.format(FPATH_SOURCEDATA)
    )
if overwrite:
    logger.info("`overwrite` is set to ``True`` ")

# %%
# path to file in question (i.e., which subject and session)
if session == 1:
    fname = FNAME_SOURCEDATA_TEMPLATE.format(
        subject=subject,
        session=session,
        ext=ext
    )
    if subject in CHECK_SUBJECTS_SES_01:
        # subject 116 does not comply with naming convention due to
        # typo in name --> this fixes that
        if subject == 116:
            # remove the wrong name from string
            fname = re.sub('_0116', '_00116', fname)

elif session == 2:
    ext = '_2' + ext
    fname = FNAME_SOURCEDATA_TEMPLATE.format(
        subject=subject,
        session=session,
        ext=ext
    )
    if subject in CHECK_SUBJECTS_SES_02:
        # subject 116 does not comply with naming convention due to
        # typo in name --> this fixes that
        if subject == 43:
            # remove the wrong name from string
            fname = re.sub('_2', '_3', fname)
else:
    raise RuntimeError("Invalid session number provided. Session number"
                       " must be 1 or 2.")

# %%
# 1) import the data
raw = read_raw_brainvision(fname,
                           eog=['vEOG_o', 'vEOG_u'],
                           preload=False)
# get sampling frequency
sfreq = raw.info['sfreq']

# %%
# 2) export to bids

# create bids path
run = 1
output_path = BIDSPath(subject=f'{subject:03}',
                       task='multiple',
                       session=str(session),
                       run=run,
                       datatype='eeg',
                       root=FPATH_BIDS)
# write file
write_raw_bids(raw,
               output_path,
               overwrite=overwrite)

# %%
# 3) check for subjects with more than one file.
# Due to technical problems some sessions had to be saved in
# two different files
add_file = False
if session == 1:
    if subject in CHECK_SUBJECTS_SES_01:
        if subject == 116:
            sys.exit()
        else:
            ext = '_' + CHECK_SUBJECTS_SES_01[subject].split(',')[-2].split('_')[-1]
            run = CHECK_SUBJECTS_SES_01[subject].split(',')[-1].split(' ')[-1]

        fname = FNAME_SOURCEDATA_TEMPLATE.format(
            subject=subject,
            session=session,
            ext=ext
        )
        if subject == 119:
            fname = re.sub('_0119_', '_', fname)
        add_file = True

elif session == 2:
    if subject == 43:
        sys.exit()
    elif subject == 50:
        ext = '_' + CHECK_SUBJECTS_SES_02[subject].split(',')[-2].split('_')[-1]
        run = CHECK_SUBJECTS_SES_02[subject].split(',')[-1].split(' ')[-1]
    elif subject == 52:
        ext = '_' + CHECK_SUBJECTS_SES_02[subject].split(',')[-2].split('_')[-1]
        run = CHECK_SUBJECTS_SES_02[subject].split(',')[-1].split(' ')[-1]
    elif subject in {73, 80}:
        ext = '_2_' + CHECK_SUBJECTS_SES_02[subject].split(',')[-2].split('_')[-1]
        run = CHECK_SUBJECTS_SES_02[subject].split(',')[-1].split(' ')[-1]
    elif subject == 135:
        ext = '_2_' + CHECK_SUBJECTS_SES_02[subject].split(',')[-2].split('_')[-1]
        run = CHECK_SUBJECTS_SES_02[subject].split(',')[-1].split(' ')[-1]
    elif subject == 138:
        ext = '_' + CHECK_SUBJECTS_SES_02[subject].split(',')[-2].split('_')[-1]
        run = CHECK_SUBJECTS_SES_02[subject].split(',')[-1].split(' ')[-1]

    fname = FNAME_SOURCEDATA_TEMPLATE.format(
        subject=subject,
        session=session,
        ext=ext
    )
    add_file = True

if add_file:
    # run BIDS transform for second file
    raw = read_raw_brainvision(fname,
                               eog=['vEOG_o', 'vEOG_u'],
                               preload=False)
    sfreq = raw.info['sfreq']
    # raw.set_montage(montage)
    # create bids path
    output_path = BIDSPath(subject=f'{subject:03}',
                           task='multiple',
                           session=str(session),
                           run=int(run),
                           datatype='eeg',
                           root=FPATH_BIDS)
    # write file
    write_raw_bids(raw,
                   output_path,
                   overwrite=overwrite)
