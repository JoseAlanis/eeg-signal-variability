"""
===========================
Create sourcedata directory
===========================

Put EEG data files in subject specific directory structure
for easier porting to EEG-BIDS.

Authors: José C. García Alanis <alanis.jcg@gmail.com>

License: BSD (3-clause)
"""
# %%
# imports
import sys
import os
from pathlib import Path
import shutil

from mne.utils import logger

from config import (
    FPATH_RAWDATA,
    MISSING_FPATH_RAWDATA_MSG,
    FNAME_RAW_VHDR_SES_1_TEMPLATE,
    FNAME_RAW_VHDR_SES_2_TEMPLATE,
    FNAME_SOURCEDATA_TEMPLATE,
    SUBJECT_IDS,
)

from utils import parse_overwrite

# %%
# default settings (use subject 1, don't overwrite output files)
subject = 1
session = 1
overwrite = False

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

if not os.path.exists(FPATH_RAWDATA):
    raise RuntimeError(MISSING_FPATH_RAWDATA_MSG.format(FPATH_RAWDATA))
if overwrite:
    logger.info(
        f"\n    > `overwrite` has been set to ``True`` "
        f"but that functionality has been disabled in this script."
    )

# %%
# get raw data files and move them to subject (and session) specific directories

# path to raw data
if session == 1:
    fname_raw = FNAME_RAW_VHDR_SES_1_TEMPLATE.format(subject=subject)
    # path to new sourcedata directory
    fname_sourcedata = FNAME_SOURCEDATA_TEMPLATE.format(
        subject=subject, session=session, ext=".vhdr"
    )
elif session == 2:
    fname_raw = FNAME_RAW_VHDR_SES_2_TEMPLATE.format(subject=subject)
    fname_sourcedata = FNAME_SOURCEDATA_TEMPLATE.format(
        subject=subject, session=session, ext="_2.vhdr"
    )
else:
    raise RuntimeError(
        f"\n    > Invalid session number provided. " 
        f"Session number must be 1 or 2."
    )

if os.path.isfile(fname_sourcedata):
    logger.info(
        f"\n    > Subject %s, session %s already in %s"
        f"\n    > Skipping." % (subject, session, Path(fname_sourcedata).parent)
    )
    sys.exit(0)

# check if directory exists (if not created; no overwrite)
Path(fname_sourcedata).parent.mkdir(parents=True, exist_ok=True)

if Path(fname_raw).exists():
    fnames_raw = [
        fname_raw.split(".")[0] + ext
        for ext in[".vhdr", ".eeg", ".vmrk"]
    ]
    fnames_sourcedata = [
        fname_sourcedata.split(".")[0] + ext
        for ext in [".vhdr", ".eeg", ".vmrk"]
    ]

    for raw, sourcedata in zip(fnames_raw, fnames_sourcedata):
        print("\n    > ", raw, " -> ", sourcedata)
        shutil.move(raw, sourcedata)
