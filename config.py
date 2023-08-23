"""
========================
Study configuration file
========================

Configuration parameters and global values that will be used across scripts.

Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
License: MIT
"""
import os
import multiprocessing

from pathlib import Path

import numpy as np

import json

from mne.channels import make_dig_montage

# -----------------------------------------------------------------------------
# check number of available CPUs in system
jobs = multiprocessing.cpu_count()
os.environ["NUMEXPR_MAX_THREADS"] = str(jobs)

# -----------------------------------------------------------------------------
# problematic subjects
NO_DATA_SUBJECTS = {
    4: "no data; subject requested deletion of data",
    34: "no data; subject requested deletion of data",
    103: "no data; subject requested deletion of data"
}
CHECK_SUBJECTS_SES_01 = {
    43: "Experiment crashed, task 'oddeven' in Exp23_0043_2.vhdr, add run 2",
    116: "Typo in file name, data in Exp23_00116.vhdr, add run 1",
    119: "Experiment crashed, task 'oddeven' in Exp23_01191.vhdr, add run 2"
}

CHECK_SUBJECTS_SES_02 = {
    6: "Experiment crashed, task 'numberletter' in Exp23_0043_2.vhdr, add run 2",
    43: "Typo in file name, data in Exp23_0043_3.vhdr, add run 1",
    50: "Experiment crashed, data in Exp23_0050_22.vhdr, add run 2",
    52: "Experiment crashed, data in Exp23_0052_21.vhdr, add run 2",
    73: "Experiment crashed, data in Exp23_0073_2_2.vhdr, add run 2",
    80: "Experiment crashed, data in Exp23_0080_2_2.vhdr, add run 2",
    119: "Experiment crashed, data in Exp23_0119_2_2.vhdr, add run 2",
    135: "Experiment crashed, data in Exp23_0080_2_Posner.vhdr, add run 2",
    138: "Experiment crashed, data in Exp23_0119_21.vhdr, add run 2"
}

# session 1 bad data
BAD_SUBJECTS_SES_01 = {
    113: "no oddeven task",
    125: "ground/reference problems",
    138: "ground/reference problems"
}

# session 2 bad data
BAD_SUBJECTS_SES_02 = {
    27: "no data",
    63: "no numberletter task",
    24: "ground/reference problems",
    47: "ground/reference problems",
    57: "ground/reference problems",
    79: "no data",
    90: "no data",
    94: "ground/reference problems"
}

# valid subject IDs, originally, subjects from 1 to 151
CANDIDATE_SUBJECTS = set(np.arange(1, 152))
# but some subjects should be excluded
SUBJECT_IDS = np.array(list(CANDIDATE_SUBJECTS - set(NO_DATA_SUBJECTS)))

# -----------------------------------------------------------------------------
# file paths
with open("./paths.json") as paths:
    paths = json.load(paths)

# path to raw data (original BrainVision directory)
FPATH_RAWDATA = Path(paths['rawdata'])
# path to sourcedata (restructured BrainVision files)
FPATH_SOURCEDATA = Path(paths["sourcedata"])
# path to BIDS compliant directory structure
FPATH_BIDS = Path(paths["bids"])
# path to derivatives
FPATH_DERIVATIVES = Path(os.path.join(FPATH_BIDS, 'derivatives'))

# -----------------------------------------------------------------------------
# file name templates
# the paths raw data in brainvision format (.vhdr)
FNAME_RAW_VHDR_SES_1_TEMPLATE = os.path.join(
    str(FPATH_RAWDATA), "Exp23_{subject:04}.vhdr"
)
FNAME_RAW_VHDR_SES_2_TEMPLATE = os.path.join(
    str(FPATH_RAWDATA), "Exp23_{subject:04}_2.vhdr"
)

# the path to the sourcedata directory
FNAME_SOURCEDATA_TEMPLATE = os.path.join(
    str(FPATH_SOURCEDATA),
    "sub-{subject:03}",
    "ses-{session:02}",
    "eeg",
    "Exp23_{subject:04}{ext}"
)

# -----------------------------------------------------------------------------
# default messages
MISSING_FPATH_RAWDATA_MSG = (
    "\n    > Could not find the path:\n\n    > {}\n"
    "\n    > Did you define the correct path to the data in `config.py`? "
    "See the `FPATH_RAWDATA` variable in `config.py`.\n"
)

MISSING_FPATH_SOURCEDATA_MSG = (
    "\n    > Could not find the path:\n\n    > {}\n"
    "\n    > Did you define the correct path to the data in `config.py`? "
    "See the `FPATH_SOURCEDATA` variable in `config.py`.\n"
)

MISSING_FPATH_BIDS_MSG = (
    "\n    > Could not find the path:\n\n    > {}\n"
    "\n    > Did you define the correct path to the data in `config.py`? "
    "See the `FPATH_BIDS` variable in `config.py`.\n"
)

# -----------------------------------------------------------------------------
eeg_markers = {
    "ses-1": {
        "oddeven": {
            "markers": {
                "start_odd_even": 6,
                "end_end_odd_even": 7,
                "altend_odd_even": 115,

                "fix_less_more_repeat": 31,
                "isi_less_more_repeat": 41,
                "target_less_more_repeat": 51,
                "fix_less_more_switch": 32,
                "isi_less_more_switch": 42,
                "target_less_more_switch": 52,

                "fix_odd_even_repeat": 33,
                "isi_odd_even_repeat": 43,
                "target_odd_even_repeat": 53,
                "fix_odd_even_switch": 34,
                "isi_odd_even_switch": 44,
                "target_odd_even_switch": 54,

                "iti": 90,

                "correct_response_d": 150,
                "correct_response_l": 160,
                "incorrect_response_d": 250,
                "incorrect_response_l": 255,
                "incorrect_response": 251,
                "miss": 222
            },
            "n_trials": 384
        },
        "globallocal": {
            "markers": {
                "start_global_local": 14,
                "end_global_local": 15,
                "altend_global_local": 114,

                "fix_global_repeat": 31,
                "isi_global_repeat": 41,
                "target_global_repeat": 51,
                "fix_global_switch": 32,
                "isi_global_switch": 42,
                "target_global_switch": 52,

                "fix_local_repeat": 33,
                "isi_local_repeat": 43,
                "target_local_repeat": 53,
                "fix_local_switch": 34,
                "isi_local_switch": 44,
                "target_local_switch": 54,

                "iti": 90,

                "correct_response_left": 150,
                "correct_response_right": 160,
                "incorrect_response_left": 250,
                "incorrect_response_right": 255,
                "incorrect_response": 251,
                "miss": 222
            },
            "n_trials": 384
        }
    },
    "ses-2": {
        "numberletter": {
            "markers": {
                "start": 12,
                "end": 13,
                "altend": 119,

                "fix_less_more_repeat": 31,
                "isi_less_more_repeat": 41,
                "target_less_more_repeat": 51,
                "fix_less_more_switch": 32,
                "isi_less_more_switch": 42,
                "target_less_more_switch": 52,

                "fix_vowel_consonant_repeat": 33,
                "isi_vowel_consonant_repeat": 43,
                "target_vowel_consonant_repeat": 53,
                "fix_vowel_consonant_switch": 34,
                "isi_vowel_consonant_switch": 44,
                "target_vowel_consonant_switch": 54,

                "iti": 90,

                "correct_response_d": 150,
                "correct_response_l": 160,
                "incorrect_response_d": 250,
                "incorrect_response_l": 255,
                "incorrect_response": 251,
                "miss": 222
            },
            "n_trials": 256
        }
    }
}
