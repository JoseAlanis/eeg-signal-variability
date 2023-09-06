import glob

import os.path as op

import numpy as np

import pandas as pd

from config import FPATH_DERIVATIVES

# files in directory
files = glob.glob(
    op.join(
        FPATH_DERIVATIVES,
        'power_spectral_density/oddeven_postcue/*.tsv'
    )
)

# native frequency range
freqs = np.arange(0.0, 500.5, 0.5)

# loop through files
for file in files:

    psd_results = pd.read_table(file)

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

    # save to disk
    psd_results.to_csv(file, index=False, sep='\t', float_format='%.5f')
