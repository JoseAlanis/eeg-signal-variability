# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD-3-Clause
import warnings

from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import numpy as np

from fooof import FOOOF

import antropy as ant

import neurokit2 as nk


def spectral_entropy(x):
    # This part handles the case when the power spectrum density
    # takes any zero value.
    # It returns x * log(x) if x is positive, 0 if x == 0, and np.nan otherwise.
    # (taken from antropy package: https://raphaelvallat.com/antropy/
    x = np.asarray(x)
    xlogx = np.zeros(x.shape)
    xlogx[x < 0] = np.nan
    valid = x > 0
    xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(2)

    se = -xlogx.sum() / np.log2(len(x))

    return se


def frequency_variability(psd, freqs, freq_lim=None):

    if freq_lim is None:
        lwf = freqs[0]
        upf = freqs[-1]
    else:
        lwf = freq_lim[0]
        upf = freq_lim[-1]

    psd = psd[
        [(freq >= lwf) & (freq <= upf) for freq in freqs]
    ]
    freqs = freqs[
        [(freq >= lwf) & (freq <= upf) for freq in freqs]
    ]

    # compute 1/f measures
    try:
        with warnings.catch_warnings():
            # just in case weird numerical issues arise
            warnings.simplefilter("error")
            fm = FOOOF(verbose=False)
            fm.fit(freqs, psd)
            exp = fm.get_params('aperiodic_params', 'exponent')
            off = fm.get_params('aperiodic_params', 'offset')
    except Exception as err:
        print("An error occurred:" % (), err)
        exp = np.nan
        off = np.nan

    # compute spectral entropy
    se = spectral_entropy(psd)

    return {'1f_offset': off,
            '1f_exponent': exp,
            'spectral_entropy': se}


def signal_variability(signal):

    # permutation entropy
    pent, _ = nk.entropy_permutation(signal,
                                     delay=1,
                                     dimension=3,
                                     corrected=True,
                                     weighted=False)
    # weighted permutation entropy
    wpent, _ = nk.entropy_permutation(signal,
                                      delay=1,
                                      dimension=3,
                                      corrected=True,
                                      weighted=True)
    # multiscale entropy
    mse, info = nk.entropy_multiscale(signal,
                                      dimension=3,
                                      scale=20,
                                      show=False)
    # multiscale entropy bins (from fine to coarse)
    bins = np.array_split(info['Value'], 4)
    mse_bins = []
    for mbin in bins:
        try:
            with warnings.catch_warnings():
                # just in case weird numerical issues arise
                warnings.simplefilter("error")
                me = np.trapz(mbin[np.isfinite(mbin)]) / len(np.isfinite(mbin))
                mse_bins.append(me)
        except Exception as err:
            print("An error occurred in MSE (bins) computation:" % (), err)
            mse_bins.append(np.nan)

    try:
        with warnings.catch_warnings():
            # just in case weird numerical issues
            mse_slope = np.polyfit(
                np.arange(0, len(info['Value'][np.isfinite(info['Value'])])),
                info['Value'][np.isfinite(info['Value'])],
                deg=1
            )[0]
    except Exception as err:
        print("An error occurred in MSE (slope) computation:" % (), err)
        mse_slope = np.nan

    # hjorth parameters
    act = np.var(signal * 1e6)
    mob, comp = ant.hjorth_params(signal)

    return {'permutation_entropy': pent,
            'weighted_permutation_entropy': wpent,
            'multi-scale entropy': mse,
            'multi-scale entropy (1)': mse_bins[0],
            'multi-scale entropy (2)': mse_bins[1],
            'multi-scale entropy (3)': mse_bins[2],
            'multi-scale entropy (4)': mse_bins[3],
            'multi-scale entropy (slope)': mse_slope,
            'activity': act,
            'mobility': mob,
            'complexity': comp}


# compute measures in parallel
def parallel_analysis(inst, freqs=None, jobs=1):

    if freqs is not None:
        # run frequency analysis
        results = np.empty((3, inst.shape[0], inst.shape[1]))

        # the analysis that should be performed
        def process_element(m, n):
            return frequency_variability(
                inst[m, n, :], freqs, freq_lim=[2.0, 45.0])

    else:
        # run amplitude analysis
        results = np.empty((11, inst.shape[0], inst.shape[1]))

        def process_element(m, n):
            return signal_variability(inst[m, n, :])

    with ThreadPoolExecutor(max_workers=jobs) as executor:

        total_tasks = inst.shape[0] * inst.shape[1]
        progress_bar = tqdm(total=total_tasks, desc="Processing", unit="task")

        for i in range(inst.shape[0]):
            for j in range(inst.shape[1]):
                out = executor.submit(process_element, i, j).result()

                measures = list(out.keys())
                for meas in range(len(measures)):
                    results[meas, i, j] = out[measures[meas]]

                progress_bar.update(1)  # Update the progress bar

        progress_bar.close()

    return results, measures
