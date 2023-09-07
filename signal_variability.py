# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD-3-Clause
import warnings

from joblib import Parallel, delayed
from tqdm import tqdm

import numpy as np

from fooof import FOOOF

import antropy as ant

import neurokit2 as nk


def spectral_entropy(psd):
    """
    Calculate the spectral entropy of a given power spectrum.

    Spectral entropy measures the complexity or unpredictability of a signal's
    frequency representation. This function handles zero or negative values in
    the PSD in a special way:
    It returns x * log(x) if x is positive, 0 if x == 0, and `np.nan` otherwise

    Inspired by the `antropy` package:
    https://raphaelvallat.com/antropy/

    Parameters
    ----------
    psd : array_like
        Power spectrum density of a signal. Should be a 1D array or list of
        non-negative values.

    Returns
    -------
    float
        Spectral entropy of the provided PSD (in bits).

    """
    x = np.asarray(psd)
    xlogx = np.zeros(x.shape)
    xlogx[x < 0] = np.nan
    valid = x > 0
    xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(2)

    se = -xlogx.sum() / np.log2(len(x))

    return se


def frequency_variability(psd, freqs, freq_lim=None):
    """
    Compute frequency variability metrics for a given power spectrum.

    This function calculates the aperiodic components (one-over-f exponent and
    offset) for the power spectrum using the `FOOOF` package
    (https://fooof-tools.github.io/fooof/index.html). It also computes the
    spectral entropy of the power spectrum.
    The calculation is limited to a specified frequency range if
    `freq_lim` is provided.

    Parameters
    ----------
    psd : array_like
        Power spectrum density (PSD) of a signal. Should be a 1D array or list.

    freqs : array_like
        List of frequency values corresponding to the PSD.

    freq_lim : tuple, optional
        Lower and upper frequency bounds as a tuple (low_freq, high_freq).
        If not provided, the entire frequency range is considered.

    Returns
    -------
    dict
        A dictionary containing:
        - '1f_offset': one-over-f offset.
        - '1f_exponent': one-over-f exponent.
        - 'spectral_entropy': Spectral entropy of the PSD.

    Notes
    -----
    `FOOOF` might encounter numerical issues during computation.
    In such cases, the function will catch exceptions and return `np.nan`
    for the affected metric.

    Examples
    --------
    >>> psd_values = [1, 2, 3, 4, 5]
    >>> freqs_values = [0.5, 1, 1.5, 2, 2.5]
    >>> frequency_variability(psd_values, freqs_values)
    """

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

    # compute one-over-f spectral components
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
    """
    Compute signal variability metrics for a signal.

    This function calculates multiple metrics that assess the variability
    or complexity of a signal, such as permutation entropy, multiscale entropy,
    and Hjorth parameters.

    Parameters
    ----------
    signal : array_like
        1D array or list containing the signal data points.

    Returns
    -------
    dict
        Dictionary containing various metrics assessing the variability
        of the signal:
        - 'permutation_entropy': Permutation entropy of the signal.
        - 'weighted_permutation_entropy': Weighted permutation entropy.
        - 'multi-scale entropy': Multiscale entropy values.
        - 'multi-scale entropy (1-4)': Binned multiscale entropy values.
        - 'multi-scale entropy (slope)': Slope of the multiscale entropy curve.
        - 'activity': Variance of the signal scaled by 1e6.
        - 'mobility': Mobility value from Hjorth parameters.
        - 'complexity': Complexity value from Hjorth parameters.

    Notes
    -----
    Some metrics might encounter numerical issues during computation.
    In such cases, the function will catch exceptions and return `np.nan`
    for the affected metric.

     Examples
    --------
    >>> signal_data = np.random.rand(1000)
    >>> variability_metrics = signal_variability(signal_data)
    >>> variability_metrics['permutation_entropy']
    """

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


def _process_element(inst, m, n, freqs=None):
    """
    This function computes either the frequency variability
    (if `freqs` is provided) or the signal variability (if `freqs` is
    not provided) for a specific element located at the coordinates (m, n)
    of the instance.
    """
    if freqs is not None:
        return frequency_variability(inst[m, n, :], freqs, freq_lim=[2.0, 45.0])
    else:
        return signal_variability(inst[m, n, :])


def parallel_analysis(inst, freqs=None, jobs=1):
    """
    Run parallel analysis on a given instance. The intance can be an array of
    amplitude values or power spectral density values for a set of EEG segments.

    The function divides the instance into tasks and processes them in parallel.
    Each task corresponds to a combination of EEG segments and EEG sensors.

    Parameters
    ----------
    inst : array_like, shape (n_epochs, n_channels, n_times or n_frequencies)
        The time series data consisting of n_epochs for separate observations
        of signals with n_channels time-series of length n_times.

    freqs : array_like, optional
        List of frequency values. If provided, `inst` is assumed to be an array
        of power spectral density values and analysis is computed
        "in the frequency domain".

    jobs : int, optional
        Number of CPU cores to use for parallel processing.
        Default is 1 (no parallelism).

    Returns
    -------
    tuple
        - results: 3D array containing the output from the analysis.
        - measures: List of measures used in the analysis.

    Examples
    --------
    >>> x = np.random.rand(20, 10, 800)
    >>> results, measures = parallel_analysis(x, jobs=2)  # noqa
    """
    if freqs is not None:
        results = np.empty((3, inst.shape[0], inst.shape[1]))
    else:
        results = np.empty((11, inst.shape[0], inst.shape[1]))

    total_tasks = inst.shape[0] * inst.shape[1]
    progress_bar = tqdm(total=total_tasks, desc="Processing", unit="task")

    tasks = [(i, j) for i in range(inst.shape[0]) for j in range(inst.shape[1])]
    batch_size = total_tasks // 10

    measures = None
    for t in range(0, len(tasks), batch_size):
        batch = tasks[t:t + batch_size]
        output_list = Parallel(n_jobs=jobs)(
            delayed(_process_element)(inst, i, j, freqs)
            for (i, j) in batch
        )

        for idx, (i, j) in enumerate(batch):
            out = output_list[idx]
            measures = list(out.keys())
            for meas in range(len(measures)):
                results[meas, i, j] = out[measures[meas]]

            progress_bar.update(1)

    progress_bar.close()
    return results, measures
