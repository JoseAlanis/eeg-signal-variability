# Authors: Jose C. Garcia Alanis <alanis.jcg@gmail.com>
#
# License: BSD-3-Clause
import numpy as np

from scipy.signal import periodogram

from fooof import FOOOF

from scipy.stats import linregress

import antropy as ant
from antropy.utils import _xlogx

import neurokit2 as nk


def compute_signal_variability(
        signal, measures='all', freq_lim=None, sfreq=None):
    if measures == 'all':
        measures = ['fooof',
                    's_entropy', 'p_entropy', 'wp_entropy',
                    'mse', 'mse_bins', 'mse_slope',
                    'mobility', 'complexity']

    fooof_measures = np.empty((1, 2))
    fooof_measures[:] = np.nan
    s_ent = np.empty((1, 1))
    s_ent[:] = np.nan
    p_ent = np.empty((1, 1))
    p_ent[:] = np.nan
    wp_ent = np.empty((1, 1))
    wp_ent[:] = np.nan
    ms_ent = np.empty((1, 1))
    ms_ent[:] = np.nan
    ms_ent_bins = np.empty((1, 4))
    ms_ent_bins[:] = np.nan
    ms_ent_slope = np.empty((1, 1))
    ms_ent_slope[:] = np.nan
    mobl = np.empty((1, 1))
    mobl[:] = np.nan
    compl = np.empty((1, 1))
    compl[:] = np.nan

    # sampling frequency
    if sfreq is None:
        fs = len(signal)
    else:
        fs = sfreq

    # compute psd
    freqs, psd = periodogram(signal - np.mean(signal),
                             detrend=False,
                             fs=fs,
                             nfft=fs * 2,
                             window='hamming')
    psd = psd / psd.sum()
    if freq_lim is None:
        psd_short = psd.copy()
    else:
        psd_short = psd[
            [(freq >= freq_lim[0]) & (freq <= freq_lim[1]) for freq in freqs]]
        freqs_short = freqs[
            [(freq >= freq_lim[0]) & (freq <= freq_lim[1]) for freq in freqs]]

    if 'fooof' in measures:
        # compute 1/f measures
        fm = FOOOF(verbose=False)
        fm.fit(freqs_short, psd_short)
        exp = fm.get_params('aperiodic_params', 'exponent')
        off = fm.get_params('aperiodic_params', 'offset')
        fooof_measures[0, 0] = exp
        fooof_measures[0, 1] = off

        # aperidic_fit = (10 ** off ) * (1 / freqs_short ** exp)
        # or
        # aperidic_fit = off - np.log10(freqs_short**exp)

    if 's_entropy':
        # compute spectral entropy
        se = -_xlogx(psd_short).sum(axis=0)
        se /= np.log2(len(psd_short))

        s_ent[0] = se

    if 'p_entropy' in measures:
        # compute permutation entropy
        pent, _ = nk.entropy_permutation(signal,
                                         delay=1,
                                         dimension=3,
                                         corrected=True,
                                         weighted=False)

        p_ent[0] = pent

    if 'wp_entropy' in measures:
        # compute weighted permutation entropy
        wpent, _ = nk.entropy_permutation(signal,
                                          delay=1,
                                          dimension=3,
                                          corrected=True,
                                          weighted=True)

        wp_ent[0] = wpent

    if 'mse' in measures:
        # compute multi-scale entropy
        # (i.e., area under the MSE values curve)
        mse, info = nk.entropy_multiscale(signal,
                                          dimension=3,
                                          scale=20,
                                          show=False)

        ms_ent[0] = mse

        if 'mse_bins' in measures:

            # remove non-finite and zeros
            bin1 = info['Value'][0:5]
            bin1 = bin1[np.isfinite(bin1)]
            # bin1 = [val if np.isfinite(val) else np.nan for val in bin1]

            bin2 = info['Value'][5:10]
            bin2 = bin2[np.isfinite(bin2)]
            # bin2 = [val if np.isfinite(val) else np.nan for val in bin2]

            bin3 = info['Value'][10:15]
            bin3 = bin3[np.isfinite(bin3)]
            # bin3 = [val if np.isfinite(val) else np.nan for val in bin3]

            bin4 = info['Value'][15:20]
            bin4 = bin4[np.isfinite(bin4)]
            # bin4 = [val if np.isfinite(val) else np.nan for val in bin4]

            # compute MSE for different scales
            if len(bin1) > 1:
                ms_ent_bins[0, 0] = np.trapz(bin1) / len(bin1)
            else:
                ms_ent_bins[0, 0] = -np.inf

            if len(bin2) > 1:
                ms_ent_bins[0, 1] = np.trapz(bin2) / len(bin2)
            else:
                ms_ent_bins[0, 1] = -np.inf

            if len(bin3) > 1:
                ms_ent_bins[0, 2] = np.trapz(bin3) / len(bin3)
            else:
                ms_ent_bins[0, 2] = -np.inf

            if len(bin4) > 1:
                ms_ent_bins[0, 3] = np.trapz(bin4) / len(bin4)
            else:
                ms_ent_bins[0, 3] = -np.inf

        if 'mse_slope' in measures:
            # remove non-finite and zeros
            mse_values = info['Value'][np.isfinite(info['Value'])]
            mse_values = mse_values[np.nonzero(mse_values)]
            # compute MSE slope
            slope, intercept, _, _, _ = \
                linregress(np.arange(1, len(mse_values) + 1), mse_values)

            ms_ent_slope[0] = slope

    if 'mobility' in measures:
        # compute Hjorth mobility parameter
        mob, _ = ant.hjorth_params(signal)
        mobl[0] = mob

    if 'complexity' in measures:
        # compute Hjorth complexity parameters
        _, comp = ant.hjorth_params(signal)
        compl[0] = comp

    var_measures = ['exp_1f', 'off_1f',
                    'shannon_entropy',
                    'permutation_entropy',
                    'wpermutation_entropy',

                    'ms_entropy',
                    'ms1-5_entropy',
                    'ms5-10_entropy',
                    'ms10-15_entropy',
                    'ms15-20_entropy',

                    'ms-slope_entropy',

                    'hjorth_mobility',
                    'hjorth_complexity']

    var_vals = np.array([fooof_measures[0, 0],
                         fooof_measures[0, 1],
                         s_ent[0, 0],
                         p_ent[0, 0],
                         wp_ent[0, 0],
                         ms_ent[0, 0],
                         ms_ent_bins[0, 0],
                         ms_ent_bins[0, 1],
                         ms_ent_bins[0, 2],
                         ms_ent_bins[0, 3],
                         ms_ent_slope[0, 0],
                         mobl[0, 0], compl[0, 0]])

    values = var_vals[var_vals != np.nan]
    names = [mes for val, mes in zip(var_vals, var_measures) if val != np.nan]

    return values, names