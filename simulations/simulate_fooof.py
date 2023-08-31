import matplotlib.pyplot as plt

import numpy as np

# import sim functions
from neurodsp.sim.combined import sim_combined
from neurodsp.sim import sim_oscillation
from neurodsp.utils import set_random_seed

# import function to compute power spectra
from neurodsp.spectral import compute_spectrum

# import utilities for plotting data
from neurodsp.utils import create_times
from neurodsp.plts.spectral import plot_power_spectra
from neurodsp.plts.time_series import plot_time_series

# Set some general settings, to be used across all simulations
fs = 1000
n_seconds = 2
times = create_times(n_seconds, fs)

# Define oscillation frequency
osc_freq = 6

# Simulate a sinusoidal oscillation
osc_sine = sim_oscillation(n_seconds, fs, osc_freq, cycle='sine')

# compute power spectra
freqs, psd_osc = compute_spectrum(osc_sine, fs)
# psd_osc = psd_osc / psd_osc.sum()

# Define the components of the combined signal to simulate
components_1 = {
    'sim_powerlaw':
        {
            'exponent': -2,
            'f_range': [1, None]
         },
    'sim_oscillation':
        {
            'freq': osc_freq
        }
}

components_2 = {
    'sim_powerlaw':
        {
            'exponent': -2,
            'f_range': [1, None]
        },
    'sim_oscillation':
        [{'freq': 6}, {'freq': 10}, {'freq': 14}, {'freq': 50}]

}

# Simulate an oscillation over an aperiodic component
set_random_seed(0)
signal = sim_combined(n_seconds, fs, components_1)
set_random_seed(0)
signal_complex = sim_combined(n_seconds, fs, components_2)

# compute power spectra
#freqs, psd_sig = compute_spectrum(signal, fs)
# psd_sig = psd_sig / psd_sig.sum()

from scipy.signal import periodogram
freqs, psd_sig = periodogram(signal,
                         detrend='constant',
                         fs=fs,
                         nfft=fs * 2,
                         window='hamming')
# psd_sig = psd_sig / psd_sig.sum()
psd_short = psd_sig[
    [(freq >= 1) & (freq <= 50) for freq in freqs]]
freqs_short = freqs[
    [(freq >= 1) & (freq <= 50) for freq in freqs]]

freqs, psd_sig_complex = periodogram(signal_complex,
                             detrend='constant',
                             fs=fs,
                             nfft=fs * 2,
                             window='hamming')
# psd_sig = psd_sig / psd_sig.sum()
psd_short_complex = psd_sig_complex[
    [(freq >= 1) & (freq <= 50) for freq in freqs]]

components_no_osc = {
    'sim_synaptic_current':
        {
            'n_neurons': 1000,
            'firing_rate': 2,
            't_ker': 1.0,
            'tau_r': 0.002,
            'tau_d': 0.02
        },
    'sim_powerlaw':
        {
            'exponent': -2,
            'f_range': [1, None]
        }
}

# Simulate an oscillation over only aperiodic component
signal_no_osc = sim_combined(n_seconds, fs, components_no_osc)

# compute power spectra
freqs, psd_no_osc = compute_spectrum(signal - osc_sine, fs)
# psd_no_osc = psd_no_osc / psd_no_osc.sum()

from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
fm = FOOOF(verbose=False)
fm.fit(freqs, psd_sig)
exp = fm.get_params('aperiodic_params', 'exponent')
off = fm.get_params('aperiodic_params', 'offset')

aperiodic = off - np.log(freqs**exp)

fm.fit(freqs, psd_sig, [2, 40])
exp_s = fm.get_params('aperiodic_params', 'exponent')
off_s = fm.get_params('aperiodic_params', 'offset')

aperiodic_s = off_s - np.log(freqs[(freqs >= 2) & (freqs <= 40)]**exp_s)

widths = [1.0, 0.5]
gs_kw = dict(width_ratios=widths,
             height_ratios=[1.0, 1.0, 1.0],
             wspace=0.10,
             hspace=0.10)
mnames_top = ['signal_1', 'specs_1']
mnames_mid = ['signal_2', 'specs_2']
mnames_bottom = ['signal_3', 'specs_3']

figsize = (8, 6)
fig, axd = plt.subplot_mosaic([mnames_top,
                               mnames_mid,
                               mnames_bottom],
                              gridspec_kw=gs_kw,
                              figsize=figsize,
                              layout='constrained')

axd['signal_1'].axhline(y=0, xmin=0, xmax=2500,
                        color='black', linestyle='dotted', linewidth=.5)
axd['signal_1'].plot(signal, color='k', linewidth=.9)
axd['signal_1'].set_ylim((-4.0, 4.0))
axd['signal_1'].set_ylabel('Voltage ($\mu$V)')
axd['signal_1'].set_xlabel('Time (milliseconds)')
axd['signal_1'].set_title('Less Complex Signal')

axd['specs_1'].plot(freqs_short, psd_short, color='k')
axd['specs_1'].set_ylim((0, 0.9))
axd['specs_1'].set_xlim((0, 30))
axd['specs_1'].set_ylabel('$PSD_{~V^2 / Hz}$')
axd['specs_1'].set_xlabel('Frequency (Hz)')
axd['specs_1'].yaxis.set_label_position("right")
axd['specs_1'].yaxis.tick_right()
axd['specs_1'].set_title('Frequency Contributors')

axd['signal_2'].axhline(y=0, xmin=0, xmax=2500,
                        color='black', linestyle='dotted', linewidth=.5)
axd['signal_2'].plot(signal_complex, color='k', linewidth=1.0)
axd['signal_2'].set_ylim((-4.0, 4.0))
axd['signal_2'].set_ylabel('Voltage ($\mu$V)')
axd['signal_2'].set_xlabel('Time (milliseconds)')
axd['signal_2'].set_title('More Complex Signal')

axd['specs_2'].plot(freqs_short, psd_short_complex, color='k')
axd['specs_2'].set_ylim((0, 0.9))
axd['specs_2'].set_xlim((0, 30))
axd['specs_2'].set_ylabel('$PSD_{~V^2 / Hz}$')
axd['specs_2'].set_xlabel('Frequency (Hz)')
axd['specs_2'].yaxis.set_label_position("right")
axd['specs_2'].set_title('Frequency Contributors')
axd['specs_2'].yaxis.tick_right()

fig.savefig('1f_sim.png', dpi=300)

# plot_time_series(times, signal)
#
# plot_power_spectra(freqs, psd)
#
# # Simulation settings
# n_seconds = 1
# times = create_times(n_seconds, fs)
#
# # Define oscillation frequency
# osc_freq = 10
#
# # Simulate a sinusoidal oscillation
# osc_sine = sim_oscillation(n_seconds, fs, osc_freq, cycle='sine')
# # Plot the simulated data, in the time domain
# plot_time_series(times, osc_sine)
#
# import matplotlib.pyplot as plt
# plt.plot(osc_sine)
# plt.plot(signal)
#
# import numpy as np
# dx = np.diff(osc_sine, axis=-1)
#
# plt.plot(dx)
# ddx = np.diff(dx, axis=-1)
# plt.plot(ddx)
#
# x_var = np.var(osc_sine, axis=-1)  # = activity
# dx_var = np.var(dx, axis=-1)
# ddx_var = np.var(ddx, axis=-1)
#
# plt.plot(osc_sine)
# plt.plot(dx)
# plt.plot(ddx)
#
# mob = np.sqrt(dx_var / x_var)
# com = np.sqrt(ddx_var / dx_var) / mob
#
# from neurodsp.sim import sim_synaptic_current
# sig = sim_synaptic_current(n_seconds=1, fs=500)
#
# plt.plot(sig)