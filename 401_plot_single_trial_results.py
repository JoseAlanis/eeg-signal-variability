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
import glob

import pyreadr

import re

import numpy as np

from mne import read_epochs
from mne.utils import logger
from mne.viz import plot_topomap

import matplotlib as mpl
import matplotlib.pyplot as plt

# all parameters are defined in config.py
from config import (
    FPATH_DERIVATIVES)

# %%
# default settings (use subject 1, don't overwrite output files)
overwrite = False
subject = 1
session = 1
task = 'oddeven'
window = 'pre'
stimulus = 'cue'

# generic epochs structure
FPATH_EPOCHS = (os.path.join(FPATH_DERIVATIVES,
                            'epochs',
                            'sub-%s' % f'{subject:03}',
                             'sub-%s_task-%s_%s%s-epo.fif'
                             % (f'{subject:03}', task, window, stimulus)))
epochs = read_epochs(FPATH_EPOCHS, preload=False)

# %%
# paths and overwrite settings
if overwrite:
    logger.info("`overwrite` is set to ``True`` ")

# %%
which_measures = [
    "permutation_entropy", "weighted_permutation_entropy",
    "multiscale_entropy", "ms_1", "ms_2", "ms_3", "ms_4",
    "activity", "mobility", "complexity",
    "1f_offset", "1f_exponent",
    "spectral_entropy"
]

# %%
# import the data

# create path for model fits
FPATH_FITS_ODDEVEN = os.path.join(FPATH_DERIVATIVES,
                                  'analysis_dataframes',
                                  'oddeven_sensor_*_*_fits_st.rds')
FPATH_FITS_ODDEVEN = glob.glob(FPATH_FITS_ODDEVEN)

# object shape
n_measures = len(which_measures)
n_sensors = 32

# load results for each sensor
fits = np.empty((n_sensors, n_measures, 3))
p_vals = np.empty((n_sensors, n_measures, 3))

for fpath in FPATH_FITS_ODDEVEN:
    basename = os.path.basename(fpath)
    m = [m for m in which_measures if m in basename]
    m = which_measures.index(m[0])
    sj = re.search(r'\d+', os.path.basename(fpath)).group(0)
    fits[int(sj), m, :] = pyreadr.read_r(fpath)[None].Omega2_partial
    # p_vals[int(sj), ...] = pyreadr.read_r(fpath)[None].p
    # p_vals[int(sj), m, ...] = pyreadr.read_r(fpath)[None].o_sq_CI_low > 0.06

# # correct p-values
# p_vals = multipletests(p_vals.flatten(), method='bonferroni')[0]
# p_vals = p_vals.reshape((n_sensors, n_measures))

# create path for contrasts
FPATH_CONTRASTS_ODDEVEN = os.path.join(FPATH_DERIVATIVES,
                                       'analysis_dataframes',
                                       'oddeven_sensor_*_*_constrats_st.rds')
FPATH_CONTRASTS_ODDEVEN = glob.glob(FPATH_CONTRASTS_ODDEVEN)

# object shape
which_contrast = [
    'Pre Cue - Post Cue',
    'Pre Cue - Post Target',
    'Post Cue - Post Target',
    'Repeat - Switch',
    'Pre Cue - Post Cue (Repeat)',
    'Pre Cue - Post Target (Repeat)',
    'Post Cue - Post Target (Repeat)',
    'Pre Cue - Post Cue (Switch)',
    'Pre Cue - Post Target (Switch)',
    'Post Cue - Post Target (Switch)'
]
n_contrasts = len(which_contrast)

# load results for each sensor
contrasts = np.empty((n_sensors, n_measures, n_contrasts))
for fpath in FPATH_CONTRASTS_ODDEVEN:
    basename = os.path.basename(fpath)
    m = [m for m in which_measures if m in basename]
    m = which_measures.index(m[0])

    sj_c = re.search(r'\d+', os.path.basename(fpath)).group(0)
    contr = pyreadr.read_r(fpath)[None]

    contrasts[int(sj_c), m, :] = contr.d

# %%
# plot omnibus test results

# measures to plot
measures = which_measures.copy()
measure_labels = ['PE', 'WPE', 'MSE',
                  'MSE$_{1}$', 'MSE$_{2}$', 'MSE$_{3}$', 'MSE$_{4}$',
                  'A', 'M', 'C',
                  r'1/$f$ (exp.)', r'1/$f$ (off.)',
                  'SE']
cmap = mpl.cm.Reds
bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

widths = [2.0] + [1.0 for _ in measures] + [0.3]
heights = [0.25, 1.0, 0.25]
gs_kw = dict(width_ratios=widths,
             height_ratios=heights,
             wspace=0.5)

mnames_top = ['X'] + [str(n) for n, t in enumerate(measures)] + ['X']
manmes_center = ['name'] + [str(n) for n, t in enumerate(measures)] + ['cbar']
mnames_bottom = ['X'] + [str(n) for n, t in enumerate(measures)] + ['X']
fig, ax = plt.subplot_mosaic(mosaic=[mnames_top,
                                     manmes_center,
                                     mnames_bottom],
                             gridspec_kw=gs_kw,
                             empty_sentinel="X",
                             figsize=(25, 2.5))
for n_meas in range(fits.shape[1]):
    plot_topomap(fits[:, n_meas, 0],
                 epochs.info,
                 cmap=cmap,
                 cnorm=norm,
                 mask=p_vals[:, n_meas],
                 mask_params=dict(marker='o',
                                  markerfacecolor='k',
                                  markeredgecolor='k',
                                  linewidth=0, markersize=1),
                 contours=0,
                 size=10,
                 axes=ax[str(n_meas)],
                 sensors=False,
                 show=False)
    ax[str(n_meas)].set_title(r'%s' % measure_labels[int(n_meas)])
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=ax['cbar'],
    extend='both',
    # extendfrac='auto',
    ticks=bounds,
    spacing='uniform',
    orientation='vertical',
    label=r'Effect size ($\omega^2$)',
)
ax['name'].set(yticks=[], yticklabels=[], xticks=[], xticklabels=[])
ax['name'].annotate('Task:\nOdd/Even', (0.1, 0.5), fontsize=14, color='k')
ax['name'].spines['right'].set_visible(False)
ax['name'].spines['top'].set_visible(False)
ax['name'].spines['left'].set_visible(False)
ax['name'].spines['bottom'].set_visible(False)
# ax['cbar'].set_ylabel(r'Effect size ($\omega^2$)', labelpad=10.0)
plt.close('all')
fig.savefig('./measures_o_sq.png', dpi=300)

# %%
cmap_d = mpl.cm.RdBu_r
bounds = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,  0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

widths = [2.0] + [1.0 for _ in measures] + [0.3]
heights = [0.25, 1.0, 0.25]
gs_kw = dict(width_ratios=widths,
             height_ratios=heights,
             wspace=0.5)

mnames_top = ['X'] + [str(n) for n, t in enumerate(measures)] + ['X']
manmes_center = ['name'] + [str(n) for n, t in enumerate(measures)] + ['cbar']
mnames_bottom = ['X'] + [str(n) for n, t in enumerate(measures)] + ['X']
fig, ax = plt.subplot_mosaic(mosaic=[mnames_top,
                                     manmes_center,
                                     mnames_bottom],
                             gridspec_kw=gs_kw,
                             empty_sentinel="X",
                             figsize=(25, 2.5))
for n_meas in range(contrasts.shape[1]):
    plot_topomap(contrasts[:, n_meas, 0] * -1,
                 epochs.info,
                 cmap=cmap_d,
                 cnorm=norm,
                 # mask=p_vals[:, n_meas],
                 # mask_params=dict(marker='o',
                 #                  markerfacecolor='k',
                 #                  markeredgecolor='k',
                 #                  linewidth=0, markersize=1),
                 contours=0,
                 size=10,
                 axes=ax[str(n_meas)],
                 sensors=False,
                 show=False)
    ax[str(n_meas)].set_title(r'%s' % measure_labels[int(n_meas)])
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap_d, norm=norm),
    cax=ax['cbar'],
    extend='both',
    # extendfrac='auto',
    ticks=bounds,
    spacing='uniform',
    orientation='vertical',
    label=r"Effect size (Cohen's $d$)",
)
ax['name'].set(yticks=[], yticklabels=[], xticks=[], xticklabels=[])
ax['name'].annotate('Change:\nBaseline to Cue', (0.1, 0.5),
                    fontsize=12, color='k')
ax['name'].spines['right'].set_visible(False)
ax['name'].spines['top'].set_visible(False)
ax['name'].spines['left'].set_visible(False)
ax['name'].spines['bottom'].set_visible(False)
ax['cbar'].set(yticks=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
               yticklabels=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
plt.close('all')
fig.savefig('./measures_d_pre_cue-post_cue.png', dpi=300)

# %%
cmap_d = mpl.cm.RdBu_r
bounds = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,  0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

widths = [2.0] + [1.0 for _ in measures] + [0.3]
heights = [0.25, 1.0, 0.25]
gs_kw = dict(width_ratios=widths,
             height_ratios=heights,
             wspace=0.5)

mnames_top = ['X'] + [str(n) for n, t in enumerate(measures)] + ['X']
manmes_center = ['name'] + [str(n) for n, t in enumerate(measures)] + ['cbar']
mnames_bottom = ['X'] + [str(n) for n, t in enumerate(measures)] + ['X']
fig, ax = plt.subplot_mosaic(mosaic=[mnames_top,
                                     manmes_center,
                                     mnames_bottom],
                             gridspec_kw=gs_kw,
                             empty_sentinel="X",
                             figsize=(25, 2.5))
for n_meas in range(contrasts.shape[1]):
    plot_topomap(contrasts[:, n_meas, 1] * -1,
                 epochs.info,
                 cmap=cmap_d,
                 cnorm=norm,
                 # mask=p_vals[:, n_meas],
                 # mask_params=dict(marker='o',
                 #                  markerfacecolor='k',
                 #                  markeredgecolor='k',
                 #                  linewidth=0, markersize=1),
                 contours=0,
                 size=10,
                 axes=ax[str(n_meas)],
                 sensors=False,
                 show=False)
    ax[str(n_meas)].set_title(r'%s' % measure_labels[int(n_meas)])
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap_d, norm=norm),
    cax=ax['cbar'],
    extend='both',
    # extendfrac='auto',
    ticks=bounds,
    spacing='uniform',
    orientation='vertical',
    label=r"Effect size (Cohen's $d$)",
)
ax['name'].set(yticks=[], yticklabels=[], xticks=[], xticklabels=[])
ax['name'].annotate('Change:\nCue to Target', (0.1, 0.5), fontsize=12, color='k')
ax['name'].spines['right'].set_visible(False)
ax['name'].spines['top'].set_visible(False)
ax['name'].spines['left'].set_visible(False)
ax['name'].spines['bottom'].set_visible(False)
ax['cbar'].set(yticks=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5],
               yticklabels=[-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
plt.close('all')
fig.savefig('./measures_d_post_cue-post_target.png', dpi=300)
