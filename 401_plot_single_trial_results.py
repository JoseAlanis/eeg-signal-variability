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

import seaborn as sns
from scipy.stats import zscore
from statsmodels.stats.multitest import multipletests

from mne import read_epochs
from mne.utils import logger
from mne.viz import plot_topomap

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# all parameters are defined in config.py
from config import (
    FPATH_DERIVATIVES
)

# %%
# default settings (use subject 1, don't overwrite output files)
overwrite = False
subject = 1
session = 1
task = 'numberletter'
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
which_measures = {
    "activity": 0,
    "mobility": 1,
    "complexity": 2,
    "permutation_entropy": 3,
    "weighted_permutation_entropy": 4,
    "multiscale_entropy": 5,
    "ms_1": 6,
    "ms_2": 7,
    "ms_3": 8,
    "ms_4": 9,
    "spectral_entropy": 10,
    "1f_offset": 11,
    "1f_exponent": 12
}

# %%
# import the data

# create path for model fits
FPATH_FITS_ODDEVEN = os.path.join(FPATH_DERIVATIVES,
                                  'analysis_dataframes',
                                  '%s_sensor_*_*_fits_st.rds' % task)
FPATH_FITS_ODDEVEN = glob.glob(FPATH_FITS_ODDEVEN)

# object shape
n_measures = len(which_measures)
n_sensors = 32

# load results for each sensor
fits = np.empty((n_sensors, n_measures, 3))
p_vals = np.ones((n_sensors, n_measures, 3))

for fpath in FPATH_FITS_ODDEVEN:
    fit_meas = pyreadr.read_r(fpath)[None]
    m = which_measures[fit_meas.measure.unique()[0]]
    sj = fit_meas.sensor.unique()[0] - 1
    fits[int(sj), m, :] = fit_meas.Omega2_partial
    p_vals[int(sj), m, :] = fit_meas.p
    # sigs = fit_meas.p < (0.05 / (13*32))
    # p_vals[int(sj), m, sigs] = sigs[sigs]
    # p_vals[int(sj), m, ...] = pyreadr.read_r(fpath)[None].o_sq_CI_low > 0.06

# correct p-values
p_vals = (multipletests(p_vals.flatten(), method='bonferroni')[1] < 0.05).reshape((n_sensors, n_measures, 3))

# %%
# plot omnibus test results

# measures to plot
measure_labels = {
    "activity": 'A',
    "mobility": 'M',
    "complexity": 'C',
    "permutation_entropy": 'PE',
    "weighted_permutation_entropy": 'WPE',
    "multiscale_entropy": 'MSE',
    "ms_1": 'MSE$_{1}$',
    "ms_2": 'MSE$_{2}$',
    "ms_3": 'MSE$_{3}$',
    "ms_4": 'MSE$_{4}$',
    "spectral_entropy": 'SE',
    "1f_offset":  r'1/$f$ (off.)',
    "1f_exponent":  r'1/$f$ (exp.)'
}
cmap = mpl.cm.bone_r
# bounds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
bounds = np.arange(0.05, 1.0, 0.05)
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

widths = [1.0, 1.0, 1.0, 0.6]
heights = [0.25] + [1.0 for _ in which_measures] + [0.25]
gs_kw = dict(width_ratios=widths,
             height_ratios=heights,
             wspace=0.15,
             hspace=0.5)

mnames_first = ['name1'] + [t + '_1' for t in which_measures] + ['cbar1']
manmes_second = ['name2'] + [t + '_2' for t in which_measures] + ['X']
mnames_third = ['name3'] + [t + '_3' for t in which_measures] + ['X']
mnames_forth = ['name4'] + [t + '_4' for t in which_measures] + ['leg']
fig, ax = plt.subplot_mosaic(
    mosaic=np.transpose(
        np.array([mnames_first, manmes_second, mnames_third, mnames_forth])),
    gridspec_kw=gs_kw,
    empty_sentinel="X",
    figsize=(16, 26)
)

for name in ['name1', 'name2', 'name3', 'name4', 'leg']:
    ax[name].spines['right'].set_visible(False)
    ax[name].spines['top'].set_visible(False)
    ax[name].spines['left'].set_visible(False)
    ax[name].spines['bottom'].set_visible(False)
    ax[name].set(yticks=[], yticklabels=[], xticks=[], xticklabels=[])

for measure in which_measures:
    plot_topomap(fits[:, which_measures[measure], 0],
                 epochs.info,
                 cmap=cmap,
                 cnorm=norm,
                 # mask=p_vals[:, which_measures[measure], 0],
                 # mask_params=dict(marker='o',
                 #                  markerfacecolor='w',
                 #                  markeredgecolor='w',
                 #                  linewidth=0, markersize=1),
                 contours=0,
                 size=10,
                 axes=ax[measure + '_1'],
                 sensors=False,
                 show=False)
    ax[measure + '_1'].set_title(r'%s' % measure_labels[measure])
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
    cax=ax['cbar1'],
    extend='both',
    pad=0.25,
    ticks=bounds,
    spacing='uniform',
    orientation='horizontal',
    label=r'Effect size ($\Omega^2$)',
)
ax['cbar1'].set_xticks(ticks=np.arange(0.1, 0.95, 0.1),
                       labels=np.round(np.arange(0.1, 0.95, 0.1), 1),
                       rotation=-45)
ax['name1'].annotate('Main Effect of Time Window:\n'
                     r'(Baseline vs Cue vs Target)',
                     (0.5, 0.5), fontsize=14, color='k', ha='center')

for measure in which_measures:
    plot_topomap(fits[:, which_measures[measure], 1],
                 epochs.info,
                 cmap=cmap,
                 cnorm=norm,
                 # mask=p_vals[:, which_measures[measure], 1],
                 # mask_params=dict(marker='o',
                 #                  markerfacecolor='w',
                 #                  markeredgecolor='w',
                 #                  linewidth=0, markersize=1),
                 contours=0,
                 size=10,
                 axes=ax[measure + '_2'],
                 sensors=False,
                 show=False)
    ax[measure + '_2'].set_title(r'%s' % measure_labels[measure])
ax['name2'].annotate('Main Effect of Condition:\n'
                     r'(Repeat vs Switch)',
                     (0.5, 0.5), fontsize=14, color='k', ha='center')

cmap_int = mpl.cm.bone_r
for measure in which_measures:
    plot_topomap(fits[:, which_measures[measure], 2],
                 epochs.info,
                 cmap=cmap_int,
                 cnorm=norm,
                 # mask=p_vals[:, which_measures[measure], 2],
                 # mask_params=dict(marker='o',
                 #                  markerfacecolor='w',
                 #                  markeredgecolor='w',
                 #                  linewidth=0, markersize=1),
                 contours=0,
                 size=10,
                 axes=ax[measure + '_3'],
                 sensors=False,
                 show=False)
    ax[measure + '_3'].set_title(r'%s' % measure_labels[measure])
ax['name3'].annotate('Interaction Effect:\n'
                     r'Time Window x Condition',
                     (0.5, 0.5), fontsize=14, color='k', ha='center')

lcmap = mpl.cm.magma
kwargs = {'markersize': 3, 'linewidth': 2}
for measure in list(which_measures.keys()):
    fpath_meas = os.path.join(FPATH_DERIVATIVES,
                              'analysis_dataframes',
                              '%s_single_trial.rds' % measure)
    fit_meas = pyreadr.read_r(fpath_meas)[None]
    fit_meas = fit_meas[fit_meas.task.str.lower().str.replace("/", "") == task]
    fit_meas = fit_meas.dropna()
    fit_meas[measure] = zscore(fit_meas[measure])
    fit_meas['tw'] = fit_meas['tw'].astype('category')
    fit_meas['tw'] = fit_meas['tw'].cat.reorder_categories(
        ['Pre Cue', 'Post Cue', 'Post Target'])

    sns.pointplot(data=fit_meas,
                  x="tw", y=measure,
                  hue="condition", dodge=True,
                  errorbar=('ci', 99),
                  ax=ax[measure + '_4'],
                  legend=False,
                  palette=[lcmap(0.05), lcmap(0.6)],
                  **kwargs)
    ax[measure + '_4'].set_ylim(-0.3, 0.3)
    ax[measure + '_4'].set_ylabel(ylabel=r'$z$-Score'
                                         '\n'
                                         r'%s' % measure_labels[measure])
    ax[measure + '_4'].set_xticks([0, 1, 2])
    ax[measure + '_4'].set_xticklabels(labels=['Baseline', 'Cue', 'Target'],
                                       rotation=-20)

custom_lines = [Line2D([0], [0], color=lcmap(0.05), lw=5),
                Line2D([0], [0], color=lcmap(0.6), lw=5)]
ax['leg'].legend(custom_lines, ['Repeat', 'Switch'],
                 fontsize='medium',
                 loc='upper center',
                 frameon=False,
                 ncols=2)

ax['name4'].annotate('Change:\n'
                     r'(scalp-wide average)',
                     (0.5, 0.5), fontsize=14, color='k', ha='center')

plt.close('all')
fig.savefig('./%s_main_effects_measures_o_sq.png' % task,
            dpi=300)













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
cmap_d = mpl.cm.RdBu_r
bounds = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25,  0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N, extend='both')

# widths = [2.0] + [1.0 for _ in measures] + [0.3]
# heights = [0.25, 1.0, 0.25]
# gs_kw = dict(width_ratios=widths,
#              height_ratios=heights,
#              wspace=0.5)
#
# mnames_top = ['X'] + [str(n) for n, t in enumerate(measures)] + ['X']
# manmes_center = ['name'] + [str(n) for n, t in enumerate(measures)] + ['cbar']
# mnames_bottom = ['X'] + [str(n) for n, t in enumerate(measures)] + ['X']
# fig, ax = plt.subplot_mosaic(mosaic=[mnames_top,
#                                      manmes_center,
#                                      mnames_bottom],
#                              gridspec_kw=gs_kw,
#                              empty_sentinel="X",
#                              figsize=(25, 2.5))
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
                 axes=ax[which_measures[n_meas] + '_2'],
                 sensors=False,
                 show=False)
    ax[which_measures[n_meas] + '_2'].set_title(r'%s' % measure_labels[int(n_meas)])
fig.colorbar(
    mpl.cm.ScalarMappable(cmap=cmap_d, norm=norm),
    cax=ax['cbar2'],
    extend='both',
    # extendfrac='auto',
    ticks=bounds,
    spacing='uniform',
    orientation='horizontal',
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
