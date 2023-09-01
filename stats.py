import numpy as np

import scipy.stats


def mean_confidence_interval(data, confidence=0.95, axis=0):
    a = 1.0 * np.array(data)
    n = a.shape[0]
    m, se = np.mean(a, axis=axis), scipy.stats.sem(a, axis=axis)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n-1)
    return m, m-h, m+h