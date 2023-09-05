"""
Statistical Utility Functions for Data Analysis

This script contains utility functions focused on statistical
analysis. It is particularly useful for calculating confidence
intervals for a given dataset.

Functions:
-----------
- `mean_confidence_interval`: Calculates the mean and corresponding
  confidence interval of a given dataset, based on a specified
  confidence level. The function returns a dictionary containing
  the mean, lower bound, and upper bound.

Dependencies:
-------------
The script relies on the NumPy and SciPy libraries to handle
numerical operations and statistical calculations.

Examples:
---------
To see examples of how the `mean_confidence_interval` function
can be used, please refer to its individual docstring.

"""

import numpy as np

from scipy.stats import t, sem


def mean_confidence_interval(data, confidence=0.95, axis=0):
    """
     Calculate the mean and confidence interval for a dataset.

     Parameters:
     -----------
     data : array_like
         Input data. The data from which to calculate the mean and
         confidence interval. Can be a list, tuple, or a NumPy array.

     confidence : float, optional
         The confidence level to calculate the interval for.
         The default value is 0.95, indicating a 95% confidence interval.

     axis : int, optional
         Axis along which the means are computed. The default is 0.

     Returns:
     --------
     tuple
         A tuple containing three floats:
         - mean: The mean of the data along the specified axis.
         - lower_bound: The lower bound of the confidence interval.
         - upper_bound: The upper bound of the confidence interval.

     Examples:
     ---------
     >>> mean_confidence_interval([1, 2, 3, 4, 5])
    {'mean': 3.0, 'lower_bound': 1.036756838522439, 'upper_bound': 4.9632431614775605}  # noqa: E501

     Notes:
     ------
     The function uses Student's t-distribution to calculate the
     confidence interval, which is more accurate for small sample sizes
     than a normal distribution.

     """

    a = 1.0 * np.array(data)
    n = a.shape[0]
    m, se = np.mean(a, axis=axis), sem(a, axis=axis)
    h = se * t.ppf((1 + confidence) / 2.0, n-1)

    return {'mean': m, 'lower_bound': m-h, 'upper_bound': m+h}
