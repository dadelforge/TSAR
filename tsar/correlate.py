"""Python functions to measure dependencies 

#TODOC

"""

# TODO: Add top level documentation

import numpy as np
import pandas as pd
from sklearn.metrics import mutual_info_score

import tsar
from tsar.dtypes import is_1darray_like
from tsar.algorithms.mutualinformation import _compute_mi_binned


# ----------------------------------------------------------------------
# Self dependency functions


def autocorrelation(ts, maxlag=20):
    """Auto Correlation Function
    
    The auto correlation function is a metric of linear self dependence.
    
    The pearson moment of correlation is applied to the time series and
    its multiple lags.
    
    The function returns the Pearson correlation coefficient as a function
    of time lag. Correlation coefficient are stored in list object and the
    lag corresponds to the list index. 
    
    Parameters
    ----------
    ts : 1d array_like
        Array like holding the time series values. Valid types are list of 
        number, dict of number values, numpy 1d array or pandas Series.
    maxlag : int
        Maximum lag to compute autocorrelation.

    Returns
    -------
    autocorrelation : list
        List holding autocorrelation values up to maxlag
    
    Examples
    --------
    
    >>> ts = tsar.data.lorenz()['x']
    >>> rho = autocorrelation(ts, maxlag=2)
    >>> print rho
    [0.99999999999999989, 0.9985116968252471, 0.9940665907167473]
    
    Raises
    ------
    TypeError
        Raised if input is not one dimensional numeric.
    IndexError
        Raised if maxlag greater than time series length.
    
    """

    # TODO: prefer not using pandas to lower dependencies
    # TODO: draft doc to be imporved

    # test for one-dimensional object

    if not is_1darray_like(ts):
        raise TypeError('Input object should be 1 dimensional numeric array like object.')

    # test for Index error

    if maxlag >= len(ts):
        raise IndexError('Maximum lag {} is greater than series length {}'.format(maxlag, len(ts)))

    # list of lags

    lags = range(maxlag + 1)

    # conversion of ts into a pd.Series

    ts = pd.Series(ts)

    autocorr = [ts.autocorr(i) for i in lags]

    return autocorr


def automutualinfo(ts, maxlag=20, bins='sqrt', logfunc=np.log, method='binned'):
    """Auto Mutual Information
    
    Compute the discrete mutual information between time series and its 
    successive lags. The discrete mutual information is given by:
    
    .. math::

        MI(X,Y)=\sum_{x \in X}^R \sum_{y \in Y} P(x,y)\log\\frac{P(x,y)}{P(x)P'(y)}
        
        
    
    Parameters
    ----------
    ts : 1d array_like
        Array like holding the time series values. Valid types are list of 
        number, dict of number values, numpy 1d array or pandas Series.
    maxlag : int
        Maximum lag to compute auto mutual information.
    bins : int or sequence of scalars or str, optional
        If `bins` is an int, it defines the number of equal-width
        bins in the given range. If `bins` is a sequence, it defines 
        the bin edges, including the rightmost edge, allowing for 
        non-uniform bin widths.
        .. versionadded:: numpy 1.11.0
        If `bins` is a string from the list below, `histogram` will use
        the method chosen to calculate the optimal bin width and
        consequently the number of bins (see [3] for more detail on
        the estimators) from the data that falls within the requested
        range. While the bin width will be optimal for the actual data
        in the range, the number of bins will be computed to fill the
        entire range, including the empty portions. For visualisation,
        using the 'auto' option is suggested. Weighted data is not
        supported for automated bin size selection.
        'auto'
            Maximum of the 'sturges' and 'fd' estimators. Provides good
            all around performance.
        'fd' (Freedman Diaconis Estimator)
            Robust (resilient to outliers) estimator that takes into
            account data variability and data size.
        'doane'
            An improved version of Sturges' estimator that works better
            with non-normal datasets.
        'scott'
            Less robust estimator that that takes into account data
            variability and data size.
        'rice'
            Estimator does not take variability into account, only data
            size. Commonly overestimates number of bins required.
        'sturges'
            R's default method, only accounts for data size. Only
            optimal for gaussian data and underestimates number of bins
            for large non-gaussian datasets.
        'sqrt'
            Square root (of data size) estimator, used by Excel and
            other programs for its speed and simplicity. This is 
            default.

    Returns
    -------
    automi : list
        List holding auto mutual information values up to maxlag.
        
    Examples
    --------
    
    >>> ts = tsar.data.lorenz()['x'].iloc[:100]
    >>> ami = automutualinfo(ts, maxlag=2, bins='sqrt')
    >>> print ami
    [2.1316557270483645, 1.7303407746505963, 1.5756941465276517]

    References
    ----------
    
    [1] Kraskov, Alexander, Harald Stoegbauer, and Peter Grassberger. 
        "Estimating Mutual Information." Physical Review E 69, no. 6 (June 23, 2004). 
        https://doi.org/10.1103/PhysRevE.69.066138.

    [1] Sklearn: Mutual Information between two clusterings
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html
    [2] numpy.histogram2d
        https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram2d.html
    [3] numpy.histogram
        https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.histogram.html
    
    """
    #TODO : draft documentation to be improved

    # check for one-dimensional object

    if not is_1darray_like(ts):
        raise TypeError('Input object should be 1 dimensional numeric array like object.')

    # check for Index error

    if maxlag >= len(ts):
        raise IndexError('Maximum lag {} is greater than series length {}'.format(maxlag, len(ts)))

    # check for method

    if method != 'binned':
        raise NotImplementedError(
            'Not implemented method {}. Only binned method is currently supported.'.format(method))

    # conversion of ts into a pd.Series

    ts = pd.Series(ts)

    # list of lags

    lags = range(maxlag + 1)

    # use numpy histogram to get bins

    #bin_edges = np.histogram(ts.values, bins=bins)[1]

    # define auto mutual information list
    automi = []

    for t in lags:

        ts_lag = ts.shift(t)

        x = ts[t:]
        y = ts_lag[t:]

        mi = _compute_mi_binned(x, y, bins=bins, logfunc=logfunc)

        automi.append(mi)

    return automi

def crosscorrelation():
    pass

def crossmutualinfo():
    pass

if __name__ == '__main__':
    import doctest

    doctest.testmod()
