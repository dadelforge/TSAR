"""Algorithm for estimating the mutual information
"""
import tsar
import numpy as np

def _compute_mi_binned(x, y, bins='sqrt', logfunc=np.log):
    """Computes Mutual Information between two data sets using bins
    
    #TODOC
    
    This code is largely inspired from [1]. Sklearn version
    of mutual wasn't used to avoid dependencies.
    
    Parameters
    ----------
    x : array-like
        the first dataset
    y : array-like
        the second dataset
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

    logfunc : function
        logarithm function to use.

    Returns
    -------
    float
        a float number corresponding to the binned 
        mutual information between two ensembles

    Examples
    --------
    >>> x = tsar.datasets.lorenz()['x']
    >>> y = tsar.datasets.lorenz()['y']
    >>> ami = _compute_mi_binned(x, y)
    >>> print ami
    1.35329070861
    
    >>> x = np.random.randint(low=0, high=10, size=1000)
    >>> y = np.random.randint(low=0, high=10, size=1000)
    >>> mi = _compute_mi_binned(x,y)
    >>> print np.allclose(mi, 0, atol=0.1)
    True

    References
    ----------

    [1] Sklearn: Mutual Information between two clusterings
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html

    """

    # check if bins is a strin

    if isinstance(bins, str):
        # Then use automated method from np.histogram

        nx, bins_x = np.histogram(x, bins=bins)
        ny, bins_y = np.histogram(y, bins=bins)

        bins = [bins_x, bins_y]

    contingency_xy, bins_x, bins_y = np.histogram2d(x, y, bins=bins)

    nzx, nzy = np.nonzero(contingency_xy)
    nz_val = contingency_xy[nzx, nzy]

    contingency_sum = contingency_xy.sum()

    pi = np.ravel(contingency_xy.sum(axis=1))
    pj = np.ravel(contingency_xy.sum(axis=0))

    log_contingency_nm = np.log(nz_val)
    contingency_nm = nz_val / contingency_sum

    # Don't need to calculate the full outer product, just for non-zeroes

    outer = pi.take(nzx) * pj.take(nzy)
    log_outer = -np.log(outer) + np.log(pi.sum()) + np.log(pj.sum())

    mi = (contingency_nm * (log_contingency_nm - np.log(contingency_sum)) +
          contingency_nm * log_outer)

    return mi.sum()

if __name__ == '__main__':
    import doctest
    doctest.testmod()