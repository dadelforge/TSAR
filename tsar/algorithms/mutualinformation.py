"""Algorithm for estimating the mutual information

"""


def _compute_mi_binned(x, y, bins='sqrt', logfunc=np.log):
    """
    
    This code is largely inspired from [1]. Sklearn version
    of mutual wasn't used to avoid dependencies.
    
    Parameters
    ----------
    x
    y
    bins
    logfunc

    Returns
    -------
    float
        a float number corresponding to the binned 
        mutual information between two ensembles

    Examples
    --------
    >>> ts = tsar.data.lorenz()['x'].iloc[:100]
    >>> ami = _compute_mi_binned(ts[1:], ts.shift(1)[1:])
    >>> print ami

    References
    ----------

    [1] Sklearn: Mutual Information between two clusterings
        http://scikit-learn.org/stable/modules/generated/sklearn.metrics.mutual_info_score.html

    """

    X = x.values  # Ensemble of x
    Y = y.values  # Ensemble of y

    Nx = float(len(x))  # Size of ensemble X
    Ny = float(len(y))  # Size of ensemble Y

    assert (all(~np.isnan(x)))
    assert (all(~np.isnan(y)))

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