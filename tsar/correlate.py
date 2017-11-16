"""Python functions to measure dependencies 

We distinguish self time delayed dependencies

"""

import numpy as np
import pandas as pd
import tsar
from tsar.dtypes import is_1darray_like

def autocorrelation(ts, maxlag = 20):
    """Autocorrelation function
    
    The autocorrelation function is a metric of linear self dependence.
    
    The pearson moment of correlation is applied to the time series and
    its multiple lags.
    
    The function returns the Pearson correlation coefficient as a function
    of time lag. Correlation coefficient are stored in list object and the
    lag corresponds to the list index. 
    
    Parameters
    ----------
    ts : 1d array like
        TVPSeries or array-like
    maxlag : int
        Maximum lag to compute autocorrelation

    Returns
    -------
    autocorrelation : list
        List holding autocorrelation values up to maxlag
    
    Examples
    --------
    
    >>> ts = tsar.data.lorenz()['x']
    >>> ac_ts = autocorrelation(ts, maxlag=10)
    >>> print len(ac_ts)
    11
    
    Raises
    ------
    
    
    """

    # test for one-dimensional object

    if not is_1darray_like(ts):

        raise TypeError('Input object should be 1 dimensional numeric array like object.')

    # test for Index error

    if maxlag >= len(ts):

        raise IndexError('Maximum lag {} is greater than series length {}'.format(maxlag, len(ts)))

    lags = range(maxlag+1)

    ts = pd.Series(ts)

    autocorr = [ts.autocorr(i) for i in lags]

    return autocorr

if __name__=='__main__':
    import doctest
    doctest.testmod()
