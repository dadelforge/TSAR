"""Python functions to measure dependencies 

We distinguish self time delayed dependencies

"""

import numpy as np
import pandas as pd
import tsar

def autocorrelation(ts, maxlag = 20):
    """Autocorrelation function
    
    Parameters
    ----------
    ts
        TVPSeries or array-like
    maxlag
        int

    Returns
    -------
    list
    
    Examples
    --------
    
    >>> ts = tsar.data.lorenz()['x']
    >>> ac_ts = autocorrelation(ts, maxlag=10)
    >>> print len(ac_ts)
    10
    
    """

    autocorr = [ts.autocorr(i) for i in range(maxlag)]
    return autocorr

if __name__=='__main__':
    import doctest
    doctest.testmod()
