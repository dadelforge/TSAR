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
    >>> print ts

    """
    if not isinstance(ts, tsar.TVPSeries):
        ts = tsar.TVPSeries(ts)

    autocorr = [ts.autocorr(i) for i in range(maxlag)]
    return autocorr



if __name__=='__main__':
    import doctest

    doctest.testmod()
