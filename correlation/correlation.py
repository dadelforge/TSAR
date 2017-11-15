"""Python functions to measure dependencies 

We distinguish self time delayed dependencies

"""
import sys
import os
import numpy as np
import pandas as pd

# TODO: this should be removed when packaging
# Relative import of the tvp module
Folder = os.path.dirname(__file__)
ParentFolder = os.path.dirname(Folder)
sys.path.insert(0, '{}/tvp'.format(ParentFolder))
import tvp


def autocorrelation(TVPSeries, maxlag = 20, method='pearson'):
    """Autocorrelation function
    
    Parameters
    ----------
    TVPSeries
    maxlag
    method

    Returns
    -------

    """
    pass



if __name__=='__main__':
    import doctest

    doctest.testmod()
