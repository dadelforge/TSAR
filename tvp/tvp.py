"""Python model to handle univariate and mutivariate time series object with pandas

The purposes of the time series library is to ensure that time series index is consistent.
And that the variable is numeric

"""

# TODO: implement forgiveness for empty series and df
# TODO: implement is numeric

import inspect
import pandas as pd
from pandas.core.generic import NDFrame

# import user defined exception
from tvpexept import *


class TVPBase(NDFrame):
    """Time-value pair base class
    
    TVPSeries and TVPDataFrame are derived from TVPBase.
    """

    def __new__(cls, *args, **kwargs):

        # dictionary of all init parameters
        user_args = cls._buildargdict(args, kwargs)

        # retrieve index
        index = user_args['index']

        # check for datetime index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError('Index is not type {}'.format(pd.DatetimeIndex.__name__))

        # check if datetime index is monotonic increasing
        if not index.is_monotonic_increasing:
            raise NotMonotonicIncreasingIndex

        # check if datetime index has fixed frequency
        if index.inferred_freq is None:
            raise NotFixedFrequencyIndex

        return NDFrame.__new__(cls, *args, **kwargs)

    @classmethod
    def _buildargdict(cls, args, kwargs):
        """Merge args and kwargs into a kwarg dictionary
        """

        # get init method arguments
        argspec = inspect.getargspec(cls.__init__)

        # arguments names
        argnames = argspec[0][1:]

        # arguments defaults values
        argdefaults = argspec[-1]

        # dictionary of arguments names and default values
        default_dict = dict(zip(argnames, argdefaults))

        # create copy to store user values
        user_dict = default_dict.copy()

        # store user argument
        for i, arg in enumerate(args):
            user_dict[argnames[i]] = arg

        # store user keyword argument
        for key, val in kwargs.iteritems():
            user_dict[key] = val

        return user_dict


class TVPSeries(TVPBase, pd.Series):
    """Time-value paired pandas Series
    
    This class constraints pandas series to have a monotonic 
    increasing pd.DatetimeIndex with fixed frequency.
    
    """

    @property
    def _constructor(self):
        return TVPSeries

    @property
    def _constructor_expanddim(self):
        return TVPDataFrame


class TVPDataFrame(TVPBase, pd.DataFrame):
    """Time-values paired pandas DataFrame
    
    This class constraints pandas DataFrame to have a monotonic 
    increasing pd.DatetimeIndex with fixed frequency.
    
    """

    @property
    def _constructor(self):
        return TVPDataFrame

    @property
    def _constructor_sliced(self):
        return TVPSeries
