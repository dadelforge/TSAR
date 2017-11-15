"""Python model to handle univariate and mutivariate time series object with pandas

The purposes of the time series library is to ensure that time series index is consistent.

"""

# TODO: modifying index should throw an error
# HOWTO: use a decorator on pandas.Series._set_axis to test label?
# TODO: adding a non numeric column to TVPDataframe should throw an error
# TODO:

import inspect
import logging  # TODO: remove it after dev

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
from pandas.core.internals import SingleBlockManager

# TODO: remove it after dev
logging.basicConfig(filename='debug.log', level=logging.DEBUG)


# User defined exceptions
class TVPError(Exception):
    """Base class for tvp errors"""
    pass


class NotMonotonicIncreasingError(TVPError):
    """Raised when a datetime index is not monotonic increasing"""
    pass


class NotFixedFrequencyError(TVPError):
    """Raised when a datetime index has no fixed frequency"""
    pass


class TVPBase(NDFrame):
    """Time-value pair base class
    
    TVPSeries and TVPDataFrame are derived from TVPBase.
    
    TVPBase checked for DataTimeIndex consitency before the 
    object is instantiated.
    
    
    Raises
    ------
    TypeError
        If index is not type `pandas.DatetimeIndex`
    
    NotMonotonicIncreasingError
        If index is not monotonic increasing
        
    NotFixedFrequencyError
        If index has no fixed frequency 
        
    """

    def __new__(cls, *args, **kwargs):

        # dictionary of all init parameters
        user_args = cls._buildargdict(args, kwargs)

        # retrieve index
        index = user_args['index']

        # retrieve data
        data = user_args['data']

        # Do not support dictionary
        if isinstance(data, dict):
            raise NotImplementedError('dict type not supported')


        # retrieve index from data if data is a NDFrame
        if index is None and isinstance(data, NDFrame):
            index = data.index

        # check for datetime index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError('Index is not type {}'.format(pd.DatetimeIndex.__name__))

        # check if datetime index is monotonic increasing
        if not index.is_monotonic_increasing:
            raise NotMonotonicIncreasingError('Index in not monotonic increasing')

        # check if datetime index has fixed frequency
        if index.inferred_freq is None:
            raise NotFixedFrequencyError('Index has no fixed frequency')

        return NDFrame.__new__(cls)

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

    @property
    def _constructor_expanddim(self):
        return TVPBase


class TVPSeries(TVPBase, pd.Series):
    """Time-value paired pandas Series
    
    This class constraints pandas series to have a monotonic 
    increasing pd.DatetimeIndex with fixed frequency.
    
    TVPSeries are instantiated using pandas.Series arguments.
        
    Examples
    --------
    
    A valid example:
    
    >>> date_range = pd.date_range(start='2010-06-15', periods=10)
    >>> values = range(10)
    >>> ts = TVPSeries(data=values, index=date_range)
    
    Using a integer type index:
    
    >>> int_range = range(10)
    >>> ts = TVPSeries(data=values, index=int_range)
    Traceback (most recent call last):
        ...
    TypeError: Index is not type DatetimeIndex
    
    Using a non monotonic increasing DatetimeIndex:
    
    >>> date_range = pd.date_range(start='2010-06-15', periods=10)
    >>> inv_date_range = date_range[::-1]
    >>> ts = TVPSeries(data=values, index=inv_date_range)
    Traceback (most recent call last):
        ...
    NotMonotonicIncreasingError: Index in not monotonic increasing
    
    Using a non fixed frequency DatetimeIndex:
    
    >>> date_range = pd.date_range(start='2010-06-15', periods=10)
    >>> broken_date_range = date_range.drop(pd.to_datetime('2010-06-18'))
    >>> values = range(9)
    >>> ts = TVPSeries(data=values, index=broken_date_range)
    Traceback (most recent call last):
        ...
    NotFixedFrequencyError: Index has no fixed frequency
    
    Expanding dimension:
    
    >>> tdf = ts.to_frame()
    >>> print type(tdf)
    <class '__main__.TVPDataFrame'>
    
    Raises
    ------
    TypeError
        If values are not numeric
         
    """

    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False):

        # checking if data is a series

        if isinstance(data, pd.Series) and index is None:
            index = data.index
            data = data.values

        # checking if data is SingleBlockManager

        if isinstance(data, SingleBlockManager):
            data = data.get_values()

        data = np.array(data)

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError('Values are not numeric')
        super(TVPSeries, self).__init__(data, index, dtype, name, copy, fastpath)

    @property
    def _constructor(self):
        return TVPSeries

    @property
    def _constructor_expanddim(self, *args, **kwargs):
        return TVPDataFrame

    def to_frame(self, *args, **kwargs):
        """
        Convert TVPSeries to TVPDataFrame

        Returns
        -------
        tdf : TVPDataFrame
        """
        kwargs['index'] = self.index
        kwargs['data'] = self.values

        tdf = self._constructor_expanddim(*args, **kwargs)

        return tdf


class TVPDataFrame(TVPBase, pd.DataFrame):
    """Time-values paired pandas DataFrame
    
    This class constraints pandas DataFrame to have a monotonic 
    increasing pd.DatetimeIndex with fixed frequency.
    
    TVPDataFrame are instantiated using pandas.DataFrame arguments.
    
    When sliced, a TVPDataFrame return a TVPSeries.
    
    Examples:
    ---------
    
    A valid example:
    
    >>> date_range = pd.date_range(start='2010-05-05', periods=5)
    >>> values = np.zeros(shape=(5,2))
    >>> cols = ['col1', 'col2']
    >>> tdf = TVPDataFrame(data = values, index=date_range, columns=cols)
    >>> print tdf
                col1  col2
    2010-05-05   0.0   0.0
    2010-05-06   0.0   0.0
    2010-05-07   0.0   0.0
    2010-05-08   0.0   0.0
    2010-05-09   0.0   0.0
    
    Slicing example:
    
    >>> print type(tdf['col1'])
    <class '__main__.TVPSeries'>
    
    Raises
    ------
    TypeError
        If values are not numeric
    
    """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False):

        # checking if data is SingleBlockManager
        if isinstance(data, SingleBlockManager):
            data = data.get_values()

        data = np.array(data)

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError('Values are not numeric')
        super(TVPDataFrame, self).__init__(data, index, columns, dtype, copy)

    @property
    def _constructor(self, data=None, index=None):
        return TVPDataFrame

    @property
    def _constructor_sliced(self):
        return TVPSeries


if __name__ == '__main__':
    import doctest

    doctest.testmod()
