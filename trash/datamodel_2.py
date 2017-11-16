"""Python data models to handle and constraint various type of time series

Pandas data model may already hold a lot of information about the nature of a time series.

Notes
-----

This module should be revised to be less complex

Available Data models:
----------------------

We basically define two data model for time series:

- Equidistant Time Unaware Numerical value pairs Series/DataFrame (ETUNSeries/ETUNDataFrame)

These are not indexed with timestamps but are assumed to be equidistant. 
Values are constraint to be numeric.

- Equidistant Time Aware Numerical value pairs Series/DataFrame (ETANSeries/ETUNDataFrame)

These are time series indexed with equidistant increasing timestamps. 
Values are constraint to be numeric.


Not implemented Models:
-----------------------

This are not implemented but one may derived multiple data model according to series metadata.
These are non exhaustive examples and they should not be implemented anytime soon.

Using categorical variable:
- Equidistant Time Aware Categorical value pairs Series/DataFrame (ETACSeries/ETUCDataFrame)

When time-zone is known:
- Time-Zoned Equidistant Time Aware Numerical value Series/DataFrame (tzETANSeries/tzETANDataFrame)

When spatial location and representative area is known:
- Equistant Time Aware Numerical value pairs GeoSeries/GeoDataFrame (ETANGeoSeries/ETANGeoDataFrame)

...


"""

import inspect
import logging  # TODO: remove it after dev

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame
from pandas.core.internals import SingleBlockManager

from tsar.errors import NotFixedFrequencyError
from tsar.errors import NotMonotonicIncreasingError

# TODO: remove it after dev
logging.basicConfig(filename='debug.log', level=logging.DEBUG)


# User defined decorators

def _validate_index(index):
    """Raise errors if index is not valid
    
    # TODO: should return bool and raise warning instead
    # This would be more flexible
    
    Parameters
    ----------
    index
        index like object

    Returns
    -------
        bool
    """

    # check for datetime index
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError('Index is not type {}'.format(pd.DatetimeIndex.__name__))

    # check if datetime index is monotonic increasing
    if not index.is_monotonic_increasing:
        raise NotMonotonicIncreasingError('Index in not monotonic increasing')

    # check if datetime index has fixed frequency
    if index.inferred_freq is None:
        raise NotFixedFrequencyError('Index has no fixed frequency')


class TVPBase(NDFrame):
    """Time-value pair base class
    
    TVPSeries and TVPDataFrame are derived from TVPBase.
    
    TVPBase checked for DataTimeIndex consistency before the 
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

    def __new__(cls, datamodel='infer', *args, **kwargs):

        # dictionary of all init parameters
        user_args = cls._buildargdict('__init__', args, kwargs)

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

        # inference of data model
        if datamodel == 'infer':
            datamodel = cls._infer_datamodel(index)

        if datamodel.startswith('ETAN'):
            _validate_index(index)

        return NDFrame.__new__(cls, datamodel=datamodel, *args, **kwargs)

    @classmethod
    def _buildargdict(cls, method, args, kwargs):
        """Merge args and kwargs into a kwarg dictionary
        """

        # get init method arguments
        argspec = inspect.getargspec(eval('cls.{}'.format(method)))

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

    @classmethod
    def _infer_datamodel(cls, index):
        """Infer data model from index and from instance
        """
        datamodel = None

        if issubclass(cls, pd.Series):
            if isinstance(index, pd.DatetimeIndex):
                datamodel = 'ETANSeries'
            else:
                datamodel = 'ETUNSeries'
        if issubclass(cls, pd.DataFrame):
            if isinstance(index, pd.DatetimeIndex):
                datamodel = 'ETANDataFrame'
            else:
                datamodel = 'ETUNDataFrame'

        return datamodel

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

    def __init__(self, data=None, index=None, dtype=None, name=None, copy=False, fastpath=False, datamodel='infer'):

        # new properties

        _metadata = [
            'datamodel',
            'is_equidistant',
            'frequency',
            'is_timeaware',
            'is_numeric'
        ]

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

        super(TVPSeries, self).__init__(data, index, dtype, name, copy, fastpath, datamodel)

        self.datamodel = self._validate_datamodel(datamodel)

    @property
    def _constructor(self):
        return TVPSeries

    @property
    def _constructor_expanddim(self, *args, **kwargs):
        return TVPDataFrame

    @property
    def is_equidistant(self):
        return self.index.infer_freq is not None

    @property
    def frequency(self):
        return self.index.infer_freq()

    @property
    def is_timeaware(self):
        return isinstance(self.index, pd.DataTimeIndex)

    @property
    def is_numeric(self):
        values = self.values
        return np.issubdtype(values.dtype, np.number)

    def _validate_datamodel(self, datamodel):
        """
        
        Parameters
        ----------
        datamodel
            dict a data model dictionary
            str 'infer'

        Returns
        -------
            dict a data model dictionary

        """

        default_model = {
            'is_equidistant': True,
            'is_timeaware': False,
            'is_numeric': True
        }

        if datamodel == 'infer':

            # Build new model

            newmodel = {
                'is_equidistant': self.is_equidistant,
                'is_timeaware': self.is_timeaware,
                'is_numeric': self.is_numeric
            }

        elif isinstance(datamodel, dict):

            # check if all required keys are in the datamodel

            if all(key in datamodel for key in default_model.keys()):

                # validate equidistant

                if self.is_equidistant != datamodel['is_equidistant']:
                    raise ValueError

            else:
                missing_keys = [k not in default_model.keys for k in datamodel.keys()]
                raise KeyError('datamodel default keys are missing')


        else:
            raise TypeError('datamodel should be \'infer\' or type dict, not type {}'.format(type(datamodel)))



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

    def _set_axis(self, axis, labels, *args, **kwargs):
        _validate_dateindex(labels)
        super(TVPSeries, self)._set_axis(axis, labels, *args, **kwargs)


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
    
    Adding columns:
    
    >>> tdf['new'] = ['hi']*len(tdf)
    Traceback (most recent call last):
        ...
    TypeError: Values are not numeric
    
    Raises
    ------
    TypeError
        If values are not numeric
    
    """

    def __init__(self, data=None, index=None, columns=None, dtype=None, copy=False, datamodel='infer'):

        # checking if data is SingleBlockManager
        if isinstance(data, SingleBlockManager):
            data = data.get_values()

        data = np.array(data)

        if not np.issubdtype(data.dtype, np.number):
            raise TypeError('Values are not numeric')

        self.datamodel = datamodel

        super(TVPDataFrame, self).__init__(data, index, columns, dtype, copy, datamodel)

    @property
    def _constructor(self, data=None, index=None):
        return TVPDataFrame

    @property
    def _constructor_sliced(self):
        return TVPSeries

    def _set_axis(self, axis, labels, *args, **kwargs):
        """ Check index before any axis setting
        """
        if self.datamodel:
            _validate_datetimeindex(labels)
        super(TVPDataFrame, self)._set_axis(axis, labels)

    def _sanitize_column(self, key, value, broadcast=True):
        """ Check column values before assignment
        """
        value = np.array(value)
        if not np.issubdtype(value.dtype, np.number):
            raise TypeError('Values are not numeric')
        return super(TVPDataFrame, self)._sanitize_column(key, value, broadcast=True)


if __name__ == '__main__':
    import doctest

    doctest.testmod()
