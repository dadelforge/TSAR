""" Tests for the time series and time dataframe model objects
The times series class is model to hold continuous time data 
Current implemented test:
# TODO: initiate TVPSeries with valid 1d array should pass
# TODO: initiate TVPSeries with nd array should not pass
# TODO: initiate TVPSeries with valid list should pass
# TODO: initiate TVPSeries with valid scalar should pass
# TODO: initiate TVPSeries with valid series should pass
"""
# To be implemented:

# TODO: TVPSeries.to_frame() should return a valid TVPDataFrame
# TODO: TVPDataFrame['col'] should return a valid TVPSeries
# TODO: initiate TVPSeries with valid dictionary should pass



import os
import sys
import unittest

# Relative import of the tvp module TODO remove when packaging
Folder = os.path.dirname(__file__)
ParentFolder = os.path.dirname(Folder)
sys.path.insert(0, '{}/tvp'.format(ParentFolder))
from tvp.base import *


class TestTVPSeriesInstantiation(unittest.TestCase):
    """Test instantiation of TVPSeries using different data types
    """

    def setUp(self):

        # Define valid index and data

        valid_index = pd.date_range(start='1988-06-15', periods=10)
        valid_data = range(10)

        # Define bad indexes and data
        notmonotonicincreasing_index = valid_index[::-1]
        sparse_index = valid_index.drop(pd.to_datetime('1988-06-18'))
        num_index = range(10)
        str_data = ['hi'] * 10

        # Define valid and bad series

        self.valid_series = pd.Series(data=valid_data, index=valid_index)
        self.notmonotonicincreasing_series = pd.Series(data=valid_data, index=notmonotonicincreasing_index)
        self.sparse_series = pd.Series(data=valid_data[:-1], index=sparse_index)
        self.num_index_series = pd.Series(data=valid_data, index=num_index)
        self.bad_values_series = pd.Series(data=str_data, index=valid_index)

        # Define valid and bad input data

        self.dtypes = ['series', 'nparr', 'list', 'scalar']

    @staticmethod
    def _series_to_other(series, other):

        if other != 'series':
            index = series.index
        else:
            index = None

        if other == 'nparr':
            data = series.values
        elif other == 'list':
            data = series.values.tolist()
        elif other == 'scalar':
            data = 1.
        else:
            data = series

        return index, data

    def test_type_valid(self):
        """Test if TVPSeries is created from a valid pd.Series
        """
        for dtype in self.dtypes:
            ix, data = self._series_to_other(self.valid_series, dtype)
            ts = TVPSeries(data=data, index=ix)
            self.assertTrue(isinstance(ts, TVPSeries))

    def test_indextype_valid_series(self):
        """Test if TVPSeries has a pd.DateTimeIndex
        """
        for dtype in self.dtypes:
            ix, data = self._series_to_other(self.valid_series, dtype)
            ts = TVPSeries(data=data, index=ix)
            self.assertTrue(isinstance(ts.index, pd.DatetimeIndex))

    def test_raise_notmonotonicincreasing_index(self):
        """Test if error is raised when not monotonic increasing index is passed
        """
        for dtype in self.dtypes:
            ix, data = self._series_to_other(self.notmonotonicincreasing_series, dtype)
            self.assertRaises(NotMonotonicIncreasingError, TVPSeries, data, ix)

    def test_raise_typeerror_index(self):
        """Test if passing a not DateTimeIndex raises TypeError
        """
        for dtype in self.dtypes:
            ix, data = self._series_to_other(self.num_index_series, dtype)
            self.assertRaises(TypeError, TVPSeries, data, ix)

    def test_raise_notfixefrequency_index(self):
        """Test if passing a sparse datetime index raises NotFixedFrequencyError
        """
        for dtype in self.dtypes:
            ix, data = self._series_to_other(self.sparse_series, dtype)
            self.assertRaises(NotFixedFrequencyError, TVPSeries, data, ix)

    def test_raise_notnumerictype_value(self):
        self.assertRaises(TypeError, TVPSeries, self.bad_values_series)

class TestTVPSeriesMethods(unittest.TestCase):

    def setUp(self):
        pass

    def test_to_frame(self):
        pass


if __name__ == '__main__':
    unittest.main()
