"""Tests for the correlate.py module

"""

import unittest
import pandas as pd
import numpy as np
from tsar import correlate as corr

class TestAutoCorrelation(unittest.TestCase):
    """Tests for the autocorrelation function"""

    def setUp(self):

        # Valid data

        data = range(10)
        valid_list   = data
        valid_dict   = dict(zip(data, data))
        valid_series = pd.Series(data)
        valid_array = np.array(data)

        # Bad data

        bad_list = [str(i) for i in data] # list is string
        bad_dict = dict(zip(data, [[1,2]]*len(data))) # dict values are list
        bad_df   = pd.DataFrame(data) # dataframe object
        bad_array = np.zeros(shape=(4,3)) # ndarray
        bad_scalar = 1. # scalar

        self.valid_inputs = [
            valid_list,
            valid_dict,
            valid_series,
            valid_array
        ]

        self.bad_inputs = [
            bad_list,
            bad_dict,
            bad_array,
            bad_df,
            bad_scalar
        ]

    def test_valid_input_results(self):
        """Test that results are the same using same data regardless valid input type"""
        results = []
        for good in self.valid_inputs:
            results.append(corr.autocorrelation(good, maxlag=5))
        self.assertTrue(all(x == results[0] for x in results))

    def test_ndtype_error(self):
        """Test if a n-dimensional object raises a TypeError"""
        for bad in self.bad_inputs:
            self.assertRaises(TypeError, corr.autocorrelation, bad)

    def test_index_error(self):
        """Test if a maxlag greater or equal to ts length raises IndexError"""
        s = self.valid_inputs[0]
        self.assertRaises(IndexError, corr.autocorrelation, s, maxlag=15)



if __name__ == '__main__':

    unittest.main()


