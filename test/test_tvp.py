""" Tests for the time series and time dataframe model objects
The times series class is model to hold continuous time data 
Current test

To be implemented

"""
import os
import random
import sys
import unittest
import numpy as np
import pandas as pd

# Relative import of the tvp module
Folder = os.path.dirname(__file__)
ParentFolder = os.path.dirname(Folder)
sys.path.insert(0, '{}/tvp'.format(ParentFolder))
import test_tvp


class TestParentMethods(unittest.TestCase):
    def __init__(self, parent_object, object_method):
        super(TestParentMethods, self).__init__()
        self.parent_object = parent_object
        self.object_method = object_method

    def runTest(self):
        self.assertTrue(hasattr(self.parent_object, self.object_method),
                        '{} object has no method {}'.format(self.parent_object.__name__, self.object_method))


class TestRandomValidTS(unittest.TestCase):
    def setUp(self):
        """#TODOC"""
        # Generate valid data
        self.valid_data = self.generate_valid_ts()
        self.valid_series = test_tvp.TVPSeries(self.valid_data)

    def test_instance(self):
        self.assertIsInstance(self.valid_series, 'TVPSeries')

    @staticmethod
    def generate_valid_ts():
        """Generate a valid random tvp
        It should pass the test
        
        :return: pd.Series
        """
        start_year = str(random.choice(range(2000, 2010)))
        start_month = str(random.choice(range(1, 13))).zfill(2)
        start_day = str(random.choice(range(1, 29))).zfill(2)
        date_string = start_year + start_month + start_day

        freq_string = ['T', 'min', 'H', 'D']
        freq_multip = random.choice(range(1, 11))
        random_freq = str(freq_multip) + random.choice(freq_string)

        date_index = pd.date_range(start=date_string, periods=100, freq=random_freq)

        data_length = len(date_index)
        data = np.random.randint(low=-100, high=100, size=data_length)
        time_series = pd.Series(index=date_index, data=data)

        return time_series


def suite():
    s = unittest.TestSuite()
    s.addTest(TestParentMethods(pd, 'infer_freq'))
    for method in dir(TestRandomValidTS):
        if method.startswith('test'):
            s.addTest(TestRandomValidTS(method))
    return s


if __name__ == '__main__':
    runner = unittest.TextTestRunner()
    test_suite = suite()
    print test_suite
    runner.run(test_suite)
