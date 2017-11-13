"""Python model to handle univariate and mutivariate time series object with pandas

The purposes of the time series library is to ensure that time series index is consistent.
And that the variable is numeric

"""

import pandas as pd
import numpy as np
import random
from tvpexept import *

class TVPSeries(pd.Series):
    def __init__(self, *args, **kwargs):
        super(TVPSeries, self).__init__(*args, **kwargs)

        # check for datetime index
        if not isinstance(self.index, pd.DatetimeIndex):
            raise TypeError('Index is not type{}'.format(pd.DatetimeIndex.__name__))

        # check if datetime index is monotonic increasing
        if not self.index.is_monotonic_increasing:
            raise
        assert self.index.is_monotonic_increasing, 'Index is not monotonic increasing'
        assert self.index.inferred_freq is not None, 'Index has no fixed frequency'
        assert np.issubdtype(self.dtype, np.number), 'Data is not numeric'

class TVPDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super(TVPDataFrame, self).__init__(*args, **kwargs)
        assert isinstance(self.index, pd.DatetimeIndex), 'Index is not type{}'.format(pd.DatetimeIndex.__name__)
        assert self.index.is_monotonic_increasing, 'Index is not monotonic increasing'
        assert self.index.inferred_freq is not None, 'Index has no fixed frequecy'
        assert self.dtypes.apply(lambda x: np.issubdtype(x, np.number)).all(), 'Data is not numeric'


def cross_period(ts1, ts2):
    assert(isinstance(ts1, TVPSeries) or isinstance(ts1, TVPDataFrame))
    assert(isinstance(ts2, TVPSeries) or isinstance(ts2, TVPDataFrame))
    assert(ts1.index.inferred_freq == ts2.index.inferred_freq)
    date_ix1 = ts1.index
    date_ix2 = ts2.index

    pass

if __name__ == '__main__':
    data = generate_valid_ts()
    ts = TVPSeries(index=data.index, data=data.values)
    print ts
