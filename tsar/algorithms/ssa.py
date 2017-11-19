"""Singular Spectrum Analysis with numpy

"""

import numpy as np

import tsar


class Ssa(object):
    """A class for Singular Spectrum Analysis 
    
        
        References
        ----------
        
        [1] Singular Spectrum Analysis for Time Series | Nina Golyandina | Springer. 
        Accessed November 19, 2017. //www.springer.com/gp/book/9783642349126.
        
    """

    def __init__(self, ts, L=None):

        # TODO check types

        self.ts = np.array(ts)
        self.n = len(ts)

        if L is None:
            L = self.n // 2

        self.L = L

        self.trajectories = self._trajectory_embed()

        u, s, v = self._decompose()
        self.u = u
        self.s = s
        self.v = v

    def _trajectory_embed(self):
        """Embed a time series into a L-trajectory matrix
        
        Parameters
        ----------
        L : int
    
        Returns
        -------
        trajectory : np.array
            numpy two-dimensional array of shape (L,N-L+1) containing lagged vectors.
             
        Examples
        --------
        
        >>> ts = np.arange(10)
        >>> trajectory = _trajectory_embed(ts, L=3)
        >>> print trajectory
        [[ 0.  1.  2.  3.  4.  5.  6.  7.]
         [ 1.  2.  3.  4.  5.  6.  7.  8.]
         [ 2.  3.  4.  5.  6.  7.  8.  9.]]
        
    
    
        """

        ts = self.ts
        L = self.L

        n = len(ts)

        # number of lagged vectors

        k = n - L + 1

        trajectory = np.zeros(shape=(L, k))

        for i in range(k):
            trajectory[:, i] = ts[i:i + L]

        return trajectory

    def _decompose(self):
        """Singular value decomposition
        
        Parameters
        ----------
        trajectory
    
        Returns
        -------
        u : array
        s : array
        v : array 
           
        """
        u, s, v = np.linalg.svd(self.trajectories)

        return u, s, v


def _recompose():
    pass


if __name__ == '__main__':
    import doctest

    doctest.testmod()
