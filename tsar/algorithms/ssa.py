"""Singular Spectrum Analysis with numpy

"""

import numpy as np
from scipy.linalg import svd

import tsar


class ssa(object):
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

        self.k = k

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

    def _recompose(self, groups=None):

        U = self.u
        V = self.v

        m = V.shape[0]
        n = U.shape[0]
        S = np.zeros((n, m))
        S[:m,:n] = np.diag(self.s)

        T = np.dot(U, np.dot(S, V))

        assert(np.allclose(self.trajectories, T))

        # Averaging all diagonals

        ts = [np.mean(T[::-1,:].diagonal(i)) for i in range(-T.shape[0]+1,T.shape[1])]

        return ts

    def _get_diagonals(self, a):
        rows, cols = a.shape
        fill = np.zeros(((cols - 1), cols), dtype=a.dtype)
        stacked = np.vstack((a, fill, a))
        major_stride, minor_stride = stacked.strides
        strides = major_stride, minor_stride * (cols + 1)
        shape = (rows + cols - 1, cols)
        return np.lib.stride_tricks.as_strided(stacked, shape, strides)

if __name__ == '__main__':
    import doctest
    import tsar
    import matplotlib.pyplot as plt
    ts = tsar.data.lorenz()['x']
    myssa = ssa(ts)
    ts2 = myssa._recompose()


    fig = plt.figure()
    ax = fig.gca()
    ax.plot(ts, label = 'original')
    ax.plot(ts2, linestyle='--', label = 'reconstruction')
    ax.legend()
    plt.show()
    print len(ts), len(ts2)
    #doctest.testmod()
