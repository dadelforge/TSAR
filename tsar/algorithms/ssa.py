"""Singular Spectrum Analysis with numpy

"""

import matplotlib.pyplot as plt
import numpy as np

# TODO: implement a get method to retrieve group time series


class ssa(object):
    """A class for Singular Spectrum Analysis 
    
    Singular Spectrum Analysis (SSA) is a non-parametric method
    to decompose and recompose a signal into specific components:
    trend, seasonality, noise, ... 
    
    The method consists in embedding the time series into a
    trajectory matrix. The matrix is then decomposed into 
    eigentriples using singular value decomposition. Eigentriples 
    can be grouped by user. Each group can be recomposed into a 
    time series component.
    
    Grouping can be user defined based on the interpretation of
    the singular value plot or the w-corr plot.
    
    Parameters
    ----------
    #TODOC
    
    Examples
    --------
    #TODOC
    
        
    References
    ----------
    
    [1] Singular Spectrum Analysis for Time Series | Nina Golyandina | Springer. 
    Accessed November 19, 2017. //www.springer.com/gp/book/9783642349126.
        
    """

    def __init__(self, ts, w=None):

        # TODO check types

        self.ts = np.array(ts)
        self.n = len(ts)

        # define window length if none

        if w is None:
            w = self.n // 2

        self.w = w
        self.k = self.n - self.w + 1
        self.x = self._trajectory_embed()
        self.xrank = np.linalg.matrix_rank(self.x)

        # SVD results
        self.u, self.s, self.v = None, None, None

        self._decompose()



    @property
    def groups(self):
        if not hasattr(self, '_grpmatrix'):
            groups = None
        else:
            groups = self._grpmatrix.keys()
        return groups

    def _trajectory_embed(self):
        """Embed a time series into a L-trajectory matrix    
        """

        ts = self.ts
        w = self.w
        n = self.n
        k = self.k

        trajectory = np.zeros(shape=(w, k))

        for i in range(k):
            trajectory[:, i] = ts[i:i + w]

        return np.matrix(trajectory)

    def _decompose(self):
        """Singular value decomposition           
        """

        # rank of the trajectory matrix x
        d = self.xrank
        x = self.x
        assert (d == min(self.x.shape))

        # decomposition of the trajectory matrix x
        # u and v are unitary and s is a 1-d array of d singular values.

        u, s, v = np.linalg.svd(self.x)

        # note: types are all np.matrix

        self.xi = dict()

        for i in range(d):
            si = np.sqrt(s[i])  # square root of eigenvalue i
            ui = u[:, i]  # eigenvector i corresponding to si
            vi = x.T * ui / si

            self.xi[i] = si * ui * vi.T

        self.svd = [u, s, v]

    def reconstruct(self, groups=None):

        # TODO: somehow ts length is reduced by 1
        # TODO: store residuals

        self._grpmatrix = dict()

        # Define a list of group indexes

        if groups is None:
            idx_list = [range(len(self.xi))]
            names = ['reconstruction']
        else:
            idx_list = [i for i in groups.values()]
            names = [name for name in groups.keys()]

        for name, idx_grp in zip(names, idx_list):
            x_group = [self.xi[key] for key in idx_grp]
            x_sum = np.sum(x_group, axis=0)

            self._grpmatrix[name] = x_sum

        all_grp_idx = [ix for sublist in idx_list for ix in sublist]

        residual_idx = [ix for ix in range(len(self.xi)) if ix not in all_grp_idx]

        x_res = [self.xi[ix] for ix in residual_idx]
        x_res_sum = np.sum(x_res, axis=0)

        if bool(x_res_sum.any()):
            self._grpmatrix['residuals'] = x_res_sum

    def _getseries(self, name):

        x = self._grpmatrix[name]

        # anti diagonal averaging

        ts = [np.mean(x[::-1,:].diagonal(i)) for i in range(-x.shape[0]+1,x.shape[1])]

        return ts

    # --------------------------------------------------------
    # Plotting

    def plot(self, name='values', show=True, **pltkw):

        if name not in self._plotnames:
            names = ','.join(self._plotnames)
            raise AttributeError('Unknown plot name \'{}\'. Name should be on of {}.'.format(name, names))

        if name == 'values':
            fig, ax = self._value_plot(**pltkw)

        if name == 'series':
            fig, ax = self._series_plot(**pltkw)

        plt.tight_layout()

        if show is True:
            plt.show()

        return fig, ax

    @property
    def _plotnames(self):
        names = [
            'values',
            'series'
        ]
        return names

    def _value_plot(self, n=50, **pltkw):
        eigenvalues = self.svd[1]
        fig = plt.figure()
        ax = fig.gca()
        ax.semilogy(eigenvalues[:n], '-ok', markersize=4., alpha=0.5)
        ax.set_ylabel('Component Norms')
        ax.set_xlabel('Index')
        return fig, ax

    def _series_plot(self):

        groups = self.groups
        if len(groups) > 1:

            fig, axarr = plt.subplots(len(groups), 1)

            for i, g in enumerate(groups):
                ts = self._getseries(g)
                axarr[i].plot(ts)
                axarr[i].set_title(g)





        return fig, axarr

    def _wcor_plot(self):
        pass


if __name__ == '__main__':
    import pandas as pd

    co2 = pd.read_csv("co2.csv", index_col=0, header=None)
    co2 = co2[co2.columns[0]]
    co2_ssa = ssa(co2)
    print co2_ssa.groups
    groups = {
        'Trend': [0, 3],
        'Season1': [1,2],
        'Season2': [4,5]
    }
    co2_ssa.reconstruct(groups)
    print co2_ssa.groups

    co2_ssa.plot('series')

    #print co2_ssa.plot(name='series')

    # doctest.testmod()
