# coding=utf-8
import numpy as np
import tsar


def lorenz(xi=0., yi=1., zi=1.05, s=10, r=28, b=2.667, n=10000, dt=0.1):
    """
    
    :..math
        x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    
    Parameters
    ----------
    xi
    yi
    zi
    s
    r
    b

    Returns
    -------
    
    References
    ----------
    
    Lorenz, Edward N. “Deterministic Nonperiodic Flow.” Journal of the Atmospheric Sciences 20, 
    no. 2 (March 1, 1963): 130–41. https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2.


    """

    # Initialization of data storage
    xs = np.empty((n + 1,))
    ys = np.empty((n + 1,))
    zs = np.empty((n + 1,))

    # Lorenz sol

    # Solving

    for i in range(n):
        # Derivatives of the X, Y, Z state
        x_dot = s*(ys[i] - xs[i])
        y_dot = r*xs[i] - ys[i] -xs[i]*zs[i]
        z_dot = xs[i]*ys[i] - b*zs[i]
        xs[i + 1] = xs[i] + (x_dot * dt)
        ys[i + 1] = ys[i] + (y_dot * dt)
        zs[i + 1] = zs[i] + (z_dot * dt)

    data = tsar.TVPDataFrame(data=[xs, ys, zs])

    return data