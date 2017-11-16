# coding=utf-8
import numpy as np
import pandas as pd
from scipy.integrate import odeint
import tsar


def _lorenz(state, t, s=10., r=28., b=2.667):
    """
    
    References
    ----------
    
    Lorenz, Edward N. “Deterministic Nonperiodic Flow.” Journal of the Atmospheric Sciences 20, 
    no. 2 (March 1, 1963): 130–41. https://doi.org/10.1175/1520-0469(1963)020<0130:DNF>2.0.CO;2.
    
    https://matplotlib.org/examples/mplot3d/lorenz_attractor.html


    """

    # unpack the state vector
    x = state[0]
    y = state[1]
    z = state[2]

    x_dot = s * (y-x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z

    return [x_dot, y_dot, z_dot]

def lorenz(xi=2., yi=3., zi=4., n=10000, dt=0.01):
    t = np.arange(start=0, stop=n ) * dt

    states = odeint(_lorenz, [xi,yi,zi], t)

    data = pd.DataFrame(data=states, columns=['x', 'y', 'z'])
    return data

if __name__=='__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    data = lorenz()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(data['x'], data['y'], data['z'], lw=1., alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    print data.describe()
    plt.show()

