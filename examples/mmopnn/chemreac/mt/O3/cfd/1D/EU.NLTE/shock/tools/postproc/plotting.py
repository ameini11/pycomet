import numpy as np
import matplotlib as mpl

from matplotlib import cm
from matplotlib import pyplot as plt 


def time_space_2d(
    t,
    x,
    y,
    tlim=None,
    xlim=None,
    tshift=None,
    xshift=None,
    eps=1.e-3,
    show=False,
    cbar_kwargs={'label': '', 'format': '%.1f'}
):
    fig = plt.figure()
    # Set 'time' axis
    t_s, t_e = get_lim(t, tlim, eps)
    t = t[t_s:t_e]
    if tshift is not None:
        t = t + tshift
    plt.xlabel('t [s]')
    # Set 'space' axis
    x_s, x_e = get_lim(x, xlim, eps)
    x = x[x_s:x_e]
    if xshift is not None:
        x = x + xshift
    plt.ylabel('x [m]')
    # Set notation
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    # Mesh
    T, X = np.meshgrid(t, x)
    # Set variables
    Y = y[t_s:t_e,x_s:x_e].T
    # Plot
    p = plt.pcolormesh(T, X, Y, cmap=cm.viridis)
    # Color bar
    fmt = mpl.ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    cbar = plt.colorbar(p, format=fmt)
    if show:
        plt.show()
    plt.close()
    
def get_lim(x, lim, eps):
    if lim is not None:
        x_s = np.where(x >= lim[0]-lim[0]*eps)[0][ 0] if lim[0] is not None else 0
        x_e = np.where(x <= lim[1]+lim[1]*eps)[0][-1] if lim[1] is not None else -1
    else:
        x_s, x_e = 0, -1
    return x_s, x_e
