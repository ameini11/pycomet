import numpy as np

from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from matplotlib import ticker
from matplotlib.animation import FuncAnimation


# Reading
###############################################################################
# Levels number densities
def get_n_ov_g(instants, file, n_lev):
    with open(file, "r") as f:
        lines = f.readlines()
        del lines[:1]
    n = []
    for i in tqdm(range(instants)):
        n.append(
            np.loadtxt(
                lines[i*(n_lev+1)+1:(n_lev+1)*(i+1)], dtype=float
            ).T[1].reshape(1,-1)
        )
    return np.vstack(n)

# Plotting
###############################################################################
style = dict(
    markeredgewidth=1,
    lw=1.,
    linestyle='',
    markersize=1,
    marker='x'
)

# 3D Animation
###############################################################################
# Log-scale ticker
def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"

# Initialize lines for levels distribution
def init_3D_lines(
    ax,
    axim=0.,
    ols=None
):
    # Set up axes
    ax.set_xlabel(r'$\epsilon_v$ [eV]')
    ax.set_ylabel(r'$\epsilon_r$ [eV]')
    ax.set_zlabel(r'$n_i/g_i$ [m$^{-3}$]')
    ax.zaxis.set_major_formatter(
        ticker.FuncFormatter(log_tick_formatter)
    )
    ax.zaxis.set_major_locator(
        ticker.MaxNLocator(integer=True)
    )
    # Initialize lines.Line2D objects
    lines = [ax.plot3D([],[],[],c='k',**style)[0]]
    if ols is not None:
        lines.append(ax.plot3D([],[],[],c='r',**style)[0])
    ax.view_init(elev=30, azim=axim)
    return ax, lines

# Create animation
def get_3D_animation(
    t_vec,
    n_ov_g,
    frames,
    axim=0.,
    ols=None,
    lev_dtb=None
):
    # Initialize a figure in which the graphs will be plotted
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    # Initialize levels distribution lines objects
    ax, lines = init_3D_lines(ax, axim=axim, ols=ols)
    xlim = [np.amin(lev_dtb['EVib']), np.amax(lev_dtb['EVib'])]
    ylim = [np.amin(lev_dtb['ERot']), np.amax(lev_dtb['ERot'])]
    t_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes)

    def animate(frame):
        print(f"Frame {frame+1}/{frames}", end = "\r")
        idx = int(frame/frames*len(t_vec))
        t_text.set_text(r'$t$ = %.2e s' % t_vec[idx])
        # Plot distribution
        for l in lines:
            l.set_xdata(lev_dtb['EVib'])
            l.set_ydata(lev_dtb['ERot'])
        true = np.log10(n_ov_g[idx])
        lines[0].set_3d_properties(true)
        if ols is not None:
            pred = np.log10(ols(n_ov_g[idx]))
            lines[1].set_3d_properties(pred)
        # Rescale axis limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if ols is not None:
            zlim = [
                min(np.amin(true),np.amin(pred)),
                max(np.amax(true),np.amax(pred))
            ]
        else:
            zlim = [np.amin(true), np.amax(true)]
        ax.set_zlim(zlim)
        return lines
        
    # Get animation
    return FuncAnimation(
        fig,
        animate,
        frames=frames,
        blit=True
    )

# 2D Animation
###############################################################################
# Initialize lines for levels distribution
def init_2D_lines(
    ax,
    ols=None
):
    # Set up axes
    ax.set_xlabel(r'$\epsilon_i$ [eV]')
    ax.set_ylabel(r'$n_i/g_i$ [m$^{-3}$]')
    ax.set_yscale('log')
    # Initialize lines.Line2D objects
    lines = [ax.plot([],[],[],c='k',**style)[0]]
    if ols is not None:
        lines.append(ax.plot([],[],[],c='r',**style)[0])
    return ax, lines

# Create animation
def get_2D_animation(
    t_vec,
    n_ov_g,
    frames,
    ols=None,
    lev_dtb=None
):
    # Initialize a figure in which the graphs will be plotted
    fig = plt.figure()
    ax = fig.add_subplot()
    # Initialize levels distribution lines objects
    ax, lines = init_2D_lines(ax, ols=ols)
    xlim = [np.amin(lev_dtb['E']), np.amax(lev_dtb['E'])]
    t_text = ax.text(0.75, 0.9, "", transform=ax.transAxes)

    def animate(frame):
        print(f"Frame {frame+1}/{frames}", end = "\r")
        idx = int(frame/frames*len(t_vec))
        t_text.set_text(r'$t$ = %.2e s' % t_vec[idx])
        # Plot distribution
        for l in lines:
            l.set_xdata(lev_dtb['E'])
        true = n_ov_g[idx]
        lines[0].set_ydata(true)
        if ols is not None:
            pred = ols(n_ov_g[idx])
            lines[1].set_ydata(pred)
        # Rescale axis limits
        ax.set_xlim(xlim)
        if ols is not None:
            ylim = [
                min(np.amin(true),np.amin(pred)),
                max(np.amax(true),np.amax(pred))
            ]
        else:
            ylim = [np.amin(true), np.amax(true)]
        ax.set_ylim(ylim)
        return lines
        
    # Get animation
    return FuncAnimation(
        fig,
        animate,
        frames=frames,
        blit=True
    )

# OLS Fitting
###############################################################################
def get_ols_fitting(X):
    X_tr = np.transpose(X)
    X_inv = np.linalg.inv(np.matmul(X_tr,X))
    def ols(n_ov_g):
        ln_n = np.log(1./n_ov_g)
        Y = np.matmul(X_tr, ln_n.reshape(-1,1))
        w = np.matmul(X_inv, Y)
        return np.exp(-np.matmul(X, w).squeeze())
    return ols
