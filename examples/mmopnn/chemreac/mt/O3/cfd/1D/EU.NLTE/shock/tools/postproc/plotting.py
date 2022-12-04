import os
import numpy as np
# import matplotlib as mpl

# from matplotlib import cm
# from matplotlib import pyplot as plt


import matplotlib as mpl
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# from matplotlib.animation import FuncAnimation
# from IPython.display import HTML
# %matplotlib notebook
# mpl.use('pgf')

# Use custom style
# See: https://matplotlib.org/1.5.3/users/style_sheets.html
style_name = 'paper_1column'
mpl.style.use(
    os.environ['WORKSPACE_PATH'] \
        + '/Research/styles/matplotlib/' \
        + style_name + '.mplstyle'
)
# Get colors list
COLORS = mpl.rcParams['axes.prop_cycle'].by_key()['color']*10


# def time_space_2d(
#     t,
#     x,
#     y,
#     tlim=None,
#     xlim=None,
#     tshift=None,
#     xshift=None,
#     eps=1.e-3,
#     cbar_lbl='',
#     cbar_fmt=None,
#     fig_name=None,
#     path=None,
#     save=True,
#     show=False
# ):
#     fig, ax = plt.subplots()
#     # Set 'time' axis
#     t_s, t_e = get_lim(t, tlim, eps)
#     t = t[t_s:t_e]
#     if tshift is not None:
#         t = t + tshift
#     ax.set_xlabel('$t$ [s]')
#     # Set 'space' axis
#     x_s, x_e = get_lim(x, xlim, eps)
#     x = x[x_s:x_e]
#     if xshift is not None:
#         x = x + xshift
#     ax.set_ylabel('$x$ [m]')
#     # Set notation
#     ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
#     # Mesh
#     T, X = np.meshgrid(t, x)
#     # Set variables
#     Y = y[t_s:t_e,x_s:x_e].T
#     # Plot
#     p = ax.pcolormesh(T,X,Y,cmap=cm.viridis,linewidth=0,rasterized=True)
#     # Set aspect
#     ax.set_aspect(1./ax.get_data_ratio()), adjustable='box')
#     # Color bar
#     if cbar_fmt is None:
#         cbar_fmt = mpl.ticker.ScalarFormatter(useMathText=True)
#         cbar_fmt.set_powerlimits((0, 0))
#     # divider = make_axes_locatable(ax)
#     # cax = divider.append_axes("right", size="5%", pad=0.05)
#     cbar = plt.colorbar(p, format=cbar_fmt, label=cbar_lbl)#, cax=cax)
#     # Save figure
#     if save:
#         fig_path = path + '/' + fig_name
#         fig.savefig(fig_path+'.pdf')
#     # Show and close figure
#     if show:
#         plt.show()
#     plt.close()
    
    

def time_space_2d(
    t,
    x,
    y,
    tlim=None,
    xlim=None,
    tshift=None,
    xshift=None,
    eps=2.e-3,
    clbl='',
    cfmt=None,
    cticks=None,
    corientation='horizontal',
    compute_cticks=True,
    interp='bilinear',
    figname=None,
    path=None,
    save=True,
    show=False
):
    fig, ax = plt.subplots(figsize=(8,8))
    # Set 'time' axis
    t_s, t_e = get_lim(t, tlim, eps)
    t = t[t_s:t_e]
    if tshift is not None:
        t = t + tshift
    ax.set_xlabel('$t$ [s]')
    # Set 'space' axis
    x_s, x_e = get_lim(x, xlim, eps)
    x = x[x_s:x_e]
    if xshift is not None:
        x = x + xshift
    x = x * 1.e3
    ax.set_ylabel('$x$ [mm]')
    # Set notation
    ax.ticklabel_format(axis='both', style='sci', scilimits=(0,0))
    # Plot
    Y = y[t_s:t_e,x_s:x_e].T
    im = ax.imshow(
        Y,
        interpolation=interp,
        cmap=cm.viridis,
        origin='lower',
        aspect='auto',
        extent=[np.amin(t), np.amax(t), np.amin(x), np.amax(x)]
    )
    # Color bar
    if cfmt is None:
        cfmt = mpl.ticker.ScalarFormatter(useMathText=True)
        cfmt.set_powerlimits((0, 0))
    cax = inset_axes(
        ax,
        width="100%",
        height="4%",
        loc="lower left",
        bbox_to_anchor=(0, 1.02, 1., 1.02),
        bbox_transform=ax.transAxes,
        borderpad=0
    )
    # cax = inset_axes(
    #     ax,
    #     width="4%",
    #     height="100%",
    #     loc="lower left",
    #     bbox_to_anchor=(1.02, 0., 1, 1),
    #     bbox_transform=ax.transAxes,
    #     borderpad=0
    # )
    if compute_cticks:
        cticks = np.linspace(np.amin(Y), np.amax(Y), 5)
    cbar = plt.colorbar(
        im,
        format=cfmt,
        label=clbl,
        cax=cax,
        ticks=cticks,
        orientation='horizontal'
    )
    cax.xaxis.tick_top()
    cax.xaxis.set_label_position('top')
    # Save figure
    if save:
        fig.savefig(path + f'/{figname}.pdf')
    # Show and close figure
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


def test(
    x,
    y,
    y_pred=None,
    var_names=None,
    fig_name=None,
    labels=[None, None],
    scales=['linear', 'linear'],
    title=None,
    xlim=None,
    ylim=None,
    path=None,
    save=True,
    show=False
):
    fig, ax = plt.subplots()
    
    # Set up figure
    if title is not None:
        ax.set_title(title)
    x_label, y_label = labels
    x_scale, y_scale = scales
    # X axis
    if x_label is not None:
        ax.set_xlabel(x_label)
    if x_scale is not None:
        ax.set_xscale(x_scale)
    if xlim is not None:
        ax.set_xlim(xlim)
    elif x_scale != 'log':
        ax.set_xlim([min(x), max(x)])
    # Y axis
    if y_label is not None:
        ax.set_ylabel(y_label)
    if y_scale == 'log':
        ax.set_yscale(y_scale)
        if ylim is None:
            ax.set_ylim([np.amin(y)*5.e-1, np.amax(y)*5.e+0])
    else:
        delta = np.amax(y)*0.1
        if ylim is None:
            ax.set_ylim([np.amin(y)-delta, np.amax(y)+delta])
    if ylim is not None:
        ax.set_ylim(ylim)
    # Check 2nd dim: looping over columns
    if len(y.shape) < 2:
        y = np.expand_dims(y, 1)
    # Set notation
    plt.ticklabel_format(axis='both', style='sci', scilimits=(0,0))

    # Define styles
    true_style = dict(
        lw=2.,
    )
    pred_style = dict(
        marker='^',
        markersize=10,
        lw=2.,
        fillstyle='none',
        markevery=y.shape[0]//50,
        ls='--'
    )
    # Define parameters
    n = y.shape[1]
    # Loop over features
    for d in range(n):
        # >> Plot true values
        ax.plot(
            x,
            y[:,d],
            **true_style,
            c=COLORS[d]
        )
        # >> Plot predicted values
        ax.plot(
            x,
            y_pred[:,d],
            **pred_style,
            c=COLORS[d]
        )
    # Add legend
    if var_names is not None:
        idx = np.arange(len(var_names))
        ax.legend(
            [plt.plot([], [], c=COLORS[i])[0] for i in idx],
            [var_names[i] for i in idx],
            ncol=int(np.ceil(n/3))
        )

    # Save figure
    if save:
        fig_path = path + '/' + fig_name
        for ext in ('pdf', 'pgf'):
            fig.savefig(fig_path+'.'+ext)

    # Show and close figure
    if show:
        plt.show()
    plt.close()