import numpy as np
import matplotlib as mpl

from matplotlib import pyplot as plt

colors = mpl.rcParams['axes.prop_cycle'].by_key()['color']


def plot_test(
    path,
    x,
    y,
    y_pred=None,
    var_names=None,
    fig_name=None,
    labels=[None, None],
    scales=['log', 'linear'],
    title=None,
    xlim=None,
    ylim=None,
    save=True,
    fig_ax=None
):
    if fig_ax is None:
        fig, ax = plt.subplots()
    else:
        fig, ax = fig_ax

    # Set up figure -----------------------------------------------------------
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

    # Plot figure -------------------------------------------------------------
    # Define styles
    true_style = dict(
        lw=1.
    )
    pred_style = dict(
        marker='^',
        fillstyle='none',
        markersize=5,
        markevery=y.shape[0]//20,
        ls='--',
        lw=1.
    )

    # Define parameters
    colors = []
    n = y.shape[1]

    # Loop over features
    for d in range(n):

        # >> Plot true values
        plt.plot(
            x,
            y[:,d],
            c=colors[d],
            **true_style
        )

        # >> Plot predicted values
        plt.plot(
            x,
            y_pred[:,d],
            **pred_style,
            c=colors[d],
            label=var_names[d]
        )
    # Add legend
    if var_names is not None:
        if n > 12:
            fs = 'x-small'
        elif n > 6:
            fs = 'small'
        else:
            fs = 'medium'
        ax.legend(
            [plt.plot([], [], c=colors[i])[0] for i in range(len(var_names))],
            var_names,
            fontsize=fs,
            ncol=int(np.ceil(n/3))
        )

    # Save figure -------------------------------------------------------------
    if save:
        # Save figure
        name = y_scale if fig_name == None else fig_name + '_' + y_scale
        fig_path = path + '/' + name + '.pdf'
        fig.savefig(fig_path)
        
    if fig_ax is not None:
        return fig, ax
