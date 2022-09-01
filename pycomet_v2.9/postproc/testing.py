import numpy as np

from pycomet import utils
matplotlib, plt = utils.import_matplotlib()
COLORS = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']


def plot_test(
    path=None,
    x=None,
    y=None,
    y_pred=None,
    var_names=None,
    fig_name=None,
    labels=[None, None],
    scales=['log', 'linear'],
    title=None,
    xlim=None,
    ylim=None,
    save=True,
    fig_ax=None,
    max_num_var=10
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
    n = y.shape[1]
    # Check number of colors available
    if n > len(COLORS):
        colors = COLORS*int(np.ceil(n/len(COLORS)))
    else:
        colors = COLORS
    # Loop over features
    for d in range(n):
        # >> Plot true values
        ax.plot(
            x,
            y[:,d],
            c=colors[d],
            **true_style
        )
        # >> Plot predicted values
        ax.plot(
            x,
            y_pred[:,d],
            **pred_style,
            c=colors[d],
            label=var_names[d]
        )
    # Add legend
    if var_names is not None:
        idx = np.arange(0,n)
        if n > 6:
            if n > max_num_var:
                step = int(np.ceil(n/max_num_var))
                idx = np.arange(0,n,step)
                n = max_num_var
            fs = 'small'
        else:
            fs = 'medium'
        ax.legend(
            [plt.plot([], [], c=colors[i])[0] for i in idx],
            [var_names[i] for i in idx],
            fontsize=fs,
            ncol=int(np.ceil(n/3))
        )

    # Save figure -------------------------------------------------------------
    if save:
        # Save figure
        name = f'output_{y_scale}' if fig_name == None else fig_name
        fig.savefig(path + '/' + name + '.pdf')
        
    if fig_ax is not None:
        return fig, ax

    plt.close()