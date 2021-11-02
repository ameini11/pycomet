import numpy as np
from matplotlib import pyplot as plt


def plot_test(
    path,
    x,
    y_true,
    y_pred,
    var_names,
    time_idx=-1,
    fig_name=None,
    labels=[None, None],
    scales=['log', 'linear'],
    locs={'col': 3, 'type': 1}
):
    """Variable plotting."""

    fig = plt.figure()
    plt.grid()

    # Preprocess figure -------------------------------------------------------
    x_label, y_label = labels
    x_scale, y_scale = scales

    # X axis
    if isinstance(x, (list,tuple)):
        x = x[time_idx] # Time input
    else:
        x = x[:,time_idx]
    if x_label is not None:
        plt.xlabel(x_label)
    if x_scale is not None:
        plt.xscale(x_scale)
    if x_scale != 'log':
        plt.xlim([min(x), max(x)])

    # Y axis
    if y_label is not None:
        plt.ylabel(y_label)
    if y_scale == 'log':
        plt.yscale(y_scale)
        plt.ylim([np.amin(y_true)*5.e-1, np.amax(y_true)*5.e+0])
    else:
        delta = np.amax(y_true)*0.1
        plt.ylim([np.amin(y_true)-delta, np.amax(y_true)+delta])

    if len(y_true.shape) < 2:
        y_true = np.expand_dims(y_true, 1)

    # Plot figure -------------------------------------------------------------
    # Define styles
    true_style = dict(
        lw=1.
    )
    pred_style = dict(
        marker='^',
        fillstyle='none',
        markersize=5,
        markevery=y_true.shape[0]//20,
        ls='--',
        lw=1.
    )

    # Define parameters
    colors = []
    n = y_true.shape[1]

    # Loop over features
    for d in range(n):

        # >> Plot true values
        plt.plot(
            x,
            y_true[:,d],
            **true_style
        )
        colors.append(plt.gca().lines[-1].get_color())

        # >> Plot predicted values
        plt.plot(
            x,
            y_pred[:,d],
            **pred_style,
            label=var_names[d],
            c=colors[-1]
        )

    # Plot legends
    fontsize = 'x-small'
    if n > 6:
        fontsize = 'xx-small'
    legend_col = plt.legend(
        [plt.plot([], [], c=colors[i])[0] for i in range(len(var_names))],
        var_names,
        fontsize=fontsize,
        loc=locs['col'],
        ncol=int(np.ceil(n/6))
    )
    legend_type = plt.legend(
        [
            plt.plot([], [], c="k", **true_style)[0],
            plt.plot([], [], c="k", **pred_style)[0]
        ],
        ["True", "Pred"],
        fontsize=fontsize,
        loc=locs['type']
    )
    plt.gca().add_artist(legend_col)

    # Save figure
    name = y_scale if fig_name == None else fig_name + '_' + y_scale
    fig_path = path + '/' + name + '.pdf'
    fig.savefig(fig_path)
    plt.close()
