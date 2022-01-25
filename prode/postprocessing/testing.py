import numpy as np

from matplotlib import pyplot as plt


def plot_test(
    path,
    x,
    y,
    y_pred,
    var_names,
    fig_name=None,
    labels=[None, None],
    scales=['log', 'linear']
):
    """Variable plotting."""

    fig = plt.figure()

    # Set up figure -----------------------------------------------------------
    x_label, y_label = labels
    x_scale, y_scale = scales

    # X axis
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
        plt.ylim([np.amin(y)*5.e-1, np.amax(y)*5.e+0])
    else:
        delta = np.amax(y)*0.1
        plt.ylim([np.amin(y)-delta, np.amax(y)+delta])

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

    # Plot color legends
    if n > 12:
        fs = 'x-small'
    elif n > 6:
        fs = 'small'
    else:
        fs = 'medium'
    plt.legend(
        [plt.plot([], [], c=colors[i])[0] for i in range(len(var_names))],
        var_names,
        fontsize=fs,
        ncol=int(np.ceil(n/3))
    )

    # Save figure -------------------------------------------------------------
    name = y_scale if fig_name == None else fig_name + '_' + y_scale
    fig_path = path + '/' + name + '.pdf'
    fig.savefig(fig_path)
    plt.close()
