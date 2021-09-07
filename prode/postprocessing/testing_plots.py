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
    scales=['log', 'linear']
):
    """Variable plotting."""

    fig = plt.figure()
    plt.grid()

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

    # Plotting
    lin = ['-', '--']
    col = ['k', 'r', 'g', 'b', 'tab:purple', 'tab:brown', 'tab:pink', \
           'tab:blue', 'tab:red', 'tab:green', 'tab:orange']
    marker_style = dict(
        marker='^',
        fillstyle='none',
        markersize=5,
        markevery=y_true.shape[0]//20
    )

    # Solution
    if len(y_true.shape) < 2:
        y_true = np.expand_dims(y_true, 1)

    for d in range(y_true.shape[1]):

        name = 'True ' + var_names[d]
        plt.plot(x, y_true[:, d], c=col[d], ls=lin[0], lw=1, label=name)

        name = 'Pred ' + var_names[d]
        plt.plot(x, y_pred[:, d], c=col[d], ls=lin[1], lw=1, label=name, \
            **marker_style)

    plt.legend(fontsize='x-small')
    name = y_scale if fig_name == None else fig_name + '_' + y_scale
    fig_path = path + '/' + name + '.pdf'
    fig.savefig(fig_path)
    plt.close()
