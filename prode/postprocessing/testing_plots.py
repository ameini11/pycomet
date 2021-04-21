import numpy as np
from matplotlib import pyplot as plt


def plot_test(
    path,
    x,
    y_true,
    y_pred,
    var_name,
    title,
    test_num,
    labels=[None, None],
    scales=['log', 'linear'],
    data_val=None):
    """Variable plotting."""

    fig = plt.figure()
    plt.grid()
    plt.title(title)

    x_label, y_label = labels
    x_scale, y_scale = scales

    # X axis
    if isinstance(x, (list,tuple)): x = x[1]
    if x_label is not None: plt.xlabel(x_label)
    if x_scale is not None: plt.xscale(x_scale)
    plt.xlim([min(x), max(x)])

    # Y axis
    if y_label is not None: plt.ylabel(y_label)
    if y_scale == 'log':
        plt.yscale(y_scale)
        plt.ylim([np.amin(y_true[-1,:])*5.e-1, np.amax(y_true[-1,:])*5.e+0])
    else:
        delta = np.amax(y_true)*0.1
        plt.ylim([np.amin(y_true)-delta, np.amax(y_true)+delta])

    # Plotting
    lin = ['-', '--']
    col  = ['k', 'r', 'g', 'b', 'tab:purple', 'tab:brown', 'tab:pink', \
            'tab:blue', 'tab:red', 'tab:green', 'tab:orange']

    # Solution
    for d in range(y_true.shape[1]):

        name = 'True ' + var_name[d]
        plt.plot(x, y_true[:, d], c=col[d], ls=lin[0], lw=1, label=name)

        name = 'Pred ' + var_name[d]
        plt.plot(x, y_pred[:, d], c=col[d], ls=lin[1], lw=1, label=name)

    # Validation Points
    if data_val is not None:
        t_val, y_val = data_val
        marker_style = dict(c=col[-1], marker='x', lw=0.5, \
                            linestyle='none', fillstyle='none', markersize=5)
        plt.plot(t_val, y_val[:,0], **marker_style, label='Validation pts')
        plt.plot(t_val, y_val[:,1:], **marker_style)


    plt.legend(fontsize='x-small')
    fig_path = path + '/TestCase' + str(test_num+1) + '_' + y_scale + '.jpeg'
    fig.savefig(fig_path)
    plt.close()
