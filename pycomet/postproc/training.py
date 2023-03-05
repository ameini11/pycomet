import numpy as np
import pandas as pd

from pycomet import utils
matplotlib, plt = utils.import_matplotlib()


# Plotting training quantities
###########################################################################
def plot_loss(
    history=None,
    path=None,
    scale='log',
    title='Loss',
    var=['loss','val_loss'],
    fig_name='loss',
    label=['Training', 'Validation'],
    window=1
):
    """Loss History plotting."""
    fig = plt.figure()
    plt.yscale(scale)
    for i, var_i in enumerate(var):
        if var_i in history:
            plt.plot(
                moving_average(history[var_i], w=window),
                lw=1,
                label=label[i]
            )
            if window != 1:
                plt.plot(
                    history[var_i],
                    lw=1,
                    c=plt.gca().lines[-1].get_color(),
                    alpha=0.2
                )
    plt.xlabel('Epoch')
    plt.legend()
    plt.title(title + ' History')
    fig.savefig(path + '/' + fig_name + '.pdf')
    plt.close()

def plot_composite_loss(
    history,
    path,
    scale='log',
    window=1
):
    """Residual History plotting."""
    identifier = {
        'ics_loss': 'ICs Loss',
        'pts_loss': 'Anchor pts Loss',
        'res_loss': 'Residual Loss',
        'reg_loss': 'Regularization Loss',
        'loss': 'Loss'
    }

    kwargs = dict(
        history=history,
        path=path,
        scale=scale,
        window=window
    )

    composite_loss = {'var': [], 'label': []}
    for c in history.columns:
        if c.endswith('loss') and not c.startswith('val'):
            if c not in {'loss', 'weighted_loss'}:
                composite_loss['var'].append(c)
                composite_loss['label'].append(identifier[c])
            if c not in {'reg_loss', 'weighted_loss'}:
                plot_loss(
                    title=identifier[c],
                    var=[c,'val_'+c],
                    fig_name=c,
                    **kwargs
                )
    plot_loss(
        **composite_loss,
        title='Losses',
        fig_name='losses',
        **kwargs
    )

def plot_loss_weights(history, path, scale='log'):
    """Loss Weights History plotting."""
    fig = plt.figure()
    plt.yscale(scale)

    labels = {
        'ics': 'ICs',
        'pts': 'Anchor pts',
        'res': 'Residual',
    }
    for c in history.columns:
        if c.endswith('weight'):
            weight_id = c.split("_")[0]
            plt.plot(history[c], lw=1, label=labels[weight_id])

    plt.xlabel('Epoch')
    plt.legend()
    plt.title('Loss Weights')
    fig.savefig(path + '/loss_weights.pdf')
    plt.close()

def plot_metric(history, path, metric, scale='log'):
    """Metric History plotting."""
    fig = plt.figure()

    plt.yscale(scale)

    plt.plot(history[[metric]], lw=1, label='Training')
    if 'val_' + metric in history:
        plt.plot(history[['val_'+metric]], lw=1, label='Validation')
    plt.xlabel('Epoch')
    plt.title(
        r'$\it{%s}$' % metric.replace("_", "\ ").title() + '  Metric History'
    )
    plt.legend()
    fig.savefig(path + '/metric_' + metric + '.pdf')
    plt.close()

def plot_lr(history, path):
    """Learning Rate plotting."""
    df = pd.read_csv(history)
    fig = plt.figure()
    plt.plot(df[['lr']], 'r', lw=1)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('Epoch')
    plt.title('Learning Rate History')
    fig.savefig(path + '/lr.pdf')
    plt.close()


# Util Functions
###########################################################################
def moving_average(x, w=20):
    return np.convolve(x, np.ones(w), 'valid') / w