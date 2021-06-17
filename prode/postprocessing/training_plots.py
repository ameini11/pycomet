import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


###########################################################################
### Plotting Training Quantities
###########################################################################
def plot_loss(history, path, scale='log', limit_y=False, 
              title='Loss', var=['loss','val_loss'], fig_name='loss',
              labels=['Training', 'Validation']):
    """Loss History plotting."""
    fig = plt.figure()

    if limit_y:
        plt.ylim(limits(history[var]))
    plt.yscale(scale)

    for i, var_i in enumerate(var):
        if var_i in history:
            plt.plot(history[var_i], lw=1, label=labels[i])
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.title(title+' History')
    fig.savefig(path+'/'+fig_name+'.pdf')
    plt.close()


def plot_composite_loss(history, path, scale='log', limit_y=False):
    """Residual History plotting."""
    identifier = {
        'ics': 'ICs Loss',
        'pts': 'Operator Loss',
        'res': 'Residual Loss',
        'reg': 'Regularization Loss',
        'tot': 'Total Loss'
        }

    composite_loss = {'var': [], 'labels': []}
    for c in history.columns:
        if c.endswith('loss') and not c.startswith('val'):
            loss_id = c.split("_")[0]
            if loss_id != 'tot':
                composite_loss['var'].append(c)
                composite_loss['labels'].append(identifier[loss_id])
            if loss_id == 'reg':
                continue
            plot_loss(history, path, scale=scale, limit_y=limit_y,
                title=identifier[loss_id], var=[c,'val_'+c], fig_name=c)
    plot_loss(history, path, scale=scale, limit_y=limit_y,
        **composite_loss, title='Losses', fig_name='losses')


def plot_loss_weights(history, path, scale='log'):
    """Loss Weights History plotting."""
    fig = plt.figure()
    plt.yscale(scale)

    labels = {
        'ics': 'ICs',
        'pts': 'Operator',
        'res': 'Residual',
        }
    for c in history.columns:
        if c.endswith('weight'):
            loss_id = c.split("_")[0]
            plt.plot(history[c], lw=1, label=labels[loss_id])

    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.title('Loss Weights')
    fig.savefig(path+'/loss_weights.pdf')
    plt.close()


def plot_metric(history, path, metric, scale='log', limit_y=False):
    """Metric History plotting."""
    fig = plt.figure()

    if limit_y:
        plt.ylim(limits(history[[metric,'val_'+metric]]))
    plt.yscale(scale)

    plt.plot(history[[metric]], lw=1, label='Training')
    if 'val_'+metric in history:
        plt.plot(history[['val_'+metric]], lw=1, label='Validation')
    plt.xlabel('Epoch')
    plt.title(r'$\it{%s}$' % metric.replace("_", "\ ").title() + '  Metric History')
    plt.legend()
    plt.grid()
    fig.savefig(path+'/metric_'+metric+'.pdf')
    plt.close()


def plot_lr(history, path):
    """Learning Rate plotting."""
    df = pd.read_csv(history)
    fig = plt.figure()
    plt.plot(df[['lr']], 'r', lw=1)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('Epoch')
    plt.title('Learning Rate History')
    plt.grid()
    fig.savefig(path+'/lr.pdf')
    plt.close()


###########################################################################
### Util Functions
###########################################################################
def limits(y):
    y = y.to_numpy()
    y_min, y_max = np.amin(y[-100:-1,:]), np.amax(y)
    if np.log10(y_max)-np.log10(y_min) > 2:
        value = np.power(10, np.log10(y_min)+2)
        delta = np.absolute(y - value)
        index = np.unravel_index(delta.argmin(), delta.shape)
        return y_min*0.9, y[index]
    else:
        return y_min*0.9, y_max*1.1
