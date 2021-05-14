import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


###########################################################################
### Plotting Training Quantities
###########################################################################
def plot_loss(history, path, scale='log'):
    """Loss History plotting."""
    data = get_data(history)

    fig = plt.figure()

    plt.ylim(limits(data[['loss','val_loss']]))
    plt.yscale(scale)

    plt.plot(data[['loss']],     lw=1, label='Training')
    plt.plot(data[['val_loss']], lw=1, label='Validation')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.title('Loss History')
    fig.savefig(path+'/loss.pdf')
    plt.close()

def plot_residual(history, path, scale='log'):
    """Residual History plotting."""
    data = get_data(history)

    fig = plt.figure()

    plt.ylim(limits(data[['res_loss','val_res_loss']]))
    plt.yscale(scale)

    plt.plot(data[['res_loss']],     lw=1, label='Training')
    plt.plot(data[['val_res_loss']], lw=1, label='Validation')
    plt.xlabel('Epoch')
    plt.legend()
    plt.grid()
    plt.title('Residual History')
    fig.savefig(path+'/residual.pdf')
    plt.close()

def plot_metric(history, path, metric, scale='log'):
    """Metric History plotting."""
    data = get_data(history)

    fig = plt.figure()

    plt.ylim(limits(data[[metric,'val_'+metric]]))
    plt.yscale(scale)

    plt.plot(data[[metric]],        lw=1, label='Training')
    plt.plot(data[['val_'+metric]], lw=1, label='Validation')
    plt.xlabel('Epoch')
    plt.title(r'$\it{%s}$' % metric.replace("_", "\ ").title() + '  Metric History')
    plt.legend()
    plt.grid()
    fig.savefig(path+'/metric_'+metric+'.pdf')
    plt.close()

def plot_lr(history, path):
    """Learning Rate plotting."""
    data = get_data(history)

    fig = plt.figure()
    plt.plot(data[['lr']], 'r', lw=1)
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

def get_data(history):
    if isinstance(history, str):
        return pd.read_csv(history)
    else:
        return pd.DataFrame.from_dict(history.history)