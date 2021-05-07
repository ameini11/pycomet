import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def plot_loss_history(history, path, scale='log'):
    """Loss History plotting."""
    fig = plt.figure()

    plt.ylim(limits(history.history['loss']))
    plt.yscale(scale)

    plt.plot(history.history['loss'],     lw=1, label='Training')
    plt.plot(history.history['val_loss'], lw=1, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.title('Training History')
    fig.savefig(path+'/loss_history.pdf')
    plt.close()


def plot_metric(history, path, metric, scale='log'):
    """Metric plotting."""

    fig = plt.figure()

    plt.ylim(limits(history.history[metric]))
    plt.yscale(scale)

    plt.plot(history.history[metric],        lw=1, label='Training')
    plt.plot(history.history['val_'+metric], lw=1, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel(metric)
    plt.legend()
    plt.grid()
    fig.savefig(path+'/metric_'+metric+'.pdf')
    plt.close()


def plot_lr(path, path_file):
    """Learning Rate plotting."""

    data = pd.read_csv(path_file+'/lr_tracker.csv', header=None, skiprows=0)
    data = data.apply(pd.to_numeric, errors='coerce')
    epoch, lr = [ np.expand_dims(np.array(data[i]), axis=1) for i in range(2) ]

    fig = plt.figure()
    plt.plot(epoch, lr, 'r', lw=1)
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.xlabel('Epoch')
    plt.title('Learning Rate')
    plt.grid()
    fig.savefig(path+'/lr.pdf')
    plt.close()


def limits(y):
    if np.log10(y[0])-np.log10(y[-1]) > 3:
        value = np.power(10, np.log10(y[-1])+3)
        delta = np.absolute(y - value)
        index = delta.argmin()
        return None, y[index]
    else: return None, None