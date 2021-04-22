import numpy    as np
from matplotlib import pyplot as plt
import pandas as pd


def plot_loss_history(history, path):
    """Loss History plotting."""

    fig = plt.figure()
    plt.plot(history.history['loss'],     lw=1, label='Training')
    plt.plot(history.history['val_loss'], lw=1, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid()
    plt.title('Training History')
    fig.savefig(path+'/loss_history.pdf')
    plt.close()


def plot_model_accuracy(history, path):
    """Model Accuracy plotting."""

    fig = plt.figure()
    plt.plot(history.history['accuracy'],     lw=1, label='Training')
    plt.plot(history.history['val_accuracy'], lw=1, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.title('Model Accuracy')
    fig.savefig(path+'/model_acc.pdf')
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
    fig.savefig(path+'/lr_tracker.pdf')
    plt.close()
