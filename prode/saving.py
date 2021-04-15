import sys
import os, errno
import os.path
from os import path

import numpy as np
import tensorflow as tf
import h5py



def save_loss_history(losshistory, fname):
    print("Saving loss history to {} ...".format(fname))
    loss = np.hstack(
        (
            np.array(losshistory.steps)[:, None],
            np.array(losshistory.loss_train),
            np.array(losshistory.loss_test),
            np.array(losshistory.metrics_test),
        )
    )
    np.savetxt(fname, loss, header="step, loss_train, loss_test, metrics_test")


def saveplot(losshistory, train_state, issave=True, isplot=True):
    if issave:
        save_loss_history(losshistory, "loss.dat")
        save_best_state(train_state, "train.dat", "test.dat")

    if isplot:
        plot_loss_history(losshistory)
        plot_best_state(train_state)
        plt.show()