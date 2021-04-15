import os
import sys
import matplotlib
matplotlib.use("Agg")
import math
import tensorflow                             as tf
import numpy                                  as np
from matplotlib                           import pyplot as plt
from matplotlib                           import animation


#=======================================================================================================================================
def plot_losseshistory(InputData, history):

    fig = plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.title('Training History')
    FigPath = InputData.PathToFigFld + '/LossesHistory.png'
    fig.savefig(FigPath, dpi=1000)
    #plt.show()
    plt.close()

#=======================================================================================================================================


def plot_loss_history(losshistory):
    loss_train = np.sum(losshistory.loss_train, axis=1)
    loss_test = np.sum(losshistory.loss_test, axis=1)

    plt.figure()
    plt.semilogy(losshistory.steps, loss_train, label="Train loss")
    plt.semilogy(losshistory.steps, loss_test, label="Test loss")
    for i in range(len(losshistory.metrics_test[0])):
        plt.semilogy(
            losshistory.steps,
            np.array(losshistory.metrics_test)[:, i],
            label="Test metric",
        )
    plt.xlabel("# Steps")
    plt.legend()


def plot_best_state(train_state):
    X_train, y_train, X_test, y_test, best_y, best_ystd = train_state.packed_data()

    y_dim = best_y.shape[1]

    # Regression plot
    if X_test.shape[1] == 1:
        idx = np.argsort(X_test[:, 0])
        X = X_test[idx, 0]
        plt.figure()
        for i in range(y_dim):
            if y_train is not None:
                plt.plot(X_train[:, 0], y_train[:, i], "ok", label="Train")
            if y_test is not None:
                plt.plot(X, y_test[idx, i], "-k", label="True")
            plt.plot(X, best_y[idx, i], "--r", label="Prediction")
            if best_ystd is not None:
                plt.plot(
                    X, best_y[idx, i] + 2 * best_ystd[idx, i], "-b", label="95% CI"
                )
                plt.plot(X, best_y[idx, i] - 2 * best_ystd[idx, i], "-b")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
    elif X_test.shape[1] == 2:
        for i in range(y_dim):
            plt.figure()
            ax = plt.axes(projection=Axes3D.name)
            ax.plot3D(X_test[:, 0], X_test[:, 1], best_y[:, i], ".")
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_zlabel("$y_{}$".format(i + 1))

    # Residual plot
    if y_test is not None:
        plt.figure()
        residual = y_test[:, 0] - best_y[:, 0]
        plt.plot(best_y[:, 0], residual, "o", zorder=1)
        plt.hlines(0, plt.xlim()[0], plt.xlim()[1], linestyles="dashed", zorder=2)
        plt.xlabel("Predicted")
        plt.ylabel("Residual = Observed - Predicted")
        plt.tight_layout()

    if best_ystd is not None:
        plt.figure()
        for i in range(y_dim):
            plt.plot(X_test[:, 0], best_ystd[:, i], "-b")
            plt.plot(
                X_train[:, 0],
                np.interp(X_train[:, 0], X_test[:, 0], best_ystd[:, i]),
                "ok",
            )
        plt.xlabel("x")
        plt.ylabel("std(y)")