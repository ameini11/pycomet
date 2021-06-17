import os
import sys
import numpy as np
import pandas as pd
# Remove the following line if you install `prode` as a Python package --------
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/ML/PrODE/prode/'))
# -----------------------------------------------------------------------------
import prode as pro
import tensorflow as tf
tf.keras.backend.set_epsilon(1.e-30)

from prode.utilities.training_utils import callbacks as pro_callbacks
from prode.utilities.model_utils import layers as pro_layers
from prode.utilities import utils as pro_utils
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from matplotlib import pyplot as plt

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator


@pro_utils.main_decorator
def main():

    ###########################################################################
    ### Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen = DataGenerator(inp)
    x, y = dataGen.training_data()

    # Initialize Data ---------------------------------------------------------
    data = pro_utils.get_class(pro.data, inp.data_name)(
        dataGen.dim,
        x_train=x,
        y_train=y)

    ###########################################################################
    ### Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    net = pro_utils.get_class(pro.networks, inp.nn_name)(
        dataGen.dim,
        inp.layer_size,
        inp.activation,
        inp.kernel_initializer,
        regularization=inp.regularization,
        dropout_rate=inp.dropout_rate,
        batch_norm=inp.batch_norm,
        block=inp.block)

    # Apply I/O Transformation ------------------------------------------------
    mn, mx = min(inp.T), max(inp.T)
    net.apply_input_transform(pro_layers.Scaling(init_scale=1./(mx-mn), \
        init_offset=-mn/(mx-mn), name='input_fn', trainable=False))
    net.apply_output_transform(layers.Lambda(tf.math.exp, name='output_fn'))


    ###########################################################################
    ### Model
    ###########################################################################
    # Initialize Model --------------------------------------------------------
    model = pro.model.Model(data, net, inp.name)

    # Build Model -------------------------------------------------------------
    model.build(inp.dim['inp'], verbose=0, load_net=inp.load_flg, visualize_graph=False)

    # Compile Model -----------------------------------------------------------
    model.compile(inp.optimizer, lr=inp.lr, loss=inp.loss, \
        metrics=inp.metrics, decay=inp.decay)

    # Fit Model ---------------------------------------------------------------
    if inp.train_flg > 0:
        cb_list = []
        if inp.plot_flg == 2:
            PL = pro_callbacks.Plotter(data_test, inp.var_names, inp.labels,
                                       inp.scales, inp.log_plot,
                                       model.train_dir, inp.period)
            cb_list.append(PL)

        model.fit(epochs=inp.epochs, callbacks=cb_list)

    # Save Model ---------------------------------------------------------------
    if inp.save_flg > 0:
        model.save()


    ###########################################################################
    ### Postprocessing
    ###########################################################################
    # Training phase ----------------------------------------------------------
    model.visualize_training(model.train_dir+'/history.csv')

    # Testing phase -----------------------------------------------------------
    if inp.test_flg > 0:
        x_pred = np.linspace(min(inp.T), max(inp.T), 1000, dtype=np.float64)
        y_pred = net.predict(x_pred)

        path = model.post_dir+'/testing_plots/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Dissociation
        y_true_diss = y[:,0:dataGen.num_diss]
        y_pred_diss = y_pred[:,0:dataGen.num_diss]
        fig_name    = path + 'dissociation.pdf'
        plot_test(
            fig_name,
            x,
            x_pred,
            y_true_diss,
            y_pred_diss,
            inp.var_names_diss,
            'Dissociation Rates',
            labels=inp.labels,
            scales=inp.scales)

        # Excitation
        y_true_exci = y[:,dataGen.num_diss:]
        y_pred_exci = y_pred[:,dataGen.num_diss:]
        fig_name    = path + 'excitation.pdf'
        plot_test(
            fig_name,
            x,
            x_pred,
            y_true_exci,
            y_pred_exci,
            inp.var_names_exci,
            'Exchange + Inelastic Rates',
            labels=inp.labels,
            scales=inp.scales)


def plot_test(
    fig_name,
    x_true,
    x_pred,
    y_true,
    y_pred,
    var_name,
    title,
    labels=[None, None],
    scales=['log', 'linear']):
    """Variable plotting."""

    fig = plt.figure()
    plt.grid()
    plt.title(title)

    x_label, y_label = labels
    x_scale, y_scale = scales

    # X axis
    if x_label is not None: plt.xlabel(x_label)
    if x_scale is not None: plt.xscale(x_scale)

    # Y axis
    if y_label is not None: plt.ylabel(y_label)
    if y_scale == 'log':
        plt.yscale(y_scale)
        plt.ylim([np.amin(y_true[y_true!=0.])*1.e-1, np.amax(y_true)*1.e+1])
    else:
        delta = np.amax(y_true)*0.1
        plt.ylim([np.amin(y_true)-delta, np.amax(y_true)+delta])

    # Plotting
    lin = ['-', '--']
    col = ['k', 'r', 'g', 'b', 'tab:purple', 'tab:brown', 'tab:pink', \
           'tab:blue', 'tab:red', 'tab:green', 'tab:orange']

    # Solution
    for d in range(y_true.shape[1]):

        name = 'True ' + var_name[d]
        marker_style = dict(c=col[d], marker='x', lw=0.5, \
                            linestyle='none', fillstyle='none', markersize=5)
        plt.plot(x_true, y_true[:,d], **marker_style, label=name)

        name = 'Pred ' + var_name[d]
        plt.plot(x_pred, y_pred[:,d], c=col[d], ls=lin[1], lw=1, label=name)


    plt.legend(fontsize='x-small')
    fig.savefig(fig_name)
    plt.close()

            

if __name__ == "__main__":
    main()
