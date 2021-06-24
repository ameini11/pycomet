import os
import sys
# Remove the following line if you install `prode` as a Python package --------
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/ML/PrODE/prode/'))
# -----------------------------------------------------------------------------
import prode as pro
from prode.utilities import utils as pro_utils
from prode.utilities.training_utils import callbacks as pro_callbacks

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator


def ode(ic, t, grads, params):

    # Extracting `mass` and `T_trans`
    ic   = tf.split(ic, num_or_size_splits=inp.dim["inp"][0], axis=1)
    # mass, T  = ic[-2], ic[-1]
    mass = ic[-1]
    T    = str(10000)

    # Getting Rate Coefficient matrices
    K_e, K_d, K_r = [ tf.transpose(params[T][name]) for name in params[T] ]

    # Rearranging variables
    Y, dY_dt = [ grad * mass for grad in grads ]
    Y_O      = tf.split(Y, Y.shape[1], axis=1)[-1]

    # Computing Residuals - Master Equation
    excit_term = tf.math.multiply(tf.matmul(Y, K_e), Y_O)
    disso_term = tf.math.multiply(tf.matmul(Y, K_d), Y_O)
    recom_term = tf.matmul(tf.math.pow(Y_O, 3), K_r)

    rhs = excit_term + disso_term + recom_term

    return dY_dt - rhs


@pro_utils.main_decorator
def main():

    ###########################################################################
    ### Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen = DataGenerator(inp)
    ic      = dataGen.initial_cond()
    if inp.test_flg > 0 and inp.test_cases > 0:
        data_test = dataGen.testing_data(ic[1], ic_train=ic[0])
    if not inp.from_file:
        train_data, valid_data = dataGen.training_data(ic[0])
        train_data['data_source'] = valid_data['data_source'] = 'array'
        col_x, col_y = None, None
    else:
        train_data = inp.data_train_file
        valid_data = inp.data_valid_file
        train_data['data_source'] = valid_data['data_source'] = 'file'
        col_x, col_y = inp.col_x, inp.col_y

    # Initialize Data ---------------------------------------------------------
    data = pro_utils.get_class(pro.data, inp.data_name)(
        inp.dim,
        ode_func=ode,
        ode_order=1,
        ode_params=dataGen.ODE_mat,
        train_data=train_data,
        valid_data=valid_data,
        col_x=col_x,
        col_y=col_y,
        batch_size=inp.batch_size,
        valid_batch_size=inp.valid_batch_size)


    ###########################################################################
    ### Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    net = pro_utils.get_class(pro.networks, inp.nn_name)(
        inp.dim,
        inp.layer_size,
        inp.activation,
        inp.kernel_initializer,
        regularization=inp.regularization,
        dropout_rate=inp.dropout_rate,
        batch_norm=inp.batch_norm,
        block=inp.block
        )

    # Apply I/O Transformation ------------------------------------------------
    net.apply_input_transform(inp.input_fn)
    net.apply_output_transform(inp.output_fn)


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

        model.fit(epochs=inp.epochs, callbacks=cb_list, \
            weighted_loss_cb=inp.weighted_loss,
            early_stopping_cb=inp.early_stopping,
            model_ckpt_cb=inp.model_ckpt)

    # Save Model ---------------------------------------------------------------
    if inp.save_flg > 0:
        model.save()


    ###########################################################################
    ### Postprocessing
    ###########################################################################
    # Training phase ----------------------------------------------------------
    model.visualize_training(model.train_dir+'/history.csv', limit_y=inp.limit_y)

    # Testing phase -----------------------------------------------------------
    if inp.test_cases > 0 and inp.test_flg > 0:
        plot_style = dict(plotting=inp.plot_flg, var_names=inp.var_names, labels=inp.labels)
        for i, test_i in enumerate(data_test):
            # Define 'path'
            path = model.post_dir+'/testing_plots/test_'+str(i)
            if not os.path.exists(path):
                os.makedirs(path)
            model.predict(test_i, path=path, scales=inp.scales, **plot_style)
            if inp.log_plot:
                model.predict(test_i, path=path, scales=['log', 'log'], **plot_style)
            

if __name__ == "__main__":
    main()
