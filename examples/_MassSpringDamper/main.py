# *************************************************************************** #
# # Tensorflow/Keras set up

# NUM_THREADS = 16
# DTYPE       = 'float64'
# EPSILON     = 1.e-30
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# import tensorflow as tf
# tf.keras.backend.set_floatx(DTYPE)
# tf.keras.backend.set_epsilon(EPSILON)
# tf.config.threading.set_inter_op_parallelism_threads(NUM_THREADS)
# tf.config.threading.set_intra_op_parallelism_threads(NUM_THREADS)

# *************************************************************************** #

import os
import sys
# Remove the following line if you install `prode` as a Python package --------
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/PrODE/prode/'))
# -----------------------------------------------------------------------------
import prode as pro

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator


def ode(ic, t, grads, params):

    Y, dY_dt = grads

    # Getting Rate Coefficient matrices
    K = tf.transpose(params)

    return dY_dt - tf.matmul(Y, K)


@pro.utils.main_decorator
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
    data = pro.utils.get_class(pro.data, inp.data_name)(
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
    net = pro.utils.get_class(pro.net, inp.nn_name)(
        inp.dim,
        inp.layer_size,
        inp.activation,
        inp.kernel_initializer,
        regularization=inp.regularization,
        dropout_rate=inp.dropout_rate,
        batch_norm=inp.batch_norm,
        block=inp.block,
        # adaptive_weights=inp.adaptive_weights
        )

    # Apply I/O Transformation ------------------------------------------------
    net.apply_input_transform(inp.input_fn)
    net.apply_output_transform(inp.output_fn)


    ###########################################################################
    ### Model
    ###########################################################################
    # Initialize Model --------------------------------------------------------
    model = pro.__model.Model(data, net, inp.name)

    # Build Model -------------------------------------------------------------
    model.build(inp.dim['inp'], verbose=0, load_net=inp.load_flg, visualize_graph=False)

    # Compile Model -----------------------------------------------------------
    model.compile(inp.optimizer, lr=inp.lr, loss=inp.loss, \
        metrics=inp.metrics, decay=inp.decay)

    # Fit Model ---------------------------------------------------------------
    if inp.train_flg > 0:
        ES = callbacks.EarlyStopping(monitor=inp.monitor,
                                     min_delta=inp.min_delta,
                                     patience=inp.patience,
                                     restore_best_weights=True,
                                     verbose=1)
        cb_list = [ES]

        if inp.plot_flg == 2:
            PL = pro.callbacks.Plotter(data_test, inp.var_names, inp.labels,
                                       inp.scales, inp.log_plot,
                                       model.train_dir, inp.period)
            cb_list.append(PL)

        model.fit(epochs=inp.epochs, callbacks=cb_list, weighted_loss_callback=inp.adaptive_weights)

    # Save Model ---------------------------------------------------------------
    if inp.save_flg > 0:
        model.save()


    ###########################################################################
    ### Postprocessing
    ###########################################################################
    # Training phase ----------------------------------------------------------
    model.visualize_training(model.train_dir+'/history.csv')

    # Testing phase -----------------------------------------------------------
    if inp.test_cases > 0 and inp.test_flg > 0:
        plot_style = dict(plotting=inp.plot_flg, var_names=inp.var_names, labels=inp.labels)
        for i, test_i in enumerate(data_test):
            model.predict(test_i, test_case_num=i, scales=inp.scales, **plot_style)
            if inp.log_plot:
                model.predict(test_i, test_case_num=i, scales=['log', 'log'], **plot_style)
            

if __name__ == "__main__":
    main()
