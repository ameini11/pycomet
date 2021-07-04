import os
import sys
import numpy as np
import pandas as pd
# Remove the following line if you install `prode` as a Python package --------
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/ML/PrODE/prode/'))
# -----------------------------------------------------------------------------
import prode as pro
import tensorflow as tf

from prode.utilities import utils
from tensorflow.keras import callbacks

from inputs import InputData as inp
sys.path.append(inp.utils_path)
from data_generator import DataGenerator
from ode import MasterEq



@utils.main_decorator
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
    me = MasterEq(inp)

    input('_________________')

    data = utils.get_class(pro.data, inp.data_name)(
        inp.dim,
        ode_func=me.ode,
        ode_order=1,
        ode_params=me.arrhenius_coeff,
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
    net = utils.get_class(pro.networks, inp.nn_name)(
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
        if 'plotter' in inp.callbacks:
            if inp.plot_flg != 2:
                inp.callbacks.pop('plotter')
            else:
                inp.callbacks['plotter']['test_cases'] = data_test

        model.fit(epochs=inp.epochs, callbacks=inp.callbacks)

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
            path = model.post_dir+'/testing/test_'+str(i+1)
            model.predict(test_i, path=path, scales=inp.scales, **plot_style)
            if inp.log_plot:
                model.predict(test_i, path=path, scales=['log', 'log'], **plot_style)
            

if __name__ == "__main__":
    main()
