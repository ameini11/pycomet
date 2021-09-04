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

# Input files
import inputs.data as inp_data
import inputs.network as inp_net
import inputs.model as inp_mdl
import inputs.postprocessing as inp_post

sys.path.append(inp_data.paths['utils'])
from data_generator import DataGenerator
from ode import MasterEq


@utils.main_decorator
def main():

    ###########################################################################
    ### Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen = DataGenerator(
        dim=inp_net.dim,
        train_pts=inp_data.train_pts,
        valid=inp_data.valid,
        train_pts_dist=inp_data.train_pts_dist,
        test_cases=inp_data.test_cases,
        test_training=inp_data.test_training,
        t_limit=inp_data.t_limit,
        inp_space=inp_data.inp_space,
        database=inp_data.paths['database'],
        grouping=inp_data.grouping,
        processes=inp_data.processes
    )

    if not inp_data.from_file:
        ic = dataGen.initial_cond()
        train_data, valid_data = dataGen.training_data(ic[0])
        if inp_mdl.test_flg > 0 and inp_data.test_cases > 0:
            test_data = dataGen.testing_data(ic[1], ic_train=ic[0])
    else:
        train_data, valid_data = inp_data.train_file, inp_data.valid_file
        if inp.test_flg > 0 and inp.test_cases > 0:
            test_data = dataGen.testing_data(inp_data.test_file, header=header)

    # Initialize Data ---------------------------------------------------------
    me = MasterEq(
        ic_dim=inp_net.dim,
        database=inp_data.paths['database'],
        grouping=inp_data.grouping,
        processes=inp_data.processes
    )

    input('=====================================')
    data = utils.get_class(pro.data, inp_data.data_type)(
        inp_net.dim,
        training=inp_mdl.train_flg,
        ode_func=me.ode,
        ode_order=1,
        ode_params=me.arrhenius_coeff,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        from_file=inp_data.from_file,
        col_x=inp_data.col_x,
        col_y=inp_data.col_y,
        header=inp_data.header,
        preproc_fn=inp_data.preproc_fn,
        batch_size=inp_data.batch_size,
        valid_batch_size=inp_data.valid_batch_size
    )

    input('=====================================')


    ###########################################################################
    ### Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    net = utils.get_class(pro.networks, inp_net.nn_name)(
        inp_net.output_names,
        inp_net.dim,
        inp_net.layer_size,
        inp_net.activation,
        inp_net.kernel_initializer,
        inp_opnn_trans=inp_net.inp_opnn_trans,
        out_opnn_trans=inp_net.out_opnn_trans,
        regularization=inp_net.regularization,
        dropout_rate=inp_net.dropout_rate,
        batch_norm=inp_net.batch_norm,
        block=inp_net.block
    )

    # Apply I/O Transformation ------------------------------------------------
    net.apply_input_transform(inp_net.input_fn)
    net.apply_output_transform(inp_net.output_fn)


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
