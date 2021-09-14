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

# Add paths
for path in inp_data.paths.values():
    sys.path.append(path)

# Add classes for generating data
from data_generator import DataGenerator
from ode import MasterEq


@utils.main_decorator
def main():

    # Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen_args = utils.get_args_values(
        args=utils.get_args(DataGenerator.__init__),
        module=inp_data
    )
    dataGen = DataGenerator(
        dim=inp_net.dim,
        database=inp_data.paths['database'],
        **dataGen_args
    )

    if not inp_data.from_file:
        ic = dataGen.initial_cond()
        train_data, valid_data = dataGen.training_data(ic[0])
        if inp_mdl.flags['test'] > 0 and inp_data.test_cases > 0:
            test_data = dataGen.testing_data(ic[1], ic_train=ic[0])
    else:
        train_data, valid_data = inp_data.train_file, inp_data.valid_file
        if inp_mdl.flags['test'] > 0 and inp_data.test_cases > 0:
            test_data = dataGen.testing_data(
                inp_data.test_file,
                header=inp_data.header
            )

    # Initialize Data ---------------------------------------------------------
    me = MasterEq(
        dim=inp_net.dim,
        database=inp_data.paths['database'],
        grouping=inp_data.grouping,
        processes=inp_data.processes
    )

    data_cls = utils.get_class(pro.data, inp_data.data_type)
    data_args = utils.get_args_values(
        args=utils.get_args(data_cls.__init__),
        module=inp_data
    )
    data = data_cls(
        dim=inp_net.dim,
        training=inp_mdl.flags['train'],
        ode_func=me.ode,
        ode_params=me.arrhenius_coeff,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        **data_args
    )


    # Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    net_cls = utils.get_class(pro.networks, inp_net.architecture)
    net_args = utils.get_args_values(
        args=utils.get_args(net_cls.__init__),
        module=inp_net
    )
    net = net_cls(**net_args)

    # Apply I/O Transformation ------------------------------------------------
    if hasattr(inp_net, 'input_fn'):
        net.inp_trans = inp_net.input_fn
        # net.apply_input_transform(inp_net.input_fn)
    if hasattr(inp_net, 'output_fn'):
        net.out_trans = inp_net.output_fn
        # net.apply_output_transform(inp_net.output_fn)


    # Model
    ###########################################################################
    # Initialize Model --------------------------------------------------------
    model = pro.model.Model(data, net, inp_mdl.name)

    # Build Model -------------------------------------------------------------
    model.build(
        input_dim=inp_net.dim['inp'],
        verbose=inp_mdl.verbose,
        load_net=inp_mdl.flags['load'],
        visualize_graph=inp_mdl.visualize_graph,
    )

    # Compile Model -----------------------------------------------------------
    compile_args = utils.get_args_values(
        args=utils.get_args(model.compile),
        module=inp_mdl
    )
    model.compile(**compile_args)

    # Fit Model ---------------------------------------------------------------
    if inp_mdl.flags['train'] > 0:
        if hasattr(inp_mdl, 'callbacks'):
            if 'plotter' in inp_mdl.callbacks and inp_mdl.flags['plot'] != 2:
                inp_mdl.callbacks.pop('plotter')
        else:
            inp_mdl.callbacks = None

        model.fit(
            epochs=inp_mdl.epochs,
            callbacks=inp_mdl.callbacks if hasattr(inp_mdl, 'callbacks') else None
        )
    
    # Save Model --------------------------------------------------------------
    if inp_mdl.flags['save'] > 0:
        model.save()


    # Postprocessing
    ###########################################################################
    # Training phase ----------------------------------------------------------
    model.visualize_training(
        history=model.train_dir+'/history.csv',
        limit_y=inp_post.limit_y if hasattr(inp_post, 'limit_y') else False
    )

    # Testing phase -----------------------------------------------------------
    utils.print_main("Postprocessing physical quantities ...")
    if inp_data.test_cases > 0 and inp_mdl.flags['test'] > 0:
        for i, test_i in enumerate(data.test):
            model.predict(
                data=test_i,
                plotting=inp_mdl.flags['plot'],
                path=model.post_dir+'/testing/test_'+str(i+1),
                **inp_post.plot_style
            )
            

if __name__ == "__main__":
    main()
