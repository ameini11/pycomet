# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Build a ML-based approximator for ODEs
# --------------------------------------------------------------------------- #


NAME = 'trainPro'

import os
import sys
import argparse
import numpy as np
import pandas as pd
import prode as pro
import tensorflow as tf

from tqdm import tqdm
from prode.utilities import utils


def dir_path(path):
    if os.path.exists(path+'/inputs'):
        return path
    else:
        raise argparse.ArgumentTypeError(
            "'inputs' folder not found in '{}'".format(os.path.normpath(path)+'/')
        )

def get_opts():
    parser = argparse.ArgumentParser(
        prog=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build a ML-based approximator for ODEs.",
        epilog=utils.app_epilog(name=NAME)
    )
    parser.add_argument('-j', '--threads',
        type=int,
        default=16,
        help='number of threads'
    )
    parser.add_argument('-d', '--dir',
        type=dir_path,
        default="./",
        help='path to the working directory (with `inputs` folder)'
    )
    parser.add_argument('-v', '--verbose',
        type=int,
        default=0,
        choices=[0,1],
        help='verbose mode'
    )
    parser.add_argument('-g', '--graph',
        type=int,
        default=0,
        choices=[0,1],
        help='visualize model graph'
    )
    return parser.parse_args()


@utils.app_decorator(name=NAME)
def main():

    # Inizialization
    ###########################################################################
    # Define arguments --------------------------------------------------------
    opts = get_opts()

    # Configure TensorFlow ----------------------------------------------------
    tf.config.threading.set_inter_op_parallelism_threads(opts.threads)
    tf.config.threading.set_intra_op_parallelism_threads(opts.threads)
    utils.environment()

    # Input files -------------------------------------------------------------
    sys.path.append(opts.dir)
    import inputs as inp

    # Add paths
    for path in inp.data.paths.values():
        sys.path.append(path)

    # Add classes for generating data
    from data_generator import DataGenerator
    from ode import MasterEq

    # Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen_args = utils.get_args_values(
        args=utils.get_args(DataGenerator.__init__),
        module=inp.data
    )
    dataGen = DataGenerator(
        dim=inp.network.dim,
        database=inp.data.paths['database'],
        **dataGen_args
    )

    test_data = None
    if not inp.data.from_file:
        ic = dataGen.initial_cond()
        train_data, valid_data = dataGen.training_data(ic[0])
        if inp.model.flags['test'] > 0 and inp.data.test_cases > 0:
            test_data = dataGen.testing_data(ic[1], ic_train=ic[0])
    else:
        train_data, valid_data = inp.data.train_file, inp.data.valid_file
        if inp.model.flags['test'] > 0 and inp.data.test_cases > 0:
            test_data = dataGen.testing_data(
                inp.data.test_file,
                header=inp.data.header
            )

    # Initialize Data Class ---------------------------------------------------
    me = MasterEq(
        dim=inp.network.dim,
        database=inp.data.paths['database'],
        grouping=inp.data.grouping,
        processes=inp.data.processes
    )

    data_cls = utils.get_class(pro.data, inp.data.data_type)
    data_args = utils.get_args_values(
        args=utils.get_args(data_cls.__init__),
        module=inp.data
    )
    data = data_cls(
        dim=inp.network.dim,
        training=inp.model.flags['train'],
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
    net_cls = utils.get_class(pro.networks, inp.network.architecture)
    net_args = utils.get_args_values(
        args=utils.get_args(net_cls.__init__),
        module=inp.network
    )
    net = net_cls(**net_args)

    # Apply I/O Transformation ------------------------------------------------
    if hasattr(inp.network, 'input_fn'):
        net.inp_trans = inp.network.input_fn
    if hasattr(inp.network, 'output_fn'):
        net.out_trans = inp.network.output_fn

    # Model
    ###########################################################################
    # Initialize Model --------------------------------------------------------
    model = pro.model.Model(data, net, inp.model.name)

    # Build Model -------------------------------------------------------------
    model.build(
        input_dim=inp.network.dim['inp'],
        load_net=inp.model.flags['load'],
        visualize_graph=opts.graph,
        verbose=opts.verbose
    )

    # Compile Model -----------------------------------------------------------
    compile_args = utils.get_args_values(
        args=utils.get_args(model.compile),
        module=inp.model
    )
    model.compile(**compile_args)

    # Fit Model ---------------------------------------------------------------
    if inp.model.flags['train'] > 0:
        if hasattr(inp.model, 'callbacks'):
            if 'plotter' in inp.model.callbacks \
                and inp.model.flags['plot'] != 2:
                inp.model.callbacks.pop('plotter')
        else:
            inp.model.callbacks = None

        model.fit(
            epochs=inp.model.epochs,
            callbacks=inp.model.callbacks if hasattr(
                inp.model, 'callbacks'
            ) else None
        )
    
    # Save Model --------------------------------------------------------------
    if inp.model.flags['save'] > 0:
        model.save()

    # Postprocessing
    ###########################################################################
    # Training phase ----------------------------------------------------------
    model.visualize_training(
        history=model.train_dir+'/history.csv',
        limit_y=inp.postprocessing.limit_y \
            if hasattr(inp.postprocessing, 'limit_y') else False
    )

    # Testing phase -----------------------------------------------------------
    utils.print_main("Postprocessing physical quantities ...")
    if inp.data.test_cases > 0 and inp.model.flags['test'] > 0:
        for i, test_i in enumerate(tqdm(data.test)):
            model.predict(
                data=test_i,
                plotting=inp.model.flags['plot'],
                path=model.post_dir+'/testing/test_'+str(i+1),
                **inp.postprocessing.plot_style
            )
