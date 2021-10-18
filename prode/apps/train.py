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

    # Add modules -------------------------------------------------------------
    sys.path.append(opts.dir)
    # Input module
    import inputs as inp
    # Physics-related module
    try:
        from physics.physical_model import PhysicalModel
        physics_informed = True
    except ModuleNotFoundError:
        physics_informed = False
    from physics.data_generator import DataGenerator

    # Add additional paths
    if hasattr(inp.general, 'paths') and inp.general.paths is not None:
        paths = list(inp.general.paths.values()) \
            if isinstance(inp.general.paths, dict) else inp.general.paths
        for path in paths:
            sys.path.append(path)

    # Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    dataGen_args = utils.get_args_values(
        args=utils.get_args(DataGenerator.__init__),
        modules=[inp.data, inp.general]
    )
    dataGen = DataGenerator(
        **dataGen_args
    )
    dataGen.data_dir = opts.dir + "/data/"
    if not os.path.exists(dataGen.data_dir):
        os.makedirs(dataGen.data_dir)

    if not inp.data.from_file:
        utils.print_main("Generating initial conditions ...")
        ic = dataGen.initial_cond()
        utils.print_main("Generating training/validation pts ...")
        train_data, valid_data = dataGen.training_data(ic[0])
        if inp.general.flags['test'] > 0 and inp.data.test_cases > 0:
            utils.print_main('Generating testing pts ...')
            test_data = dataGen.testing_data(ic[1], ic_train=ic[0])
    else:
        utils.print_main("Extracting training/validation pts from file ...")
        train_data, valid_data = inp.data.train_file, inp.data.valid_file
        if inp.general.flags['test'] > 0 and inp.data.test_cases > 0:
            utils.print_main('Generating testing pts ...')
            test_data = dataGen.testing_data(
                inp.data.test_file,
                header=inp.data.header
            )

    # Initialize Data Class ---------------------------------------------------
    data_cls = utils.get_class(pro.data, inp.data.data_type)
    utils.print_main(
        "Initializing '{}' data ...".format(inp.data.data_type)
    )
    data_args = utils.get_args_values(
        args=utils.get_args(data_cls.__init__),
        modules=[inp.data, inp.general]
    )

    if physics_informed and inp.data.data_type == 'ODE':
        PhysicalModel_args = utils.get_args_values(
            args=utils.get_args(PhysicalModel.__init__),
            modules=[inp.data, inp.general]
        )
        physicalMdl = PhysicalModel(
            **PhysicalModel_args
        )
        data_args.update(dict(ode_func=physicalMdl.ode))

    data = data_cls(
        training=inp.general.flags['train'],
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data or None,
        **data_args
    )

    # Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    net_cls = utils.get_class(pro.networks, inp.network.architecture)
    utils.print_main(
        "Initializing '{}' network ...".format(inp.network.architecture)
    )
    net_args = utils.get_args_values(
        args=utils.get_args(net_cls.__init__),
        modules=[inp.network, inp.general]
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
    model = pro.model.Model(data, net, inp.general.model_name)

    # Build Model -------------------------------------------------------------
    utils.print_main(
        "Building '{}' network ...".format(inp.network.architecture)
    )
    model.build(
        input_dim=inp.general.dim['inp'],
        load_net=inp.general.flags['load'],
        visualize_graph=opts.graph,
        verbose=opts.verbose
    )

    # Compile Model -----------------------------------------------------------
    utils.print_main(
        "Compiling '{}' network ...".format(inp.network.architecture)
    )
    compile_args = utils.get_args_values(
        args=utils.get_args(model.compile),
        modules=inp.engine
    )
    model.compile(**compile_args)

    # Fit Model ---------------------------------------------------------------
    if inp.general.flags['train'] > 0:
        if hasattr(inp.engine, 'callbacks'):
            if 'plotter' in inp.engine.callbacks \
                and inp.general.flags['plot'] != 2:
                inp.engine.callbacks.pop('plotter')
        else:
            inp.engine.callbacks = None

        model.fit(
            epochs=inp.engine.epochs,
            callbacks=inp.engine.callbacks if hasattr(
                inp.engine, 'callbacks'
            ) else None
        )
    
    # Save Model --------------------------------------------------------------
    if inp.general.flags['save'] > 0:
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
    if inp.data.test_cases > 0 and inp.general.flags['test'] > 0:
        for i, test_i in enumerate(tqdm(data.test)):
            model.predict(
                data=test_i,
                plotting=inp.general.flags['plot'],
                path=model.post_dir+'/testing/test_'+str(i+1),
                **inp.postprocessing.plot_style
            )
