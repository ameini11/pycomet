# *************************************************************************** #
#                                   PyCOMET                                   #
# --------------------------------------------------------------------------- #
#                    Physics-informed machine learning for                    #
#                scientific computations and operator discovery               #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Build a ML-based approximator
# --------------------------------------------------------------------------- #


NAME = 'train'

import os
import sys
import pycomet
import argparse
import importlib
import tensorflow as tf

from tqdm import tqdm
from pycomet import config as cfg
from pycomet.utilities import utils
from pycomet.apps import functions as fun


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Build a ML-based approximator for ODEs.",
        epilog=cfg.EPILOG
    )
    parser.add_argument('-j', '--threads',
        type=int,
        default=16,
        help='number of threads'
    )
    parser.add_argument('-d', '--dir',
        type=fun.dir_path,
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
    parser.add_argument('-s', '--seed',
        type=int,
        default=None,
        help='set random seed'
    )
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
@utils.app_decorator()
def main(args):

    # Inizialization
    ###########################################################################
    if args.graph:
        args.verbose = 1
    os.chdir(args.dir)
    # Environment
    utils.environment()

    # Add modules -------------------------------------------------------------
    sys.path.append(args.dir)
    # Input module
    import inputs
    from inputs.general import flags

    # Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    if inputs.data.from_file:
        utils.print_main("Extracting data points from file ...")
        train_data = inputs.data.file['train']
        valid_data = inputs.data.file['valid']
        test_data = None
        if flags['test'] > 0 and inputs.data.test_cases > 0:
            test_data = inputs.data.file['test']
    else:
        utils.print_main("Generating initial conditions ...")
        # Physics-related module
        from physics.data_generator import DataGenerator

        dataGen_args = utils.get_args_values(
            args=utils.get_args(DataGenerator.__init__),
            modules=[inputs.data, inputs.general]
        )
        dataGen = DataGenerator(
            **dataGen_args
        )
        dataGen.data_dir = args.dir + "/data/"
        if not os.path.exists(dataGen.data_dir):
            os.makedirs(dataGen.data_dir)

        ic = dataGen.initial_cond()
        utils.print_main("Generating training/validation pts ...")
        train_data, valid_data = dataGen.training_data(ic[0])
        test_data = None
        if flags['test'] > 0 and inputs.data.test_cases > 0:
            utils.print_main('Generating testing pts ...')
            test_data = dataGen.testing_data(ic[1], ic_train=ic[0])

    # Data Class --------------------------------------------------------------
    data = fun._data(
        flags=flags,
        input_mod=inputs,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        wdir=args.dir
    )

    # Network 
    ###########################################################################
    net = fun._net(
        input_mod=inputs
    )

    # Model 
    ###########################################################################
    model = fun._model(
        flags=flags,
        input_mod=inputs,
        net=net,
        data=data,
        graph=args.graph,
        verbose=args.verbose
    )

    # Postprocessing
    ###########################################################################
    # Training phase ----------------------------------------------------------
    model.visualize_training(history=model.dir['train']+'/history.csv')

    # Testing phase -----------------------------------------------------------
    if flags['test'] > 0 and data.test is not None:
        utils.print_main("Postprocessing physical quantities ...")
        for name, d in tqdm(data.test.items()):
            model.predict(
                data=d,
                plotting=flags['plot'],
                path=model.dir['post'] + f'/testing/{name}',
                **inputs.postproc.plot_style
            )
