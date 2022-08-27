# *************************************************************************** #
#                                   PyCOMET                                   #
# --------------------------------------------------------------------------- #
#                    Physics-informed machine learning for                    #
#                scientific computations and operator discovery               #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Test a ML-based approximator
# --------------------------------------------------------------------------- #


NAME = 'test'

import os
import sys
import pycomet
import argparse
import tensorflow as tf

from tqdm import tqdm
from pycomet mport utils
from pycomet import config as cfg
from pycomet.apps import functions as fun


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Test a ML-based approximator.",
        epilog=utils._EPILOG
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
@utils.app_dec()
def main(args):

    # Inizialization
    ###########################################################################
    if args.graph:
        args.verbose = 1
    os.chdir(args.dir)
    # Environment
    utils._environment()

    # Add modules -------------------------------------------------------------
    sys.path.append(args.dir)
    # Input module
    import inputs
    from inputs.general import flags

    # Check flags -------------------------------------------------------------
    for k in ('train', 'save'):
        flags[k] = 0
    if flags['load'] not in (1, 2):
        utils._raise_value_err(
            "The network has to be restored from a saved or a checkpointed "
            "one. Modified the 'load' entry in the `flags` input variable."
        )

    # Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    if inputs.data.from_file:
        utils.print_main("Extracting testing pts from file ...")
        test_data = inputs.data.file['test']
    else:
        raise NotImplementedError
        # utils.print_main("Generating testing pts ...")
        # # Physics-related module
        # from physics.data_generator import DataGenerator

        # dataGen_args = utils.get_args_values(
        #     args=utils.get_args(DataGenerator.__init__),
        #     modules=[inputs.data, inputs.general]
        # )
        # dataGen = DataGenerator(
        #     **dataGen_args
        # )
        # dataGen.data_dir = args.dir + "/data/"
        # if not os.path.exists(dataGen.data_dir):
        #     os.makedirs(dataGen.data_dir)

        # ic = dataGen.initial_cond()
        # utils.print_main("Generating training/validation pts ...")
        # train_data, valid_data = dataGen.training_data(ic[0])
        # if flags['test'] > 0 and inputs.data.test_cases > 0:
        #     utils.print_main('Generating testing pts ...')
        #     test_data = dataGen.testing_data(ic[1], ic_train=ic[0])

    # Data Class --------------------------------------------------------------
    data = fun._data(
        flags=flags,
        input_mod=inputs,
        test_data=test_data or None,
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
    if flags['test'] > 0:
        utils.print_main("Postprocessing physical quantities ...")
        for name, d in tqdm(data.test.items()):
            model.predict(
                data=d,
                plotting=flags['plot'],
                path=model.dir['post'] + f'/testing/{name}',
                **inputs.postproc.plot_style
            )
