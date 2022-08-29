# *************************************************************************** #
#                                   PyCOMET                                   #
# --------------------------------------------------------------------------- #
#                    Physics-informed machine learning for                    #
#                scientific computations and operator discovery               #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Test a machine-learning-based model
# --------------------------------------------------------------------------- #


NAME = 'test'

import os
import sys
import pycomet
import argparse
import tensorflow as tf

from tqdm import tqdm
from pycomet import utils
from pycomet.apps import functions as fun


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Test a machine-learning-based model.",
        epilog=utils.EPILOG
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
    utils.environment()

    # Add modules -------------------------------------------------------------
    sys.path.append(args.dir)
    # Input module
    utils.generate_init_mod_file(path=args.dir + '/inputs/')
    import inputs

    # Check flags -------------------------------------------------------------
    for k in ('train', 'save'):
        inputs.flags[k] = 0
    if inputs.flags['load'] not in (1, 2, 3):
        utils.raise_value_err(
            "The network has to be restored from a saved or a checkpointed "
            "one. Modified the 'load' entry in the 'flags' input variable."
        )

    # Data
    ###########################################################################
    if not inputs.from_file:
        utils.raise_value_err(
            "The test cases have to be retrieved from a list "
            "of `.csv` files. Check 'from_file' input variable."
        )
    data = fun._data(
        inputs_mod=inputs,
        wdir=args.dir
    )

    # Network 
    ###########################################################################
    net = fun._net(
        inputs_mod=inputs
    )

    # Model 
    ###########################################################################
    model = fun._model(
        inputs_mod=inputs,
        net=net,
        data=data,
        graph=args.graph,
        verbose=args.verbose
    )

    # Postprocessing
    ###########################################################################
    fun._postproc(
        inputs_mod=inputs,
        data=data,
        model=model
    )
