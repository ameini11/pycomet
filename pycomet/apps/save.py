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


NAME = 'save'

import os
import argparse

from pycomet import utils
from pycomet.apps import functions as fun


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Save a recompiled machine-learning-based model.",
        epilog=utils.EPILOG
    )
    parser.add_argument('-j', '--threads',
        type=int,
        default=4,
        help='number of threads'
    )
    parser.add_argument('-d', '--dir',
        type=fun._dir_path,
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
    # Input module
    inputs = fun.import_inputs(args.dir)

    # Check flags -------------------------------------------------------------
    for k in ('train', 'test', 'plot'):
        inputs.flags[k] = 0
    inputs.flags['save'] = 1
    if inputs.flags['load'] == 0:
        utils.raise_value_err(
            "The network has to be restored from a saved or a checkpointed "
            "one. Modified the 'load' entry in the 'flags' input variable."
        )

    # # Data
    # ###########################################################################
    # data = fun._data(
    #     inputs_mod=inputs,
    #     wdir=args.dir
    # )

    # Network 
    ###########################################################################
    net = fun._network(
        inputs_mod=inputs,
        wdir=args.dir
    )

    # Model
    ###########################################################################
    model = fun._model(
        inputs_mod=inputs,
        net=net,
        data=None,
        graph=args.graph,
        verbose=args.verbose
    )
