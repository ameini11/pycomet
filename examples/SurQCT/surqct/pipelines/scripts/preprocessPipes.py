# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #


NAME = 'preprocessPipes'

import argparse

from surqct import config as cfg
from surqct.src.utilities import utils
from surqct.pipelines.scripts import functions as fun


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Pipelines for SurQCT model preprocessing.",
        epilog=cfg.EPILOG
    )
    parser.add_argument('-j', '--threads',
        type=int,
        default=4,
        help='number of threads'
    )
    parser.add_argument('-d', '--dir',
        type=fun.dir_path,
        default="./",
        help='path to the working directory containing `InputPipes.py`'
    )
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
@utils.app_decorator()
def main(args):
    raise NotImplementedError