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
        type=str,
        default='',
        help='path to the working directory containing the `inputs` folder'
    )
    parser.add_argument('-s', '--system',
        type=str,
        default='',
        help='name of the chemical system under investigation (e.g., O3_UMN)'
    )
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
@utils.app_decorator()
def main(args):
    raise NotImplementedError