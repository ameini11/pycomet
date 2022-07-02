# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #


NAME = 'preprocessVariables'

import os
import sys
import argparse

from surqct import config as cfg
from surqct.src.utilities import utils
from surqct.apps import functions as fun


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Preprocess input variables for SurQCT model.",
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
        help='path to the working directory (with `InputData.py` file)'
    )
    parser.add_argument('-v', '--verbose',
        type=int,
        default=0,
        choices=[0,1],
        help='verbose mode'
    )
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
@utils.app_decorator()
def main(args):

    # Inizialization
    ###########################################################################
    # Define arguments --------------------------------------------------------
    opts = get_opts()
    os.chdir(opts.dir)
    
    # Modules -----------------------------------------------------------------
    # InputData module
    sys.path.append(opts.dir)
    import InputData
    
    # SurQCT module
    sys.path.append(InputData.SurQCTFld)
    import surqct
    
    # InputManip module
    from surqct.src.Preprocessing import InputManip

    # Manipulating input data
    ###########################################################################
    utils.print_main('Preprocessing diatomic potential input features ...')
    diat_data = InputManip.get_diat_data(InputData)
    utils.print_main('Saving reprocessed input features ...')
    InputManip.save_diat_data(InputData, diat_data)
