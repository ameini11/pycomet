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
from surqct.apps import functions as fun

from surqct.src.utilities import utils
from surqct.src.preprocessing import input_manipulation
from surqct.src.chemical_systems import chemical_systems as chem_sys


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
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
@utils.app_decorator()
def main(args):
    
    os.chdir(args.dir)
    
    # InputData module --------------------------------------------------------
    sys.path.append(args.dir)
    import InputData

    # Initializing chemical system --------------------------------------------
    utils.print_main(
        f"Initializing '{InputData.SystemName}' chemical system ...",
        start=''
    )
    System = chem_sys.get_system(InputData.SystemName)
    for Mol in System.Molecules:
        Mol.read_diatdata()

    # Manipulating input diatomic data ----------------------------------------
    utils.print_main(
        'Preprocessing diatomic potential input features ...'
    )
    ModifiedDiatData = input_manipulation.get_modified_diatdata(
        InputData, System
    )

    # Saving preprocessed input diatomic data ---------------------------------
    utils.print_main(
        'Saving preprocessed diatomic potential input features ...'
    )
    input_manipulation.save_modified_diatdata(
        InputData, System, ModifiedDiatData
    )
