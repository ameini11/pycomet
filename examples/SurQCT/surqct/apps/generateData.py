# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #


NAME = 'generateData'

import os
import sys
import argparse
import numpy as np
import pandas as pd

from surqct import config as cfg
from surqct.apps import functions as fun

from surqct.src.utilities import utils
from surqct.src.rates_type.rates_type import get_rates_type_mod
from surqct.src.chemical_systems import chemical_systems as chem_sys


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate data for SurQCT model.",
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
    parser.add_argument('-tr', '--training',
        type=int,
        default=1,
        choices=[0,1],
        help='generate training data'
    )
    parser.add_argument('-ts', '--testing',
        type=int,
        default=0,
        choices=[0,1],
        help='generate testing data'
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
    # Initializing thermochemical processes
    System.ThermochemicalProc.ProcBool = System.ThermochemicalProc.ProcNames == InputData.RatesType
    System.ThermochemicalProc.PathToHDF5File = InputData.PathToHDF5File
    System.ThermochemicalProc.init_processes()
    # Initializing molecules
    for Mol in System.Molecules:
        Mol.read_diatdata()

    # Initializing rates type module
    rates_type_mod = get_rates_type_mod(System)

    # Generating data
    ###########################################################################
    # Training/Validation -----------------------------------------------------
    if args.training:
        for path in (args.dir+'/data/train/', args.dir+'/data/valid/'):
            if not os.path.exists(path):
                os.makedirs(path)

        TrainData, ValidData, AllData = rates_type_mod.generate_trainingdata(InputData, System)

        utils.print_main('Saving training and validation data points ...')
        TrainData = pd.concat(TrainData, axis=1)
        TrainData.to_csv(args.dir+'/data/train/pts.csv', index=False, float_format='%.10e')
        utils.print_submain('Number of training pts:   %10d' % len(TrainData))

        ValidData = pd.concat(ValidData, axis=1)
        ValidData.to_csv(args.dir+'/data/valid/pts.csv', index=False, float_format='%.10e')
        utils.print_submain('Number of validation pts: %10d' % len(ValidData))
    
    # Testing -----------------------------------------------------------------
    if args.testing:
        for path in (args.dir+'/data/test/',):
            if not os.path.exists(path):
                os.makedirs(path)

        # Generating
        TestData = rates_type_mod.generate_testingdata(InputData, System)
        # Saving
        TestData = pd.concat(TestData, axis=1)
        TestData.to_csv(args.dir+'/data/test/labeled.csv', index=False, float_format='%.10e')
        utils.print_submain('Number of labeled testing pts:   %10d' % len(TestData))
