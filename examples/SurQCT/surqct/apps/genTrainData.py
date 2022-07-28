# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #


NAME = 'genTrainData'

import os
import sys
import argparse
import numpy as np
import pandas as pd

from surqct import config as cfg
from surqct.apps import functions as fun

from surqct.src.utilities import utils
from surqct.src.chemical_systems import systems as syst
from surqct.src.rates_type.rates_type import RatesType


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Generate training/validation/testing data to construct the SurQCT model.",
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
        help='path to the working directory containing `InputData.py`'
    )
    parser.add_argument('-sd', '--savedir',
        type=os.path.abspath,
        help='path to the saving directory'
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
def _main(args=None, InputData=None):

    # Initializing chemical system --------------------------------------------
    utils.print_main(
        f"Initializing '{InputData.SystemName}' chemical system and rates type ...",
        start=''
    )
    system = syst.get_system(
        NameLong=InputData.SystemName,
        ProcNames=InputData.RatesType,
        PathToHDF5Fld=InputData.PathToHDF5Fld
    )
    rates_type = RatesType(system, InputData)

    # Generating data
    ###########################################################################
    if args.savedir is None:
        args.savedir = args.dir
        
    # Training/Validation -----------------------------------------------------
    if args.training:
        for path in (args.savedir+'/data/train/', args.savedir+'/data/valid/'):
            if not os.path.exists(path):
                os.makedirs(path)

        TrainData, ValidData, AllData = rates_type.generate_trainingdata()

        utils.print_main('Saving training and validation data points ...')
        TrainData = pd.concat(TrainData, axis=1)
        TrainData.to_csv(args.dir+'/data/train/pts.csv', index=False, float_format='%.10e')
        utils.print_submain('Number of training pts:   %10d' % len(TrainData))

        ValidData = pd.concat(ValidData, axis=1)
        ValidData.to_csv(args.dir+'/data/valid/pts.csv', index=False, float_format='%.10e')
        utils.print_submain('Number of validation pts: %10d' % len(ValidData))
    
    # Testing -----------------------------------------------------------------
    if args.testing:
        path = args.savedir+'/data/test/'
        if not os.path.exists(path):
            os.makedirs(path)

        # Generating
        TestData = rates_type.generate_testingdata()

        utils.print_main('Saving labeled testing data points ...')
        TestData = pd.concat(TestData, axis=1)
        TestData.to_csv(path+'/labeled.csv', index=False, float_format='%.10e')
        utils.print_submain('Number of labeled testing pts:   %10d' % len(TestData))


@utils.app_decorator()
def main(args):
    
    os.chdir(args.dir)
    
    # InputData module
    sys.path.append(args.dir)
    import InputData

    _main(args, InputData)
