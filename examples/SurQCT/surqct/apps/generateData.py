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
import pandas as pd

from surqct import config as cfg
from surqct.src.utilities import utils
from surqct.apps import functions as fun


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
    parser.add_argument('-v', '--verbose',
        type=int,
        default=0,
        choices=[0,1],
        help='verbose mode'
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

    # Inizialization
    ###########################################################################
    # Define arguments --------------------------------------------------------
    args = get_opts()
    os.chdir(args.dir)
    
    # Modules -----------------------------------------------------------------
    # InputData module
    sys.path.append(args.dir)
    import InputData
    
    # SurQCT module
    sys.path.append(InputData.SurQCTFld)
    import surqct
    
    # RatesType module
    RatesType = __import__(
        f'surqct.src.RatesType.{InputData.NbAtoms}Atoms.{InputData.RatesType}',
        globals(),
        locals(),
        [InputData.RatesType],
        0
    )

    # Generating data
    ###########################################################################
    # Training/Validation -----------------------------------------------------
    if args.training:
        for path in (args.dir+'/data/train/', args.dir+'/data/valid/'):
            if not os.path.exists(path):
                os.makedirs(path)
        # Generating
        TrainData, ValidData, AllData = RatesType.generate_trainingdata(InputData)
        # Saving
        TrainData = pd.concat(TrainData, axis=1)
        TrainData.to_csv(args.dir+'/data/train/pts.csv', index=False, float_format='%.10e')

        print('\n[SurQCT]:    Number of training pts:   %10d' % len(TrainData))
        ValidData = pd.concat(ValidData, axis=1)
        ValidData.to_csv(args.dir+'/data/valid/pts.csv', index=False, float_format='%.10e')
        print('[SurQCT]:    Number of validation pts: %10d' % len(ValidData))
    
    # Testing -----------------------------------------------------------------
    if args.testing:
        utils.print_main('Generating testing data ...')
        for path in (args.dir+'/data/test/',):
            if not os.path.exists(path):
                os.makedirs(path)
        # Generating
        TestData = RatesType.generate_testingdata(InputData, Labeled=True)
        # Saving
        TestData = pd.concat(TestData, axis=1)
        TestData.to_csv(args.dir+'/data/test/labeled.csv', index=False, float_format='%.10e')
        print('\n[SurQCT]:    Number of labeled testing pts:   %10d' % len(TestData))
        
        # Generating
        TestData = RatesType.generate_testingdata(InputData, Labeled=False)
        # Saving
        TestData = pd.concat(TestData, axis=1)
        TestData.to_csv(args.dir+'/data/test/unlabeled.csv', index=False, float_format='%.10e')
        print('[SurQCT]:    Number of unlabeled testing pts: %10d' % len(TestData))
