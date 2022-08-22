# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #


NAME = 'postprocMEData'

import os
import sys
import argparse

from surqct import config as cfg
from surqct.apps import functions as fun

from surqct.src.utilities import utils
from surqct.src.chemical_systems import systems as syst


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Postprocess MCoarseAIR data and 0D master equation solution.",
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
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
def _main(args=None, InputData=None):

    utils.print_main(
        f"Initializing '{InputData.SystemName}' chemical system ...", start=''
    )
    system = syst.get_system(
        NameLong=InputData.SystemName,
        KinSource=InputData.KinSource
    )

    utils.print_main(
        "Postprocessing master equation solution ..."
    )
    system.MasterEquation.postprocess_data(CaseSuffix=InputData.SuffixName)

    utils.print_main(
        "Saving postprocessed master equation data in "
        f"'{os.path.abspath(system.MasterEquation.MEPath)}/MEData.hdf5' ..."
    )
    system.MasterEquation.save_data()


@utils.app_decorator()
def main(args):
    
    os.chdir(args.dir)
    
    # InputData module
    sys.path.append(args.dir)
    import InputData

    _main(args, InputData)
