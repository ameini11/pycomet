# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #


NAME = 'writeToDtb'

import os
import sys
import argparse
import numpy as np
import pandas as pd

from pathlib import Path
from surqct import config as cfg
from surqct.apps import functions as surqct_fun
from pycomet.apps import functions as pycomet_fun

from surqct.src.utilities import utils
from surqct.src.chemical_systems import systems as syst
from surqct.src.rates_type.rates_type import get_rates_type_mod


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Write kinetics and thermo files to Plato database (see 'config.MERuns').",
        epilog=cfg.EPILOG
    )
    parser.add_argument('-j', '--threads',
        type=int,
        default=4,
        help='number of threads'
    )
    parser.add_argument('-d', '--dir',
        type=pycomet_fun.dir_path,
        default="./",
        help='path to the working directory containing `InputData.py`'
    )
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
def _main(args=None, InputData=None):

    utils.print_main(
        f"Initializing '{InputData.SystemName}' chemical system ..."
    )
    System = syst.get_system(
        NameLong=InputData.SystemName,
        ProcNames=InputData.RatesType,
        KinSource=InputData.KinSource,
        PathToHDF5File=cfg.HDF5Dtb + '/' + InputData.KinSource + '/' + InputData.SuffixHDF5
    )
    # Reading rates
    utils.print_main("Reading rates ...")
    syst.read_rates(TTranVec=InputData.TTranVec)
    # Reading rates masking
    if InputData.PathToRatesMaskingHDF5 is not None
        utils.print_main("Reading rates masking ...")
        syst.read_ratesmasking(
            TTranVec=InputData.TTranVec,
            PathToHDF5File=InputData.PathToRatesMaskingHDF5
        )
    # Reading rates
    utils.print_main("Writing thermo files to database ...")
    for Comp in syst.Components:
        Comp.create_thermofile()
    # Reading rates
    utils.print_main("Writing kinetic files to database ...")
    syst.write_rates(
        TTranVec=InputData.TTranVec,
        SuffixName=InputData.SuffixName,
        MinValueRate=InputData.MinValueRate
    )


@utils.app_decorator()
def main(args):
    
    os.chdir(args.dir)
    
    # InputData module
    sys.path.append(args.dir)
    import InputData

    _main(args, InputData)
