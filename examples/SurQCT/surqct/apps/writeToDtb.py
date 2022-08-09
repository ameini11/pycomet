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
        description="Write kinetics and thermo files to Plato database (see 'config.MERuns').",
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
        KinSource=InputData.KinSource,
        ProcNames=InputData.RatesType,
        PathToHDF5Fld=cfg.paths.HDF5Dtb + '/' + InputData.KinSource + '/' + InputData.SuffixHDF5
    )
    # Reading rates
    utils.print_main(
        "Reading rates from "
        f"'{os.path.abspath(system.Processes.PathToHDF5File)}'..."
    )
    system.Processes.read_rates(TTranVec=InputData.TTranVec)
    # Reading rates masking
    if InputData.PathToRatesMaskingHDF5 is not None:
        utils.print_main("Reading rates masking ...")
        system.Processes.read_rates(
            TTranVec=InputData.TTranVec,
            Group='RatesMasking',
            PathToHDF5File=InputData.PathToRatesMaskingHDF5
        )
    # Reading rates
    utils.print_main("Writing thermo files to database ...")
    ThermoDtb = cfg.paths.MERuns + '/database/thermo/' + system.NameLong
    for Comp in system.Components:
        Comp.create_thermofile(ThermoDtb)
    # Reading rates
    utils.print_main("Writing kinetic files to database ...")
    system.Processes.write_rates(
        TTranVec=InputData.TTranVec,
        SuffixName=InputData.SuffixName,
        MinRateValue=InputData.MinRateValue
    )


@utils.app_decorator()
def main(args):
    
    os.chdir(args.dir)
    
    # InputData module
    sys.path.append(args.dir)
    import InputData

    _main(args, InputData)
