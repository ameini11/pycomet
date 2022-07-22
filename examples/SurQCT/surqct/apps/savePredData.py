# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #


NAME = 'savePredData'

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
from surqct.src.rates_type import rates_type
from surqct.src.chemical_systems import systems as syst


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Save predicted rates matrix to HDF5 file from trained SurQCT model.",
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
@utils.app_decorator()
def main(args):
    
    os.chdir(args.dir)
    
    # InputData module --------------------------------------------------------
    sys.path.append(args.dir)
    import InputData

    for RatesType in tf.nest.flatten(InputData.RatesType):
        utils.print_main(
            f"Generating rates matrix for '{InputData.RatesType}' process",
            start=''
        )
        wd = PathToRunFld[RatesType]

        # PyCOMET model
        #######################################################################
        os.chdir(wd)
        utils.print_main("Initializing PyCOMET ...")

        # Add modules ---------------------------------------------------------
        sys.path.append(wd)
        # Input module
        import inputs
        print(inputs.__file__)
        from inputs.general import flags

        # Check flags ---------------------------------------------------------
        for k in ('train', 'save'):
            flags[k] = 0
        flags['load'] = 1
        if flags['load'] not in (1, 2):
            utils.raise_value_err(
                "The network has to be restored from a saved or a checkpointed "
                "one. Modified the 'load' entry in the `flags` input variable."
            )

        # Network -------------------------------------------------------------
        net = pycomet_fun._net(
            input_mod=inputs
        )

        # Model ---------------------------------------------------------------
        model = pycomet_fun._model(
            flags=flags,
            input_mod=inputs,
            net=net
        )
        
        # SurQCT model
        #######################################################################
        utils.print_main(f"Initializing SurQCT ...")

        # Initializing chemical system ----------------------------------------
        utils.print_main(
            f"Initializing '{InputData.SystemName}' chemical system ..."
        )
        system = syst.get_system(
            NameLong=InputData.SystemName,
            ProcNames=RatesType
        )
        # Initializing rates type module
        rates_type_mod = rates_type.get_rates_type_mod(system)

        # Saving data
        #######################################################################
        rates_type_mod.save_predictiondata(
            InputData=InputData,
            System=system,
            Model=model.net,
            Columns=inputs.data.columns['x']
        )

        del sys.path[-1], model, net
