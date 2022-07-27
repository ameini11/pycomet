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
import tensorflow as tf

from surqct import config as cfg
from pycomet.apps import functions as pcm_fun
from surqct.apps import functions as sqct_fun

from pycomet.utilities import utils as pcm_utils
from surqct.src.utilities import utils as sqct_utils
from surqct.src.rates_type import rates_type
from surqct.src.chemical_systems import systems as syst
from surqct.src.rates_type.rates_type import RatesType


# Parse arguments
###############################################################################
def get_parser(subparsers):
    parser = subparsers.add_parser(
        name=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Save predicted rates matrices from trained SurQCT model to HDF5 file.",
        epilog=cfg.EPILOG
    )
    parser.add_argument('-j', '--threads',
        type=int,
        default=4,
        help='number of threads'
    )
    parser.add_argument('-d', '--dir',
        type=sqct_fun.dir_path,
        default="./",
        help='path to the working directory containing `InputData.py`'
    )
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
def _main(args=None, InputData=None):
    
    for iRatesType in tf.nest.flatten(InputData.RatesType):
        sqct_utils.print_main(
            f"Generating rates matrix for '{iRatesType}' process", start=''
        )
        wd = InputData.PathToModelFld[iRatesType]

        # PyCOMET model
        #######################################################################
        os.chdir(wd)
        sqct_utils.print_main("Initializing PyCOMET ...")

        # Add modules ---------------------------------------------------------
        sys.path.insert(0, wd)
        # Input module 
        # if 'inputs'
        # sys.modules[name]
        # import inputs
        inputs = pcm_utils.reload_mod(mod_names=['inputs'])
        from inputs.general import flags
        
        # Check flags ---------------------------------------------------------
        for k in ('train', 'save'):
            flags[k] = 0
        flags['load'] = 1
        if flags['load'] not in (1, 2):
            sqct_utils.raise_value_err(
                "The network has to be restored from a saved or a checkpointed "
                "one. Modified the 'load' entry in the `flags` input variable."
            )

        # Network -------------------------------------------------------------
        net = pcm_fun._net(
            input_mod=inputs
        )

        # Model ---------------------------------------------------------------
        model = pcm_fun._model(
            flags=flags,
            input_mod=inputs,
            net=net
        )
        
        # SurQCT model
        #######################################################################
        sqct_utils.print_main(f"Initializing SurQCT ...")

        # Initializing chemical system ----------------------------------------
        utils.print_main(
            f"Initializing '{InputData.SystemName}' chemical system and rates type ...",
            start=''
        )
        system = syst.get_system(
            NameLong=InputData.SystemName,
            ProcNames=iRatesType
        )
        rates_type = RatesType(system, InputData)

        # Saving data
        #######################################################################
        rates_type.save_predictiondata(
            Model=model.net,
            Columns=inputs.data.columns['x'],
            RatesType=iRatesType
        )
        
        del inputs, net, model, sys.path[0]


@sqct_utils.app_decorator()
def main(args):
    
    os.chdir(args.dir)
    
    # InputData module
    sys.path.append(args.dir)
    import InputData

    _main(args, InputData)