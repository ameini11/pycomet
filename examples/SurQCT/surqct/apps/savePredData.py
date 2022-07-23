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
from surqct.apps import functions as sq_fun

from pycomet.utilities import utils as pcm_utils
from surqct.src.utilities import utils as sq_utils
from surqct.src.rates_type import rates_type
from surqct.src.chemical_systems import systems as syst


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
        type=sq_fun.dir_path,
        default="./",
        help='path to the working directory containing `InputData.py`'
    )
    parser.set_defaults(process=main)
    return parser

# Main
###############################################################################
def _main(args=None, InputData=None):
    
    for RatesType in tf.nest.flatten(InputData.RatesType):
        sq_utils.print_main(
            f"Generating rates matrix for '{RatesType}' process", start=''
        )
        wd = InputData.PathToModelFld[RatesType]

        # PyCOMET model
        #######################################################################
        os.chdir(wd)
        sq_utils.print_main("Initializing PyCOMET ...")

        # Add modules ---------------------------------------------------------
        sys.path.insert(0, wd)
        # Input module
        import inputs
        pcm_utils.reload_mod(modules=[inputs])
        from inputs.general import flags
        
        # Check flags ---------------------------------------------------------
        for k in ('train', 'save'):
            flags[k] = 0
        flags['load'] = 1
        if flags['load'] not in (1, 2):
            sq_utils.raise_value_err(
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
        sq_utils.print_main(f"Initializing SurQCT ...")

        # Initializing chemical system ----------------------------------------
        sq_utils.print_main(
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
            Columns=inputs.data.columns['x'],
            RatesType=RatesType
        )
        
        del inputs, net, model, sys.path[0]


@sq_utils.app_decorator()
def main(args):
    
    os.chdir(args.dir)
    
    # InputData module
    sys.path.append(args.dir)
    import InputData

    _main(args, InputData)