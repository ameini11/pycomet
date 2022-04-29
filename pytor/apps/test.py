# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Test a ML-based approximator for ODEs
# --------------------------------------------------------------------------- #


NAME = 'testPro'

import os
import sys
import prode
import argparse
import tensorflow as tf

from tqdm import tqdm
from prode.utilities import utils
from prode.apps import functionObjects as fun


def dir_path(path):
    if os.path.exists(path+'/inputs'):
        return path
    else:
        raise argparse.ArgumentTypeError(
            "'inputs' folder not found in '{}'".format(
                os.path.normpath(path)+'/'
            )
        )

def get_opts():
    parser = argparse.ArgumentParser(
        prog=NAME,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Test a ML-based approximator for ODEs.",
        epilog=utils.app_epilog(name=NAME)
    )
    parser.add_argument('-j', '--threads',
        type=int,
        default=16,
        help='number of threads'
    )
    parser.add_argument('-d', '--dir',
        type=dir_path,
        default="./",
        help='path to the working directory (with `inputs` folder)'
    )
    parser.add_argument('-v', '--verbose',
        type=int,
        default=0,
        choices=[0,1],
        help='verbose mode'
    )
    parser.add_argument('-g', '--graph',
        type=int,
        default=0,
        choices=[0,1],
        help='visualize model graph'
    )
    return parser.parse_args()


@utils.app_decorator(name=NAME)
def main():

    # Inizialization
    ###########################################################################
    # Define arguments --------------------------------------------------------
    opts = get_opts()
    if opts.graph:
        opts.verbose = 1

    # Configure TensorFlow ----------------------------------------------------
    tf.config.threading.set_inter_op_parallelism_threads(opts.threads)
    tf.config.threading.set_intra_op_parallelism_threads(opts.threads)
    utils.environment()

    # Add modules -------------------------------------------------------------
    sys.path.append(opts.dir)
    # Input module
    import inputs
    from inputs.general import flags
    # Physics-related module
    from physics.data_generator import DataGenerator

    # Data
    ###########################################################################
    # Generate Testing Data ---------------------------------------------------
    dataGen_args = utils.get_args_values(
        args=utils.get_args(DataGenerator.__init__),
        modules=[inputs.data, inputs.general]
    )
    dataGen = DataGenerator(
        **dataGen_args
    )
    dataGen.data_dir = opts.dir + "/data/"
    if not os.path.exists(dataGen.data_dir):
        os.makedirs(dataGen.data_dir)

    utils.print_main("Generating testing pts ...")
    if inputs.data.from_file:
        test_data = dataGen.testing_data(
            inputs.data.file['test'],
            header=inputs.data.header
        )
    else:
        raise NotImplementedError

    # Network 
    ###########################################################################
    net = fun._net(
        input_module=inputs
    )

    # Model 
    ###########################################################################
    for k in ('train', 'save'):
        flags[k] = 0
    if flags['load'] not in (1, 2):
        utils.raise_value_err(
            "The network has to be restored from a saved or a checkpointed "
            "one. Modified the 'load' entry in the `flags` input variable."
        )
    model = fun._model(
        flags=flags,
        input_module=inputs,
        net=net,
        graph=opts.graph,
        verbose=opts.verbose
    )

    # Postprocessing - Testing 
    ###########################################################################
    if flags['test'] > 0:
        utils.print_main("Testing physical scenarios ...")
        for i, data in enumerate(tqdm(test_data)):
            model.predict(
                data=data,
                plotting=flags['plot'],
                path=model.dir['post']+'/testing/test_'+str(i+1),
                **inputs.postprocessing.plot_style
            )
