# *************************************************************************** #
#                                    PyCOMET                                    #
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
import pycomet
import argparse
import tensorflow as tf

from tqdm import tqdm
from pycomet.utilities import utils
from pycomet.apps import functionObjects as fun


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

    # Configure ---------------------------------------------------------------
    # TensorFlow
    tf.config.threading.set_inter_op_parallelism_threads(opts.threads)
    tf.config.threading.set_intra_op_parallelism_threads(opts.threads)
    utils.environment()
    # Seed
    utils.set_seeds(opts.seed)

    # Add modules -------------------------------------------------------------
    sys.path.append(opts.dir)
    # Input module
    import inputs
    from inputs.general import flags

    # Check flags -------------------------------------------------------------
    for k in ('train', 'save'):
        flags[k] = 0
    if flags['load'] not in (1, 2):
        utils.raise_value_err(
            "The network has to be restored from a saved or a checkpointed "
            "one. Modified the 'load' entry in the `flags` input variable."
        )

    # Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
    if inputs.data.from_file:
        utils.print_main("Extracting testing pts from file ...")
        test_data = inputs.data.file['test']
    else:
        raise NotImplementedError
        # utils.print_main("Generating testing pts ...")
        # # Physics-related module
        # from physics.data_generator import DataGenerator

        # dataGen_args = utils.get_args_values(
        #     args=utils.get_args(DataGenerator.__init__),
        #     modules=[inputs.data, inputs.general]
        # )
        # dataGen = DataGenerator(
        #     **dataGen_args
        # )
        # dataGen.data_dir = opts.dir + "/data/"
        # if not os.path.exists(dataGen.data_dir):
        #     os.makedirs(dataGen.data_dir)

        # ic = dataGen.initial_cond()
        # utils.print_main("Generating training/validation pts ...")
        # train_data, valid_data = dataGen.training_data(ic[0])
        # if flags['test'] > 0 and inputs.data.test_cases > 0:
        #     utils.print_main('Generating testing pts ...')
        #     test_data = dataGen.testing_data(ic[1], ic_train=ic[0])

    # Data Class --------------------------------------------------------------
    data = fun._data(
        flags=flags,
        input_module=inputs,
        test_data=test_data or None,
        wdir=opts.dir
    )

    # Network 
    ###########################################################################
    net = fun._net(
        input_module=inputs
    )

    # Model 
    ###########################################################################
    model = fun._model(
        flags=flags,
        input_module=inputs,
        net=net,
        data=data,
        graph=opts.graph,
        verbose=opts.verbose
    )

    # Postprocessing
    ###########################################################################
    # Training phase ----------------------------------------------------------
    model.visualize_training(history=model.dir['train']+'/history.csv')

    # Testing phase -----------------------------------------------------------
    if flags['test'] > 0:
        utils.print_main("Postprocessing physical quantities ...")
        for name, d in tqdm(data.test.items()):
            model.predict(
                data=d,
                plotting=flags['plot'],
                path=model.dir['post'] + f'/testing/{name}',
                **inputs.postprocessing.plot_style
            )
