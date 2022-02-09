# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Build a ML-based approximator for ODEs
# --------------------------------------------------------------------------- #


NAME = 'trainPro'

import os
import sys
import prode
import argparse
import importlib
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
        description="Build a ML-based approximator for ODEs.",
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
    input_modules = {
        'data': inputs.data,
        'engine': inputs.engine,
        'general': inputs.general,
        'network': inputs.network,
        'post': inputs.postprocessing
    }
    from inputs.general import flags

    # Physics-related module
    from physics.data_generator import DataGenerator

    # Data
    ###########################################################################
    # Generate Data -----------------------------------------------------------
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

    if inputs.data.from_file:
        utils.print_main("Extracting training/validation pts from file ...")
        train_data = inputs.data.file['train']
        valid_data = inputs.data.file['valid']
        if flags['test'] > 0 and inputs.data.test_cases > 0:
            utils.print_main('Generating testing pts ...')
            test_data = dataGen.testing_data(
                inputs.data.file['test'],
                header=inputs.data.header
            )
    else:
        utils.print_main("Generating initial conditions ...")
        ic = dataGen.initial_cond()
        utils.print_main("Generating training/validation pts ...")
        train_data, valid_data = dataGen.training_data(ic[0])
        if flags['test'] > 0 and inputs.data.test_cases > 0:
            utils.print_main('Generating testing pts ...')
            test_data = dataGen.testing_data(ic[1], ic_train=ic[0])

    # Data Class --------------------------------------------------------------
    data = fun._data(
        flags=flags,
        input_modules=input_modules,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data or None,
    )

    # Network 
    ###########################################################################
    net = fun._net(
        input_modules=input_modules
    )

    # Model 
    ###########################################################################
    model = fun._model(
        flags=flags,
        input_modules=input_modules,
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
    if inputs.data.test_cases > 0 and flags['test'] > 0:
        utils.print_main("Postprocessing physical quantities ...")
        for i, data in enumerate(tqdm(data.test)):
            model.predict(
                data=data,
                plotting=flags['plot'],
                path=model.dir['post']+'/testing/test_'+str(i+1),
                **inputs.postprocessing.plot_style
            )
