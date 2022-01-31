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


NAME = 'testPro'

import os
import sys
import argparse
import prode
import tensorflow as tf

from tqdm import tqdm
from prode.utilities import utils


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
    import inputs as inp
    # Physics-related module
    try:
        from physics.physical_model import PhysicalModel
        physics_informed = True
    except ModuleNotFoundError:
        physics_informed = False
    from physics.data_generator import DataGenerator

    # Data
    ###########################################################################
    # Generate Testing Data ---------------------------------------------------
    dataGen_args = utils.get_args_values(
        args=utils.get_args(DataGenerator.__init__),
        modules=[inp.data, inp.general]
    )
    dataGen = DataGenerator(
        **dataGen_args
    )
    dataGen.data_dir = opts.dir + "/data/"
    if not os.path.exists(dataGen.data_dir):
        os.makedirs(dataGen.data_dir)

    utils.print_main("Generating testing pts ...")
    if inp.data.from_file:
        test_data = dataGen.testing_data(
            inp.data.file['test'],
            header=inp.data.header
        )
    else:
        ic = dataGen.initial_cond()
        test_data = dataGen.testing_data(ic)

    # Network
    ###########################################################################
    # Initialize Network ------------------------------------------------------
    net_cls = utils.get_class(prode.networks, inp.network.architecture)
    utils.print_main(
        "Initializing '{}' network ...".format(inp.network.architecture)
    )
    net_args = utils.get_args_values(
        args=utils.get_args(net_cls.__init__),
        modules=[inp.network, inp.general]
    )
    net = net_cls(**net_args)

    # Apply I/O Transformation ------------------------------------------------
    if hasattr(inp.network, 'input_fn'):
        net.inp_trans = inp.network.input_fn
    if hasattr(inp.network, 'output_fn'):
        net.out_trans = inp.network.output_fn

    # Model
    ###########################################################################
    # Initialize Model --------------------------------------------------------
    model = prode.model.Model(net=net, name=inp.general.model_name)

    # Build Model -------------------------------------------------------------
    utils.print_main(
        "Building '{}' network ...".format(inp.network.architecture)
    )
    model.build(
        load_net=inp.general.flags['load'],
        visualize_graph=opts.graph,
        verbose=opts.verbose
    )

    # Compile Model -----------------------------------------------------------
    utils.print_main(
        "Compiling '{}' network ...".format(inp.network.architecture)
    )
    compile_args = utils.get_args_values(
        args=utils.get_args(model.compile),
        modules=inp.engine
    )
    model.compile(**compile_args)

    # Postprocessing - Testing
    ###########################################################################
    if inp.general.flags['test'] > 0:
        utils.print_main("Testing physical scenarios ...")
        for i, data in enumerate(tqdm(data.test)):
            model.predict(
                data=data,
                plotting=inp.general.flags['plot'],
                path=model.dir['post']+'/testing/test_'+str(i+1),
                **inp.postprocessing.plot_style
            )
