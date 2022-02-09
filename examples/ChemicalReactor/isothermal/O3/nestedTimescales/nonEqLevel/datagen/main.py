# *************************************************************************** #
#                                    PrODE                                    #
# --------------------------------------------------------------------------- #
#                     Machine-Learning-Based Approximators                    #
#                  for Ordinary Differential Equations (ODEs)                 #
#                                                                             #
# *************************************************************************** #

# -------------------------------- EXEC FILE -------------------------------- #
# Description:
# >> Data generator
# --------------------------------------------------------------------------- #


# INPUTS
# -----------------------------------------------------------------------------
NAME = 'dataGenPro'
case_dir = '../'
ics_test_file = './trained_model/data/test/ics.csv'
# -----------------------------------------------------------------------------

import os
import sys
import tensorflow as tf

# PrODE module
sys.path.append(os.path.normpath(os.environ['WORKSPACE_PATH'] + '/ML/PrODE/prode/'))
from prode.utilities import utils

# Data generator module
sys.path.append(os.path.abspath('./'))
from generator import DataGenerator

# Input module
sys.path.append(os.path.abspath(case_dir))
import inputs
from inputs.general import flags


@utils.app_decorator(name=NAME)
def main():

    # Configure TensorFlow ----------------------------------------------------
    tf.config.threading.set_inter_op_parallelism_threads(4)
    tf.config.threading.set_intra_op_parallelism_threads(4)
    utils.environment()

    # Generate data -----------------------------------------------------------
    dataGen_args = utils.get_args_values(
        args=utils.get_args(DataGenerator.__init__),
        modules=[inputs.data, inputs.general]
    )
    dataGen = DataGenerator(
        **dataGen_args
    )
    dataGen.data_dir = case_dir + "/data/"
    if not os.path.exists(dataGen.data_dir):
        os.makedirs(dataGen.data_dir)

    utils.print_main("Generating initial conditions ...")
    ic = dataGen.initial_cond()

    utils.print_main("Generating training/validation pts ...")
    dataGen.training_data(ic[0])

    if flags['test'] > 0 and inputs.data.test_cases > 0:
        utils.print_main('Generating testing pts ...')
        if ics_test_file is not None:
            kwargs = dict(
                ic_test=ics_test_file,
                header=inputs.data.header
            )
        else:
            kwargs = dict(
                ic_test=ic[1],
                ic_train=ic[0]
            )
        dataGen.testing_data(**kwargs)


if __name__ == '__main__':
    main()
