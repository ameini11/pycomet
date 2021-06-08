from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from prode.config import tf_setup

# TensorFlow setup
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_setup['TF_CPP_MIN_LOG_LEVEL']
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx(tf_setup['DTYPE'])
tf.keras.backend.set_epsilon(tf_setup['EPSILON'])
tf.config.threading.set_inter_op_parallelism_threads(tf_setup['NUM_THREADS'])
tf.config.threading.set_intra_op_parallelism_threads(tf_setup['NUM_THREADS'])

# PrODE setup
__author__  = "Ivan Zanardi"
__email__   = "zanardi3@illinois.edu"
__url__     = "https://github.com/ivanZanardi/prode"
__license__ = "Apache-2.0"
__version__ = "0.0.1"

from prode import config
from prode import data
from prode import networks
from prode.model import Model
from prode import postprocessing

__all__ = [
    "config",
    "data",
    "networks",
    "Model",
    "postprocessing"
]