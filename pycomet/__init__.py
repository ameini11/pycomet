from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TensorFlow 
###############################################################################
import os
import pycomet
from pycomet.config import tf_setup
os.environ['TF_CPP_MIN_LOG_LEVEL'] = tf_setup['TF_CPP_MIN_LOG_LEVEL']
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
tf.keras.backend.set_floatx(tf_setup['DTYPE'])
tf.keras.backend.set_epsilon(tf_setup['EPSILON'])

# PyCOMET
###############################################################################
__author__  = "Ivan Zanardi"
__email__   = "zanardi3@illinois.edu"
__url__     = "https://github.com/ivanZanardi/pycomet"
__license__ = "Apache-2.0"
__version__ = "0.0.1"

# Import all modules
from pycomet import apps
from pycomet import config
from pycomet import data
from pycomet import networks
from pycomet import physics
from pycomet.model import Model
from pycomet import postprocessing

__all__ = [
    "apps",
    "config",
    "data",
    "networks",
    "physics",
    "Model",
    "postprocessing"
]
