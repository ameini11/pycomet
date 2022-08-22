# *************************************************************************** #
#                                   PyCOMET                                   #
# --------------------------------------------------------------------------- #
#                    Physics-informed machine learning for                    #
#                scientific computations and operator discovery               #
#                                                                             #
# *************************************************************************** #

# --------------------------- INITIALIZATION FILE --------------------------- #


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TensorFlow 
###############################################################################
import os
from pycomet.config import tf_setup
from pycomet.config import tf_env

# Setting verbosity level
os.environ.update(tf_env)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Setting python/keras backends
from tensorflow.keras import backend as keras_backend
keras_backend.set_floatx(tf_setup['DTYPE'])
keras_backend.set_epsilon(tf_setup['EPSILON'])
from tensorflow.python.keras import backend as python_backend
python_backend.set_floatx(tf_setup['DTYPE'])
python_backend.set_epsilon(tf_setup['EPSILON'])

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
from pycomet import model
from pycomet import postprocessing

__all__ = [
    "apps",
    "config",
    "data",
    "networks",
    "physics",
    "model",
    "postprocessing"
]
