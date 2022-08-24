# *************************************************************************** #
#                                   PyCOMET                                   #
# --------------------------------------------------------------------------- #
#                    Physics-informed machine learning for                    #
#                 scientific computing and operator discovery                 #
#                                                                             #
# *************************************************************************** #

# --------------------------- INITIALIZATION FILE --------------------------- #


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TensorFlow ------------------------------------------------------------------
import os
from pycomet.config import _TF

# Setting verbosity level
os.environ.update(_TF['env'])
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Setting backend
from tensorflow.python.keras import backend
backend.set_floatx(_TF['setup']['FLOATX'])
backend.set_epsilon(_TF['setup']['EPSILON'])

# PyCOMET ---------------------------------------------------------------------
__author__  = "Ivan Zanardi"
__email__   = "zanardi3@illinois.edu"
__url__     = "https://github.com/ivanZanardi/pycomet"
__license__ = "Apache-2.0"
__version__ = "0.1"

# Import all modules
from pycomet import apps
from pycomet import config
from pycomet import data
from pycomet import model
from pycomet import networks
from pycomet import operations
from pycomet import optimization
from pycomet import postprocessing
from pycomet import utils

__all__ = [
    "apps",
    "config",
    "data",
    "model",
    "networks",
    "operations",
    "optimization",
    "postprocessing",
    "utils"
]
