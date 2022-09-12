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
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
# Setting backend
from tensorflow.python.keras import backend as be_py
from tensorflow.keras import backend as be_ke
for k in (be_py, be_ke):
    k.set_floatx(_TF['setup']['FLOATX'])
    k.set_epsilon(_TF['setup']['EPSILON'])

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
from pycomet import physics
from pycomet import postproc
from pycomet import utils
from pycomet.apps import import_inputs, load_model

__all__ = [
    "apps",
    "config",
    "data",
    "model",
    "networks",
    "operations",
    "optimization",
    "physics",
    "postproc",
    "utils"
]
