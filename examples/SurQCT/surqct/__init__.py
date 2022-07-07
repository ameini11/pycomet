# *************************************************************************** #
#                                   SurQCT                                    #
# --------------------------------------------------------------------------- #
#                   Surrogate modeling for QCT calculations                   #
#                                                                             #
# *************************************************************************** #

# --------------------------- INITIALIZATION FILE --------------------------- #


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TensorFlow 
###############################################################################
import os
from surqct.config import tf_cfg

# Setting verbosity level
os.environ.update(tf_cfg['env'])
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
# Setting python/keras backends
from tensorflow.keras import backend as keras_backend
keras_backend.set_floatx(tf_cfg['setup']['DTYPE'])
keras_backend.set_epsilon(tf_cfg['setup']['EPSILON'])
from tensorflow.python.keras import backend as python_backend
python_backend.set_floatx(tf_cfg['setup']['DTYPE'])
python_backend.set_epsilon(tf_cfg['setup']['EPSILON'])

# SurQCT
###############################################################################
# Import all modules
from surqct import apps
from surqct import src

__all__ = [
    "apps",
    "src"
]
