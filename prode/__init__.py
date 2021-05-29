from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from . import activations
from . import callbacks
from . import data
from . import initializers
from . import layers
from . import net
from .model import Model
from . import postprocessing
# from .utils import apply
from . import regularizers


__all__ = [
    "activations",
    "callbacks",
    "data",
    "initializers",
    "layers",
    "net",
    "Model",
    "postprocessing",
    "regularizers",
    # "apply",
]


# PrODE
__author__  = "Ivan Zanardi"
__email__   = "zanardi3@illinois.edu"
__url__     = "https://github.com/ivanZanardi/prode"
__license__ = "Apache-2.0"
__version__ = "0.0.1"
