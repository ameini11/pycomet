import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from . import activations
from . import callbacks
from . import config
from . import data
from . import initializers
from . import layers
from . import net
# from .model import Model
from ._model import Model
from . import postprocessing
from . import regularizers


# PrODE
__author__  = "Ivan Zanardi"
__email__   = "zanardi3@illinois.edu"
__url__     = "https://github.com/ivanZanardi/prode"
__license__ = "Apache-2.0"
__version__ = "0.0.1"

__all__ = [
    "activations",
    "callbacks",
    "config",
    "data",
    "initializers",
    "layers",
    "net",
    "Model",
    "postprocessing",
    "regularizers"
]