from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


# Configure
from pycomet.config import *
create_config()
cfg = read_config()

# Environment
import os
os.environ.update(cfg.env)

# Backend
from pycomet import backend
backend.set_floatx(cfg.floatx)
backend.set_epsilon()

# Logging
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(absl.logging.ERROR)

# Custom packages
from pycomet.utils.modules import append_mod_dir
append_mod_dir(cfg.custom_pkg)

# PyCOMET
# -------------------------------------
__author__  = "Ivan Zanardi"
__email__   = "zanardi3@illinois.edu"
__url__     = "https://github.com/ivanZanardi/pycomet"
__license__ = "Apache-2.0"
__version__ = "0.1"

# Import all modules
from pycomet import apps
from pycomet import backend
from pycomet import config
from pycomet import data
from pycomet import math
from pycomet import model
from pycomet import networks
from pycomet import operations
from pycomet import optimization
from pycomet import physics
from pycomet import postproc
from pycomet import utils

__all__ = [
    "apps",
    "backend",
    "config",
    "data",
    "math",
    "model",
    "networks",
    "operations",
    "optimization",
    "physics",
    "postproc",
    "utils"
]
