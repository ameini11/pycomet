# Future imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Package
__author__ = "Ivan Zanardi"
__email__ = "zanardi3@illinois.edu"
__url__ = "https://github.com/ivanZanardi/pycomet"
__license__ = "Apache-2.0"
__version__ = "0.0.1"

__all__ = [
  "apps",
  "backend",
  "config",
  "data",
  "device",
  "model",
  "networks",
  "operations",
  "optimization",
  "physics",
  "postproc",
  "sampling",
  "utils"
]

# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")

# Configuration
from pycomet import config
config.read_config()
_CFG = config.get_config()

# Environment
import os
os.environ.update(_CFG.env)

# Backend library
import tensorflow as tf

# Backend
from pycomet import backend
backend.set(_CFG)

# Logging
tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

# Custom packages
from pycomet import utils
utils.append_mod_dir(getattr(_CFG, "custom_pkg", {}))

# Remaining modules
from pycomet import apps
from pycomet import data
from pycomet import device
from pycomet import model
from pycomet import networks
from pycomet import operations
from pycomet import optimization
from pycomet import physics
from pycomet import postproc
from pycomet import sampling

# Checking
res = config.check_config()
if (res is not None):
  utils.raise_err(**res)
