# Generic models
# =====================================
__all__ = [
  "AugOpNN",
  "EncodeProcessDecode",
  "FFNN",
  "FlexOpNN",
  "MIONet",
  "MixedNet",
  "MMOpNN",
  "OpNN"
]

_NN_MOD = []

from .don import *
from .fno import *
from .gno import *
from .ffnn import FFNN
from .mixednet import MixedNet

from . import elements as nn_elem
from .don import blocks as don_blk
from .gno import blocks as gno_blk
_NN_MOD.extend([nn_elem, don_blk, gno_blk])

# # Thermochemistry-specific models
# # =====================================
# import importlib
# lib = importlib.util.find_spec("hypernet")
# if (lib is not None):
#   from .chem import *
#   from . import chem
#   _NN_MOD.append(chem)

# TensorFlow default layers
# =====================================
from keras import layers as tf_layers
_NN_MOD.append(tf_layers)
