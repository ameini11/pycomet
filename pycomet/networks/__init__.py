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

from .chemkin import *
from .don import *
from .gno import *
from .meno import *
from .ffnn import FFNN
from .mixednet import MixedNet

from . import chemkin
from . import elements as nn_elem
from .don import blocks as don_blk
from .gno import blocks as gno_blk
from .gno import layers as gno_lay
from .meno import blocks as meno_blk
from .meno import layers as meno_lay
_NN_MOD.extend(
  [chemkin, nn_elem, don_blk, gno_blk, gno_lay, meno_blk, meno_lay]
)

# TensorFlow default layers
# =====================================
from keras import layers as tf_layers
_NN_MOD.append(tf_layers)
