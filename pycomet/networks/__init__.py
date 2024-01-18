# Generic models
# =====================================
__all__ = [
  "AugOpNN",
  "FFNN",
  "FlexOpNN",
  "MIONet",
  "MixedNet",
  "MMOpNN",
  "OpNN"
]

from .don import *
from .ffnn import FFNN
from .mixednet import MixedNet

# Thermochemistry-specific models
# =====================================
import importlib
# Check if external library 'hypernet' exists
lib = importlib.util.find_spec("hypernet")
if (lib is not None):
  from .chemreacnn import *
