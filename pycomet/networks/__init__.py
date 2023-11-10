# Generic models
# =====================================
from .augopnn import AugOpNN
from .ffnn import FFNN
from .flexopnn import FlexOpNN
from .mionet import MIONet
from .mixednet import MixedNet
from .mmopnn import MMOpNN
from .opnn import OpNN

__all__ = [
  "AugOpNN",
  "FFNN",
  "FlexOpNN",
  "MIONet",
  "MixedNet",
  "MMOpNN",
  "OpNN"
]

# Thermochemistry-specific models
# =====================================
import importlib
# Check if external library 'hypernet' exists
lib = importlib.util.find_spec("hypernet")
if (lib is not None):
  from .chemreacnn import *
