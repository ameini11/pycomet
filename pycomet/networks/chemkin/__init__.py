import importlib
hypernet = importlib.util.find_spec("hypernet")

from .combustion import *
from .plasma import *
from .plasma_cr import *
if (hypernet is not None):
  from .noneq import *
