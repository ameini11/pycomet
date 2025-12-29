import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as bkd_ke
from tensorflow.python.keras import backend as bkd_py


def set(cfg={}):
  _set_floatx(getattr(cfg, "floatx", "float32"))
  _set_epsilon(getattr(cfg, "epsilon", None))

# Precision
# -------------------------------------
def finfo():
  return tf.experimental.numpy.finfo(get_floatx())

def get_floatx(bkd="tf"):
  if (bkd == "tf"):
    return _FLOATX
  else:
    return {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]

def _set_floatx(value):
  global _FLOATX
  _FLOATX = str(value).lower()
  for k in (bkd_py, bkd_ke):
    k.set_floatx(_FLOATX)

# Epsilon
# -------------------------------------
def get_epsilon():
  return _EPSILON

def _machine_eps():
  return float(finfo().eps)

def _set_epsilon(value=None):
  global _EPSILON
  _EPSILON = _machine_eps() if (value is None) else value
  for k in (bkd_py, bkd_ke):
    k.set_epsilon(_EPSILON)
