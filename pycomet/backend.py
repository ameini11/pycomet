import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as bkd_ke
from tensorflow.python.keras import backend as bkd_py
from pycomet.config import VALID_FLOATXS, get_config


def set(cfg=None):
  if (cfg is None):
    cfg = get_config()
  _set_floatx(getattr(cfg, "floatx", "float32"))
  _set_epsilon()

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
  if (_FLOATX not in VALID_FLOATXS):
    _FLOATX = "float32"
  for k in (bkd_py, bkd_ke):
    k.set_floatx(_FLOATX)

# Epsilon
# -------------------------------------
def get_epsilon():
  return _EPSILON

def _machine_eps():
  return float(finfo().eps)

def _set_epsilon():
  global _EPSILON
  _EPSILON = _machine_eps()
  for k in (bkd_py, bkd_ke):
    k.set_epsilon(_EPSILON)
