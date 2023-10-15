import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet.config import device as device_mod
from tensorflow.keras import backend as be_ke
from tensorflow.python.keras import backend as be_py


# Device
# -------------------------------------
def device():
  return device_mod.device()

def set_device(parsed_args):
  return device_mod.set_device(parsed_args)

# Epsilon
# -------------------------------------
def machine_eps():
  return float(np.finfo(
    {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]
  ).eps)

def epsilon():
  return _EPSILON

def set_epsilon():
  global _EPSILON
  _EPSILON = machine_eps()
  # TensorFlow
  for k in (be_py, be_ke):
    k.set_epsilon(_EPSILON)

# Float
# -------------------------------------
def floatx(fmt="tf"):
  if fmt == "tf":
    return _FLOATX
  else:
    return {
      "float16": np.float16,
      "float32": np.float32,
      "float64": np.float64
    }[_FLOATX]

def set_floatx(value):
  global _FLOATX
  accepted_dtypes = {"float16", "float32", "float64"}
  if value not in accepted_dtypes:
    utils.raise_err(
      error=ValueError,
      text=f"Unknown `floatx` value: {value}." \
        " Expected one of {accepted_dtypes}"
    )
  _FLOATX = str(value)
  # TensorFlow
  for k in (be_py, be_ke):
    k.set_floatx(floatx(fmt="tf"))

def recast(structure, fmt="tf"):
  if fmt == "tf":
    fn = lambda x: tf.cast(x, floatx(fmt))
  else:
    fn = lambda x: x.astype(floatx(fmt))
  return tf.nest.map_structure(fn, structure)
