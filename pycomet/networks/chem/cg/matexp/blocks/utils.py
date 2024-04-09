import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import operations as ops
from silx.io.dictdump import h5todict


# State-wise static properties
# =====================================
def read_static_prop(prop):
  if isinstance(prop, (np.ndarray, tf.Tensor)):
    pass
  elif isinstance(prop, str):
    prop = np.loadtxt(prop)
  else:
    utils.raise_err(
      ValueError,
      "The input data must be either an array with shape [nb_states,:] " \
      "or a string representing a valid file path for reading."
    )
  prop = ops.to_numpy(prop).squeeze()
  # Check shape
  if (len(prop.shape) == 1):
    prop = prop.reshape(-1,1)
  return prop

# Reaction rates
# =====================================
def read_rates(rates=None):
  if (rates is not None):
    if isinstance(rates, str):
      rates = h5todict(h5file=rates)
    return tf.nest.map_structure(lambda x: ops.to_numpy(x).squeeze(), rates)
