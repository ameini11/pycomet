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

def split_static_prop(prop, indices):
  return [prop[i] for i in indices]

# Reaction rates
# =====================================
def read_rates(rates):
  if isinstance(rates, str):
    rates = h5todict(h5file=rates)
  return tf.nest.map_structure(lambda x: ops.to_numpy(x).squeeze(), rates)

def split_rates(rates, indices):
  return [
    tf.nest.map_structure(lambda x: _split_rates(x, i), rates) for i in indices
  ]

def _split_rates(x, indices):
  xs = x[indices]
  if (len(x.shape) == 2):
    xs = xs[:,indices]
  return xs
