import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import backend as bkd


def tensor_split(tensor, indices_or_sections, axis=0):
  try:
    ntot = tensor.shape[axis]
  except AttributeError:
    ntot = len(tensor)
  try:
    nsec = len(indices_or_sections) + 1
    div_points = [0] + list(indices_or_sections) + [ntot]
  except TypeError:
    nsec = int(indices_or_sections)
    if (nsec <= 0):
      utils.raise_err(ValueError, "Number of sections must be larger than 0.")
    neach, extras = ntot // nsec, ntot % nsec
    section_sizes = tf.nest.flatten(
      [0] + extras * [neach+1] + (nsec-extras) * [neach]
    )
    div_points = np.array(section_sizes, dtype=int).cumsum().tolist()
  sub_tensors = []
  for i in range(nsec):
    indices = np.arange(div_points[i],div_points[i+1])
    itensor = tf.gather(tensor, indices, axis=axis)
    sub_tensors.append(itensor)
  return sub_tensors

def to_tensor(x=None, dtype=None):
  if (x is not None):
    if (dtype is None):
      dtype = bkd.get_floatx()
    if (tf.is_tensor(x)):
      return tf.cast(x, dtype)
    else:
      return tf.convert_to_tensor(to_numpy(x), dtype=dtype)

def to_numpy(x=None, dtype=None):
  if (x is not None):
    if (dtype is None):
      dtype = bkd.get_floatx("np")
    if (tf.is_tensor(x)):
      return x.numpy().astype(dtype)
    elif isinstance(x, (int, float, list, tuple)):
      return np.array(x, dtype=dtype)
    else:
      return x.astype(dtype)
    
def random_permutation(n):
  return tf.convert_to_tensor(np.random.permutation(n), dtype=int)
