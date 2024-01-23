import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import operations as ops


# Graph data
# =====================================
def unpack_graph(
  graph,
  edge_index=None,
  edge_attr=None
):
  if ((edge_index is not None) and (edge_attr is not None)):
    node_attr = graph[-1] if isinstance(graph, (list, tuple)) else graph
  elif ((edge_index is not None) and (edge_attr is None)):
    node_attr, edge_attr = utils.unpack_list(graph, nb_elem=2)
  else:
    node_attr, edge_index, edge_attr = utils.unpack_list(graph, nb_elem=3)
  return node_attr, edge_index, edge_attr

# Edge data
# =====================================
def read_edge_index(
  data=None,
  identifier="edge_index"
):
  data = _read_edge_data(data, identifier)
  if (data is not None):
    sh = tf.shape(data)
    if ((len(sh) > 2) or (2 not in sh)):
      utils.raise_err(
        ValueError,
        f"'{identifier}' input must be a 2-dimensional " \
        "array with a shape of [nb_edges, 2]."
      )
    if (sh[1] == 2):
      data = tf.transpose(data)
    return tf.cast(data, dtype=tf.int32)

def read_edge_attr(
  data=None,
  identifier="edge_attr"
):
  data = _read_edge_data(data, identifier)
  if (data is not None):
    return tf.expand_dims(data, axis=0)

def _read_edge_data(
  data=None,
  identifier="edge"
):
  if (data is not None):
    if isinstance(data, (np.ndarray, tf.Tensor)):
      pass
    elif isinstance(data, str):
      data = np.loadtxt(data).squeeze()
    else:
      utils.raise_err(
        ValueError,
        f"'{identifier}' input must be either an array " \
        "with a shape of [nb_edges, :] or a string " \
        "representing a valid file path for reading."
      )
    return tf.constant(ops.to_tensor(data))
