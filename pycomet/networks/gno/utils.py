import numpy as np
import tensorflow as tf

from pycomet import operations as ops
from pycomet import utils as pcm_utils


# Graph data
# =====================================
def unpack_graph(
  graph,
  edge_attr=None,
  edge_index=None
):
  elem = pcm_utils.unpack_list(graph, nb_elem=4)
  node_attr, _edge_attr, cond_attr, _edge_index = elem
  edge_attr = edge_attr if (edge_attr is not None) else _edge_attr
  edge_index = edge_index if (edge_index is not None) else _edge_index
  return (node_attr, edge_attr, cond_attr, edge_index)

def return_graph(
  node_attr,
  edge_attr=None,
  cond_attr=None,
  edge_index=None
):
  return (node_attr, edge_attr, cond_attr, edge_index)

# def unpack_graph(
#   graph,
#   edge_attr=None,
#   edge_index=None
# ):
#   node_attr, _edge_attr, _edge_index = pcm_utils.unpack_list(graph, nb_elem=3)
#   edge_attr = edge_attr if (edge_attr is not None) else _edge_attr
#   edge_index = edge_index if (edge_index is not None) else _edge_index
#   return (node_attr, edge_attr, edge_index)

# def return_graph(
#   node_attr,
#   edge_attr=None,
#   edge_index=None
# ):
#   graph = [node_attr]
#   if (edge_attr is not None):
#     graph.append(edge_attr)
#   if (edge_index is not None):
#     graph.append(edge_index)
#   return tuple(graph)

# Edge/Node data
# =====================================
def read_edge_index(
  data=None,
  identifier="edge_index"
):
  data = _read_data(data, identifier)
  if (data is not None):
    sh = tf.shape(data)
    if ((len(sh) != 2) or (2 not in sh)):
      pcm_utils.raise_err(
        ValueError,
        f"'{identifier}' input must be a 2-dimensional " \
        "array with a shape of [nb_edges, 2]."
      )
    if (sh[1] == 2):
      data = tf.transpose(data)
    return tf.cast(data, dtype=tf.int32)

def read_attr(
  data=None,
  identifier="edge_attr"
):
  data = _read_data(data, identifier)
  if (data is not None):
    data = tf.expand_dims(data, axis=0)
    # Force 3D tensor
    rank = len(data.shape)
    for _ in range(3-rank):
      data = tf.expand_dims(data, axis=-1)
    return data

def _read_data(
  data=None,
  identifier="edge"
):
  if (data is not None):
    if isinstance(data, (np.ndarray, tf.Tensor)):
      pass
    elif isinstance(data, str):
      data = np.loadtxt(data)
    else:
      pcm_utils.raise_err(
        ValueError,
        f"'{identifier}' input must be either an array " \
        "with a shape of [nb_edges/nb_nodes, :] or a string " \
        "representing a valid file path for reading."
      )
    data = tf.constant(ops.to_tensor(data))
    return tf.squeeze(data)
