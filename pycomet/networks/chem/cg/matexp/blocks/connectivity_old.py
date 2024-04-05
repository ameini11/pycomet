import json
import itertools
import tensorflow as tf

from pycomet import utils
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LastToCurrent(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    last_to_current,
    **kwargs
  ):
    super(LastToCurrent, self).__init__(**kwargs)
    # Last-to-current timescale connectivity
    self.last_to_current = last_to_current

  # Calling
  # ===================================
  def call(self, x, training=None):
    if (self.last_to_current is not None):
      y = []
      for indices in self.last_to_current:
        yi = tf.gather(x, indices, axis=-1)
        yi = tf.math.reduce_sum(yi, axis=-1, keepdims=True)
        y.append(yi)
      return tf.concat(y, axis=-1) if (len(y) > 1) else y[-1]
    else:
      return x

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "last_to_current": self.last_to_current
    }
    base_config = super(LastToCurrent, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class CurrentToLast(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    current_to_last,
    **kwargs
  ):
    super(CurrentToLast, self).__init__(**kwargs)
    # Current-to-last timescale connectivity
    self.current_to_last = current_to_last

  # Calling
  # ===================================
  def call(self, x, training=None):
    y = x[0]
    for i in range(1,len(x)):
      indices = self.current_to_last[str(i-1)]
      if (indices is not None):
        # Split the previous slower timescale
        yi = tf.split(y, num_or_size_splits=y.shape[1], axis=-1)
        # Gather the current faster timescale by indices
        xi = [tf.gather(x[i], j, axis=-1) for j in indices]
        # Multiply slower and faster timescales
        y = [z[0] * z[1] for z in zip(yi, xi)]
        # Concatenate
        y = tf.concat(y, axis=-1) if (len(y) > 1) else y[-1]
    # Return the last timescale
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "current_to_last": self.current_to_last
    }
    base_config = super(CurrentToLast, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GroupsConnectivity(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    groups_graph
  ):
    self.groups_graph = groups_graph
    if isinstance(self.groups_graph, str):
      with open(self.groups_graph) as file:
        self.groups_graph = json.load(file)
    # Last-to-current timescale connectivity
    self.last_to_current = self._convert_keys(
      self.connect_last_to_current(self.groups_graph)
    )
    # Current-to-last timescale connectivity
    self.current_to_last = self._convert_keys(
      self.connect_current_to_next(self.groups_graph)
    )

  def _convert_keys(self, d):
    return {str(k): v for (k, v) in d.items()}

  # Groups connectivity/indexing
  # ===================================
  def connect_current_to_next(
    self,
    graph,
    level=0,
    connectivity={}
  ):
    d = {}
    connectivity[level] = []
    if isinstance(graph, dict):
      for g in graph.values():
        if hasattr(g, "keys"):
          l = list(g.keys())
          l = tf.nest.map_structure(int, l)
          d = {**d, **g}
        else:
          l = d = g
        connectivity[level].append(l)
    else:
      connectivity[level] = None
    if (d != {}):
      return self.connect_current_to_next(
        d, level=level+1, connectivity=connectivity
      )
    else:
      return connectivity

  def connect_last_to_current(
    self,
    graph,
    ts={}
  ):
    nb_ts = utils.compute_dict_depth(graph)
    connectivity = self.graph_to_nested_list(graph)
    for l in range(-1,nb_ts):
      ts[l+1] = self.merge(connectivity, it_max=l+1)
    return ts

  def graph_to_nested_list(
    self,
    graph
  ):
    if isinstance(graph, dict):
      ls = []
      for g in graph.values():
        ls.append(self.graph_to_nested_list(g))
      return ls
    else:
      if isinstance(graph, (list, tuple, set)):
        return list(graph)
      else:
        utils.raise_err(
          ValueError,
          "The last level of the groups graph has to be a list."
        )

  def merge(
    self,
    ls,
    it=0,
    it_max=1
  ):
    if any(isinstance(l, list) for l in ls):
      if (it < it_max):
        ls = list(itertools.chain.from_iterable(ls))
        return self.merge(ls, it=it+1, it_max=it_max)
      else:
        return self.join(ls)

  def join(
    self,
    ls
  ):
    try:
      ls = [list(itertools.chain.from_iterable(l)) for l in ls]
      return self.join(ls)
    except:
      return ls


# -------------------------------------
# Example
# -------------------------------------
#
# groups_graph = {
#     0: {
#         0: [0,1,2],
#         1: [3,4,5],
#         2: [6,7,8]
#     }
# }
#
# last_to_current = {
#     0: [
#         [0, 1, 2, 3, 4, 5, 6, 7, 8]
#     ],
#     1: [
#         [0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]
#     ],
#     2: None
# }
#
# current_to_last = {
#     0: [
#         [0, 1, 2]
#     ],
#     1: [
#         [0, 1, 2],
#         [3, 4, 5],
#         [6, 7, 8]
#     ],
#     2: None
# }
#
# -------------------------------------
