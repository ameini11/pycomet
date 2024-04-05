import json
import itertools
import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LastToCurrent(nn_elem.Layer):
  """Last-to-current timescale connectivity."""

  # Initialization
  # ===================================
  def __init__(
    self,
    indices=None,
    **kwargs
  ):
    super(LastToCurrent, self).__init__(**kwargs)
    self.nb_elem = 1
    self.indices = indices
    if (self.indices is not None):
      self.indices = tf.constant(self.indices, dtype=tf.int32)
      self.indices = tf.reshape(self.indices, shape=[-1])
      self.nb_elem = tf.math.reduce_max(self.indices)+1

  # Calling
  # ===================================
  def call(self, x, training=None):
    y = x
    if (self.indices is not None):
      y = tf.transpose(y)
      y = tf.math.unsorted_segment_sum(
        data=y,
        segment_ids=self.indices,
        num_segments=self.nb_elem
      )
      y = tf.transpose(y)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "indices": ops.to_numpy(self.indices)
    }
    base_config = super(LastToCurrent, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class CurrentToNext(nn_elem.Layer):
  """Current-to-next timescale connectivity."""

  # Initialization
  # ===================================
  def __init__(
    self,
    indices={},
    **kwargs
  ):
    super(CurrentToNext, self).__init__(**kwargs)
    self.indices = tf.nest.map_structure(
      lambda x: tf.reshape(tf.constant(x, dtype=tf.int32), shape=[-1]),
      indices
    )

  # Calling
  # ===================================
  def call(self, x, training=None):
    y = x[0]
    for i in range(len(x)-1):
      y = x[i+1] * tf.gather(y, indices=self.indices[str(i)], axis=-1)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "indices": tf.nest.map_structure(ops.to_numpy, self.indices)
    }
    base_config = super(CurrentToNext, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class GroupsConnectivity(object):
  """
  Example:
  -------
    graph = {
      0: {
        0: [0,1,2],
        1: [3,4,5],
        2: [6,7,8]
      }
    }
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    graph
  ):
    self.graph = graph
    if isinstance(self.graph, str):
      with open(self.graph) as file:
        self.graph = json.load(file)
    # Current-to-next timescale connectivity/indexing
    self.current_to_next = {
      "conn": self._convert_keys(self.connect_current_to_next(self.graph))
    }
    self.current_to_next["indices"] = self.get_indices(
      self.current_to_next["conn"]
    )
    # Last-to-current timescale connectivity/indexing
    self.last_to_current = {
      "conn": self._convert_keys(self.connect_last_to_current(self.graph))
    }
    self.last_to_current["indices"] = self.get_indices(
      self.last_to_current["conn"]
    )

  def _convert_keys(self, d):
    return {str(k): v for (k, v) in d.items()}

  # Groups graph indexing
  # ===================================
  def get_indices(self, connectivity):
    indices = {}
    for (l, lindices) in connectivity.items():
      if (lindices is not None):
        lindices = [np.full(len(idx), j) for (j,idx) in enumerate(lindices)]
        lindices = np.concatenate(lindices)
      indices[l] = lindices
    return indices

  # Groups graph connectivity
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
