import tensorflow as tf

from pycomet.networks import gno
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet import utils as pcm_utils
from pycomet.networks import elements as nn_elem
_CFG = get_config()


class NonLinearMessage(nn_elem.MixedLaysBlock):

  # Initialization
  # ===================================
  def __init__(
    self,
    layers,
    **kwargs
  ):
    if any([k not in layers for k in ("gather", "nonlin")]):
      pcm_utils.raise_err(
        ValueError,
        "The 'NonLinearMessage' block necessitates the " \
        "specification of both 'gather' (e.g., 'GatherMessage') and " \
        "'nonlin' (i.e., a generic nonlinear function) layers."
      )
    super(NonLinearMessage, self).__init__(layers, **kwargs)

  # Calling
  # ===================================
  def call(self, graph, training=False):
    graph = self.layers["gather"](graph, training=training)
    for k in ("xform", "nonlin"):
      if (k in self.layers):
        graph[1] = self.layers[k](graph[1], training=training)
    return graph


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearMessage(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    origin="sender",
    reduction="prod",
    include_edge=True,
    node_attr_indices=None,
    edge_attr_indices=None,
    static_edge_index=None,
    **kwargs
  ):
    super(LinearMessage, self).__init__(**kwargs)
    self.origin = origin
    self.include_edge = include_edge
    self.node_attr_indices = node_attr_indices
    self.edge_attr_indices = edge_attr_indices
    self.static_edge_index = gno.utils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )
    self.gather_lay = GatherMessage(
      origin=self.origin,
      include_edge=self.include_edge,
      node_attr_indices=self.node_attr_indices,
      edge_attr_indices=self.edge_attr_indices,
      static_edge_index=self.static_edge_index
    )
    self.reduction = reduction
    self.reduction_fun = {
      "sum": tf.math.reduce_sum,
      "prod": tf.math.reduce_prod
    }[self.reduction]

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Gather messages
    node_attr, edge_attr, cond_attr, edge_index = self.gather_lay(graph)
    # Reduce over 'edge_attr'
    edge_attr = self.reduction_fun(edge_attr, axis=-1, keepdims=True)
    # Return graph
    return gno.utils.return_graph(node_attr, edge_attr, cond_attr, edge_index)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "origin": self.origin,
      "reduction": self.reduction,
      "include_edge": self.include_edge,
      "node_attr_indices": self.node_attr_indices,
      "edge_attr_indices": self.edge_attr_indices,
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(LinearMessage, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GatherMessage(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    origin="both",
    include_edge=True,
    node_attr_indices=None,
    edge_attr_indices=None,
    static_edge_index=None,
    **kwargs
  ):
    super(GatherMessage, self).__init__(**kwargs)
    # Define where the information comes from
    self.origin = origin
    self.origin_index = {
      "sender": [0],    # Only from the sender node
      "receiver": [1],  # Only from the receiver node
      "both": [0,1]     # From both
    }[self.origin]
    # Include edge information
    self.include_edge = include_edge
    # Extract specific node/edge features
    self.node_attr_indices = node_attr_indices
    if (self.node_attr_indices is not None):
      self.node_attr_indices = tf.nest.flatten(self.node_attr_indices)
    self.edge_attr_indices = edge_attr_indices
    if (self.edge_attr_indices is not None):
      self.edge_attr_indices = tf.nest.flatten(self.edge_attr_indices)
    # Static edge index
    self.static_edge_index = gno.utils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, edge_attr, cond_attr, edge_index = gno.utils.unpack_graph(
      graph=graph, edge_index=self.static_edge_index
    )
    # Extract specific node/edge features
    if (self.node_attr_indices is not None):
      node_attr = tf.gather(node_attr, self.node_attr_indices, axis=-1)
    if (self.edge_attr_indices is not None):
      edge_attr = tf.gather(edge_attr, self.edge_attr_indices, axis=-1)
    # Gather features
    features = []
    for i in self.origin_index:
      features.append(tf.gather(node_attr, edge_index[i], axis=-2))
    if self.include_edge:
      features.append(edge_attr)
    # Compose edge attributes
    edge_attr = tf.concat(features, axis=-1)
    # Return graph
    return gno.utils.return_graph(node_attr, edge_attr, cond_attr, edge_index)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "origin": self.origin,
      "include_edge": self.include_edge,
      "node_attr_indices": self.node_attr_indices,
      "edge_attr_indices": self.edge_attr_indices,
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(GatherMessage, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
