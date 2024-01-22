import tensorflow as tf

from pycomet import utils
from . import utils as gutils
from .. import elements as nn_elem
from pycomet import operations as ops
from pycomet.config import get_config
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GraphEdgeBlock(nn_elem.FFBlock):

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=['linear'],
    kernel_initializer=None,
    regularizer=None,
    dropout_kwargs=dict(rate=0),
    batch_norm=None,
    layer_norm=None,
    static_edge_index=None,
    **kwargs
  ):
    super(GraphEdgeBlock, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout_kwargs=dropout_kwargs,
      batch_norm=batch_norm,
      layer_norm=layer_norm,
      **kwargs
    )
    self.static_edge_index = gutils.read_edge_index(
      edge_index=static_edge_index,
      identifier="static_edge_index"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, edge_index, edge_attr = gutils.unpack_graph(
      graph, self.static_edge_index, None
    )
    # Gather features
    features = self._gather_features(node_attr, edge_index, edge_attr)
    return super(GraphEdgeBlock, self).call(features, training=training)

  def _gather_features(self, node_attr, edge_index, edge_attr):
    features = [
      tf.gather(node_attr, edge_index[0], axis=-2), # Gather senders
      tf.gather(node_attr, edge_index[1], axis=-2), # Gather receivers
      edge_attr
    ]
    return tf.concat(features, axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'static_edge_index': ops.to_numpy(self.static_edge_index)
    }
    base_config = super(GraphEdgeBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GraphNodeBlock(GraphEdgeBlock):

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=['linear'],
    kernel_initializer=None,
    regularizer=None,
    dropout_kwargs=dict(rate=0),
    batch_norm=None,
    layer_norm=None,
    static_edge_index=None,
    reduction="sum",
    **kwargs
  ):
    super(GraphNodeBlock, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout_kwargs=dropout_kwargs,
      batch_norm=batch_norm,
      layer_norm=layer_norm,
      static_edge_index=static_edge_index,
      **kwargs
    )
    self.reduction = reduction
    self.reduction_fun = {
      "max": tf.math.unsorted_segment_max,
      "mean": tf.math.unsorted_segment_mean,
      "min": tf.math.unsorted_segment_min,
      "prod": tf.math.unsorted_segment_prod,
      "sqrt_n": tf.math.unsorted_segment_sqrt_n,
      "sum": tf.math.unsorted_segment_sum
    }[self.reduction]

  # Calling
  # ===================================
  def _gather_features(self, node_attr, edge_index, edge_attr):
    # Check rank
    rank = tf.rank(edge_attr)
    if (rank not in (2,3)):
      utils.raise_err(
        ValueError,
        "The 'edge_attr' input tensor has to be of rank 2 or 3."
      )
    # Move batches as last dimension
    if (rank == 3):
      edge_attr = tf.transpose(edge_attr, perm=[1,2,0])
    # Aggregate
    edge_attr_aggr = self.reduction_fun(
      data=edge_attr,
      segment_ids=edge_index[1],            # Receivers
      num_segments=tf.shape(node_attr)[-2]  # Number of nodes
    )
    # Move batches as first dimension
    if (rank == 3):
      edge_attr_aggr = tf.transpose(edge_attr_aggr, perm=[2,0,1])
    # Return features
    return tf.concat([node_attr, edge_attr_aggr], axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'reduction': self.reduction
    }
    base_config = super(GraphNodeBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class EdgeEncoder(nn_elem.FFBlock):

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=['linear'],
    kernel_initializer=None,
    regularizer=None,
    dropout_kwargs=dict(rate=0),
    batch_norm=None,
    layer_norm=None,
    static_edge_index=None,
    static_edge_attr=None,
    **kwargs
  ):
    super(EdgeEncoder, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout_kwargs=dropout_kwargs,
      batch_norm=batch_norm,
      layer_norm=layer_norm,
      **kwargs
    )
    self.static_edge_index = gutils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )
    self.static_edge_attr = gutils.read_edge_attr(
      data=static_edge_attr,
      identifier="static_edge_attr"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, _, edge_attr = gutils.unpack_graph(
      graph, self.static_edge_index, self.static_edge_attr
    )
    # Call mlp
    edge_attr = super(EdgeEncoder, self).call(edge_attr, training=training)
    # Manipulate edges
    if (self.static_edge_attr is not None):
      multiples = tf.nest.map_structure(lambda _: 1, tf.shape(edge_attr))
      multiples[0] = tf.shape(node_attr)[0]
      edge_attr = tf.tile(edge_attr, multiples)
    # Return edge attributes
    return edge_attr

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'static_edge_index': ops.to_numpy(self.static_edge_index),
      'static_edge_attr': ops.to_numpy(self.static_edge_attr)
    }
    base_config = super(EdgeEncoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class NodeEncoder(nn_elem.FFBlock):

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=['linear'],
    kernel_initializer=None,
    regularizer=None,
    dropout_kwargs=dict(rate=0),
    batch_norm=None,
    layer_norm=None,
    **kwargs
  ):
    super(NodeEncoder, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout_kwargs=dropout_kwargs,
      batch_norm=batch_norm,
      layer_norm=layer_norm,
      **kwargs
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr = tf.nest.flatten(graph)[0]
    # Call mlp
    return super(NodeEncoder, self).call(node_attr, training=training)
