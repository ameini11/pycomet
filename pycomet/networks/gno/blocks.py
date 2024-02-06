import tensorflow as tf

from . import utils as gutils
from .. import elements as nn_elem
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet import utils as pcm_utils
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GraphNet(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size={},
    activation={},
    kernel_initializer=None,
    regularizer=None,
    dropout=dict(rate=0),
    layer_norm=None,
    static_edge_index=None,
    reduction="sum",
    **kwargs
  ):
    super(GraphNet, self).__init__(**kwargs)
    # Hyperparameters
    # -----------------------
    self.layer_size = self.check_dict("layer_size", layer_size)
    self.activation = self.check_dict("activation", activation)
    self.kernel_initializer = kernel_initializer
    self.regularizer = regularizer
    self.dropout = dropout
    self.layer_norm = layer_norm
    self.reduction = reduction
    self.static_edge_index = gutils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )
    # Kernel
    # -----------------------
    common = dict(
      kernel_initializer=self.kernel_initializer,
      regularizer=self.regularizer,
      dropout=self.dropout,
      layer_norm=self.layer_norm,
      static_edge_index=self.static_edge_index
    )
    self.edge_blk = GraphEdgeBlock(
      layer_size=self.layer_size["edge"],
      activation=self.activation["edge"],
      **common
    )
    self.node_blk = GraphNodeBlock(
      layer_size=self.layer_size["node"],
      activation=self.activation["node"],
      reduction=self.reduction,
      **common
    )

  def check_dict(self, key, value):
    if (not isinstance(value, dict)):
      pcm_utils.raise_err(
        ValueError,
        f"The '{key}' input provided to '{self.name}' must be " \
        "a dictionary with keys ('edge', 'node') for " \
        "constructing the respective MLP transformation."
      )
    return value

  # Calling
  # ===================================
  def call(self, graph, training=False):
    graph_new = tf.nest.map_structure(tf.identity, graph)
    # Apply edge transformation
    graph_new[-1] = self.edge_blk(graph_new, training=training) + graph[-1]
    # Apply node transformation
    graph_new[ 0] = self.node_blk(graph_new, training=training) + graph[ 0]
    return graph_new

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "layer_size": self.layer_size,
      "activation": self.activation,
      "kernel_initializer": self.kernel_initializer,
      "regularizer": self.regularizer,
      "dropout": self.dropout,
      "layer_norm": self.layer_norm,
      "static_edge_index": ops.to_numpy(self.static_edge_index),
      "reduction": self.reduction
    }
    base_config = super(GraphNet, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GraphEdgeBlock(nn_elem.FFBlock):

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=["linear"],
    kernel_initializer=None,
    regularizer=None,
    dropout=dict(rate=0),
    layer_norm=None,
    static_edge_index=None,
    **kwargs
  ):
    super(GraphEdgeBlock, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    self.static_edge_index = gutils.read_edge_index(
      data=static_edge_index,
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
    # Transform
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
      "static_edge_index": ops.to_numpy(self.static_edge_index)
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
    activation=["linear"],
    kernel_initializer=None,
    regularizer=None,
    dropout=dict(rate=0),
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
      dropout=dropout,
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
    rank = len(edge_attr.shape)
    if (rank not in (2,3)):
      pcm_utils.raise_err(
        ValueError,
        "The 'edge_attr' input tensor has to be of rank 2 " \
        "(shape [nb_edges, nb_edge_features]) or 3 " \
        "(shape [nb_samples, nb_edges, nb_edge_features])."
      )
    # Move batches as last dimension
    if (rank == 3):
      edge_attr = tf.transpose(edge_attr, perm=[1,2,0])
    # Aggregate
    edge_attr_aggr = self.reduction_fun(
      data=edge_attr,
      segment_ids=edge_index[1],            # Receivers
      num_segments=node_attr.shape[-2]      # Number of nodes
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
      "reduction": self.reduction
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
    activation=["linear"],
    kernel_initializer=None,
    regularizer=None,
    dropout=dict(rate=0),
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
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    self.static_edge_index = gutils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )
    self.static_edge_attr = gutils.read_attr(
      data=static_edge_attr,
      identifier="static_edge_attr"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, edge_index, edge_attr = gutils.unpack_graph(
      graph, self.static_edge_index, self.static_edge_attr
    )
    # Call mlp
    edge_attr = super(EdgeEncoder, self).call(edge_attr, training=training)
    # Manipulate edges
    if (self.static_edge_attr is not None):
      edge_attr = tf.repeat(
        input=edge_attr,
        repeats=tf.shape(node_attr)[0],
        axis=0
      )
      # if (nb_samples is not None):
      #   multiples = tf.nest.map_structure(lambda _: 1, list(edge_attr.shape))
      #   multiples[0] = nb_samples
      #   edge_attr = tf.tile(edge_attr, multiples)
      # else:
      #   pcm_utils.raise_err(
      #     ValueError,
      #     "The 'static_edge_attr' input tensor is incompatible with " \
      #     "dynamic input tensors, where the first dimension is set to 'None'."
      #   )
    # Return graph
    return self._return_graph(node_attr, edge_index, edge_attr)

  def _return_graph(self, node_attr, edge_index, edge_attr):
    if (self.static_edge_index is not None):
      return [node_attr, edge_attr]
    else:
      return [node_attr, edge_index, edge_attr]

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_edge_index": ops.to_numpy(self.static_edge_index),
      "static_edge_attr": ops.to_numpy(self.static_edge_attr)
    }
    base_config = super(EdgeEncoder, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
