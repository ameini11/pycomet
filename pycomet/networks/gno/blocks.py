import numpy as np
import tensorflow as tf

from pycomet import utils
from . import utils as gutils
from .. import elements as nn_elem
from pycomet import operations as ops
from pycomet.config import get_config
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class EdgeBlock(nn_elem.FFBlock):

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
    super(EdgeBlock, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout_kwargs=dropout_kwargs,
      batch_norm=batch_norm,
      layer_norm=layer_norm,
      **kwargs
    )
    self.static_edge_index = static_edge_index
    if (self.static_edge_index is not None):
      self.static_edge_index = tf.constant(
        ops.to_tensor(self.static_edge_index), dtype=tf.int32
      )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, edge_index, edge_attr = gutils.unpack_graph(
      graph, self.static_edge_index
    )
    # Collect features
    features = self._collect_features(node_attr, edge_index, edge_attr)
    features = tf.concat(features, axis=-1)
    # Call MLP
    edge_attr = super(EdgeBlock, self).call(features, training=training)
    # Return graph
    if (self.static_edge_index is not None):
      return [node_attr, edge_attr]
    else:
      return [node_attr, edge_index, edge_attr]

  def _collect_features(self, node_attr, edge_index, edge_attr):
    return [
      tf.gather(node_attr, edge_index[0], axis=-2), # Collect senders
      tf.gather(node_attr, edge_index[1], axis=-2), # Collect receivers
      edge_attr
    ]

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'static_edge_index': ops.to_numpy(self.static_edge_index)
    }
    base_config = super(EdgeBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class NodeBlock(EdgeBlock):

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
    super(NodeBlock, self).__init__(
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
  def _collect_features(self, node_attr, edge_index, edge_attr):
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
    return [node_attr, edge_attr_aggr]

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'reduction': self.reduction
    }
    base_config = super(NodeBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GraphNetBlock(nn_elem.Layer):

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
    super(GraphNetBlock, self).__init__(**kwargs)
    # Hyperparameters
    # -----------------------
    self.check_dict("layer_size", layer_size)
    self.check_dict("activation", activation)
    self.layer_size = layer_size
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.regularizer = regularizer
    self.dropout_kwargs = dropout_kwargs
    self.batch_norm = batch_norm
    self.layer_norm = layer_norm
    self.static_edge_index = self.read_edge_index(static_edge_index)
    self.reduction = reduction
    # Kernel
    # -----------------------
    common = dict(
      kernel_initializer=self.kernel_initializer,
      regularizer=self.regularizer,
      dropout_kwargs=self.dropout_kwargs,
      batch_norm=self.batch_norm,
      layer_norm=self.layer_norm,
      static_edge_index=self.static_edge_index
    )
    self.edge_xform = EdgeBlock(
      layer_size=self.layer_size["edge"],
      activation=self.activation["edge"],
      **common
    )
    self.node_xform = NodeBlock(
      layer_size=self.layer_size["node"],
      activation=self.activation["node"],
      reduction=self.reduction,
      **common
    )

  def check_dict(self, key, value):
    if (not isinstance(value, dict)):
      utils.raise_err(
        ValueError,
        f"The '{key}' input provided to {self.name} must be " \
        "a dictionary with keys ('edge', 'node') for constructing " \
        "the respective MLP transformation."
      )

  def read_edge_index(self, static_edge_index):
    if (static_edge_index is not None):
      if isinstance(static_edge_index, (np.ndarray, tf.tensor)):
        pass
      elif isinstance(static_edge_index, str):
        static_edge_index = np.loadtxt(static_edge_index).squeeze()
      else:
        utils.raise_err(
          ValueError,
          "The 'static_edge_index' input  must be either an array " \
          "or a string representing a valid file path for reading."
        )
      sh = static_edge_index.shape
      if ((len(sh) > 2) or (2 not in sh)):
        utils.raise_err(
          ValueError,
          "The 'static_edge_index' input must be a 2-dimensional " \
          "array with a shape of [2, nb_edges]."
        )
      elif (sh[1] == 2):
        static_edge_index = static_edge_index.T
      return static_edge_index

  # Calling
  # ===================================
  def call(self, graph, training=False):
    graph_old = graph
    # Apply edge/node transformation
    graph_new = self.edge_xform(graph_old, training=training)
    graph_new = self.node_xform(graph_new, training=training)
    # Add residual connections
    graph_new[0] += graph_old[0]
    graph_new[-1] += graph_old[-1]
    return graph_new

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'layer_size': self.layer_size,
      'activation': self.activation,
      'kernel_initializer': self.kernel_initializer,
      'regularizer': self.regularizer,
      'dropout_kwargs': self.dropout_kwargs,
      'batch_norm': self.batch_norm,
      'layer_norm': self.layer_norm,
      'static_edge_index': self.static_edge_index,
      'reduction': self.reduction
    }
    base_config = super(GraphNetBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
