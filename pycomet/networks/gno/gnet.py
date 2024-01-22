import tensorflow as tf

from pycomet import utils
from . import utils as gutils
from . import blocks as gblocks
from .. import elements as nn_elem
from pycomet import operations as ops
from pycomet.config import get_config
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GraphNet(nn_elem.Layer):

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
    super(GraphNet, self).__init__(**kwargs)
    # Hyperparameters
    # -----------------------
    self.layer_size = self.check_dict("layer_size", layer_size)
    self.activation = self.check_dict("activation", activation)
    self.kernel_initializer = kernel_initializer
    self.regularizer = regularizer
    self.dropout_kwargs = dropout_kwargs
    self.batch_norm = batch_norm
    self.layer_norm = layer_norm
    self.reduction = reduction
    self.static_edge_index = gutils.read_edge_index(
      edge_index=static_edge_index,
      identifier="static_edge_index"
    )
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
    self.edge_blk = gblocks.GraphEdgeBlock(
      layer_size=self.layer_size["edge"],
      activation=self.activation["edge"],
      **common
    )
    self.node_blk = gblocks.GraphNodeBlock(
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
      'layer_size': self.layer_size,
      'activation': self.activation,
      'kernel_initializer': self.kernel_initializer,
      'regularizer': self.regularizer,
      'dropout_kwargs': self.dropout_kwargs,
      'batch_norm': self.batch_norm,
      'layer_norm': self.layer_norm,
      'static_edge_index': ops.to_numpy(self.static_edge_index),
      'reduction': self.reduction
    }
    base_config = super(GraphNet, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
