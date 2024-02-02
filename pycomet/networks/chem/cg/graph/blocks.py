import tensorflow as tf

from pycomet.networks import gno
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearMEGraphNet(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    static_edge_index=None,
    **kwargs
  ):
    super(LinearMEGraphNet, self).__init__(**kwargs)
    self.message = gno.LinearMessage(
      origin="sender",
      reduction="prod",
      include_edge=True,
      static_edge_index=static_edge_index
    )
    self.aggregate = gno.Aggregate(
      fluxes="both",
      reduction="sum"
    )
    self.update = gno.LinearUpdate(
      reduction="diff"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Edges - Message passing
    graph = self.message(graph, training=training)
    # Edges - Aggregation
    graph = self.aggregate(graph, training=training)
    # Nodes - Updating
    return self.update(graph, training=training)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_edge_index": self.static_edge_index
    }
    base_config = super(LinearMEGraphNet, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


# @tf.keras.utils.register_keras_serializable(package=_CFG.name)
# class NonLinearMEGraphNet(nn_elem.Layer):

#   # Initialization
#   # ===================================
#   def __init__(
#     self,
#     layer_size=None,
#     activation=["linear"],
#     kernel_initializer=None,
#     regularizer=None,
#     dropout=dict(rate=0),
#     layer_norm=None,
#     static_edge_index=None,
#     log_xform=None,
#     **kwargs
#   ):
#     super(NonLinearMEGraphNet, self).__init__(**kwargs)
#     # Hyperparameters
#     # -----------------------
#     self.layer_size = layer_size
#     self.activation = activation
#     self.kernel_initializer = kernel_initializer
#     self.regularizer = regularizer
#     self.dropout = dropout
#     self.layer_norm = layer_norm
#     self.log_xform = log_xform
#     self.static_edge_index = gutils.read_edge_index(
#       data=static_edge_index,
#       identifier="static_edge_index"
#     )
#     # Kernel
#     # -----------------------
#     self.message = glay.NonLinearMessagePassing(
#       layer_size=self.layer_size,
#       activation=self.activation,
#       kernel_initializer=self.kernel_initializer,
#       regularizer=self.regularizer,
#       dropout=self.dropout,
#       layer_norm=self.layer_norm,
#       static_edge_index=self.static_edge_index,
#       log_xform=self.log_xform,
#       **kwargs
#     )
#     self.aggregate = glay.LinearAggregate(
#       static_edge_index=self.static_edge_index,
#       **kwargs
#     )

#   # Calling
#   # ===================================
#   def call(self, graph, training=False):
#     graph_new = tf.nest.map_structure(tf.identity, graph)
#     # Apply edge transformation
#     graph_new[-1] = self.message(graph, training=training)
#     # Nodes - Aggregation
#     return self.aggregate(graph_new, training=training)

#   # Configuration update
#   # ===================================
#   def get_config(self):
#     config = {
#       "layer_size": self.layer_size,
#       "activation": self.activation,
#       "kernel_initializer": self.kernel_initializer,
#       "regularizer": self.regularizer,
#       "dropout": self.dropout,
#       "layer_norm": self.layer_norm,
#       "static_edge_index": ops.to_numpy(self.static_edge_index),
#       "log_xform": self.log_xform
#     }
#     base_config = super(NonLinearMEGraphNet, self).get_config()
#     return dict(list(base_config.items()) + list(config.items()))
