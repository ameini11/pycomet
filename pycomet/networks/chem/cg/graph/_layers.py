import tensorflow as tf

from keras import layers as tf_layers
from pycomet.networks import gno
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet import utils as pcm_utils
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class NonLinearMessagePassing(gno.GraphEdgeBlock):

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
    log_xform=None,
    **kwargs
  ):
    super(NonLinearMessagePassing, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      static_edge_index=static_edge_index,
      **kwargs
    )
    self.log_xform = log_xform
    if (self.log_xform is not None):
      self.log = nn_elem.Log(**log_xform)

  # Calling
  # ===================================
  def _gather_features(self, node_attr, edge_index, edge_attr):
    features = [
      tf.gather(node_attr, edge_index[0], axis=-2), # Gather senders
      edge_attr
    ]
    features = tf.concat(features, axis=-1)
    if (self.log_xform is not None):
      features = self.log(features)
    return features

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "log_xform": self.log_xform
    }
    base_config = super(NonLinearMessagePassing, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearMessagePassing(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    starting="senders",
    reduction="prod",
    **kwargs
  ):
    super(LinearMessagePassing, self).__init__(**kwargs)
    self.starting = starting
    self.gather = GatherMessage(self.starting)
    self.reduction = reduction
    self.reduction_fun = {
      "sum": tf.math.reduce_sum,
      "prod": tf.math.reduce_prod
    }[self.reduction]

  # Calling
  # ===================================
  def call(self, graph, training=False):
    features = self.gather(graph)
    return self.reduction_fun(features, axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "starting": self.starting,
      "reduction": self.reduction
    }
    base_config = super(LinearMessagePassing, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GatherMessage(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    starting="senders",
    **kwargs
  ):
    super(GatherMessage, self).__init__(**kwargs)
    self.starting = starting
    self.starting_idx = {
      "senders": [0],
      "receivers": [1],
      "both": [0,1]
    }[self.starting]

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, edge_index, edge_attr = graph
    # Gather features
    features = []
    for i in self.starting_idx:
      features.append(tf.gather(node_attr, edge_index[i], axis=-2))
    features.append(edge_attr)
    # Manipulate features
    return tf.concat(features, axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "starting": self.starting
    }
    base_config = super(GatherMessage, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Aggregate(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    fluxes="both",
    **kwargs
  ):
    super(Aggregate, self).__init__(**kwargs)
    self.fluxes = fluxes
    # > incoming: gather receivers (index 1)
    # > outgoing: gather senders (index 0)
    self.fluxes_idx = {
      "in": [1],
      "out": [0],
      "both": [0,1]
    }[self.fluxes]

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, edge_index, edge_attr = graph
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
    # Compute fluxes by aggregation
    fluxes = [tf.math.unsorted_segment_sum(
      data=edge_attr,
      segment_ids=edge_index[i],
      num_segments=node_attr.shape[-2]
    ) for i in self.fluxes_idx]
    # Move batches as first dimension
    if (rank == 3):
      fluxes = [tf.transpose(f, perm=[2,0,1]) for f in fluxes]
    # Concatenate if needed
    return tf.concat(fluxes, axis=-1) if (len(fluxes) > 1) else fluxes[-1]

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "fluxes": self.fluxes
    }
    base_config = super(Aggregate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearUpdate(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    reduction="diff",
    **kwargs
  ):
    super(LinearUpdate, self).__init__(**kwargs)
    self.reduction = reduction
    self.reduction_fun = {
      "sum": lambda fout, fin: fin + fout,
      "diff": lambda fout, fin: fin - fout
    }[self.reduction]

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Extract fluxes from 'edge_attr'
    fluxes = graph[-1]
    # Combine fluxes on nodes
    if (fluxes.shape[-1] > 1):
      fluxes = tf.split(fluxes, 2, axis=-1)
      return self.reduction_fun(*fluxes)
    else:
      return fluxes

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "reduction": self.reduction
    }
    base_config = super(LinearUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
