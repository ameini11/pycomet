import tensorflow as tf

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
    static_edge_index=None,
    **kwargs
  ):
    super(LinearMessagePassing, self).__init__(**kwargs)
    self.static_edge_index = gno.utils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    graph = gno.utils.unpack_graph(graph, self.static_edge_index, None)
    # Edges - Message passing
    return self.message(*graph)

  def message(self, node_attr, edge_index, edge_attr):
    # Compute fluxes
    # -----------------------
    # > node_attr: number densities (n_i)
    # > edge_attr: rate constants (k_ij*n_atom)
    # > flux: n_i * (k_ij*n_atom)
    senders_attr = tf.gather(node_attr, edge_index[0], axis=-2)
    return senders_attr * edge_attr


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class GatherMessage(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    directions=[0], # Senders or receivers or both
    static_edge_index=None,
    **kwargs
  ):
    super(GatherMessage, self).__init__(**kwargs)
    self.directions = directions
    self.static_edge_index = gno.utils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, edge_index, edge_attr = gno.utils.unpack_graph(
      graph, self.static_edge_index, None
    )
    # Gather features
    features = []
    for d in self.directions:
      features.append(tf.gather(node_attr, edge_index[d], axis=-2))
    features.append(edge_attr)
    # Manipulate features
    return tf.concat(features, axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "directions": self.directions,
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(GatherMessage, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
  

@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearAggregate(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    static_edge_index=None,
    **kwargs
  ):
    super(LinearAggregate, self).__init__(**kwargs)
    self.static_edge_index = gno.utils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    graph = gno.utils.unpack_graph(graph, self.static_edge_index, None)
    # Nodes - Aggregation
    graph[0] = self.aggregate(*graph)
    # Nodes - Updating
    return self.update(*graph)

  def aggregate(self, node_attr, edge_index, edge_attr):
    # Aggregate fluxes
    # -----------------------
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
    # > incoming: gather receivers (index 1)
    # > outgoing: gather senders (index 0)
    fluxes = [tf.math.unsorted_segment_sum(
      data=edge_attr,
      segment_ids=edge_index[i],
      num_segments=node_attr.shape[-2]
    ) for i in range(2)]
    # Move batches as first dimension
    if (rank == 3):
      fluxes = [tf.transpose(f, perm=[2,0,1]) for f in fluxes]
    return tf.concat(fluxes, axis=-1)

  def update(self, node_attr, edge_index, edge_attr):
    # Update nodes
    # -----------------------
    fin = tf.gather(node_attr, [1], axis=-1)
    fout = tf.gather(node_attr, [0], axis=-1)
    return fin - fout

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(LinearAggregate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
