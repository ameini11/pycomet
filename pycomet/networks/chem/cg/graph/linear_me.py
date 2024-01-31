import tensorflow as tf

from pycomet import utils
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
from pycomet.networks.gno import utils as gutils
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
    self.static_edge_index = gutils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )
    self.edge_blk = LinearMEGraphEdgeBlock(
      static_edge_index=self.static_edge_index
      **kwargs
    )
    self.node_blk = LinearMEGraphNodeBlock(
      static_edge_index=self.static_edge_index
      **kwargs
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Apply edge transformation
    graph[-1] = self.edge_blk(graph, training=training)
    # Apply node transformation
    return self.node_blk(graph, training=training)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(LinearMEGraphNet, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearMEGraphEdgeBlock(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    static_edge_index=None,
    **kwargs
  ):
    super(LinearMEGraphEdgeBlock, self).__init__(**kwargs)
    self.static_edge_index = gutils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    # -----------------------
    node_attr, edge_index, edge_attr = gutils.unpack_graph(
      graph, self.static_edge_index, None
    )
    # Compute fluxes
    # -----------------------
    # > node_attr: number densities (n_i)
    # > edge_attr: rate constants (k_ij*n_atom)
    # > flux: n_i * (k_ij*n_atom)
    senders_attr = tf.gather(node_attr, edge_index[0], axis=-2)
    return senders_attr * edge_attr

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(LinearMEGraphEdgeBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class LinearMEGraphNodeBlock(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    static_edge_index=None,
    **kwargs
  ):
    super(LinearMEGraphNodeBlock, self).__init__(**kwargs)
    self.static_edge_index = gutils.read_edge_index(
      data=static_edge_index,
      identifier="static_edge_index"
    )

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    # -----------------------
    node_attr, edge_index, edge_attr = gutils.unpack_graph(
      graph, self.static_edge_index, None
    )
    # Aggregate fluxes
    # -----------------------
    # Check rank
    rank = len(edge_attr.shape)
    if (rank not in (2,3)):
      utils.raise_err(
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
    # Update nodes
    # -----------------------
    return fluxes[1] - fluxes[0]

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_edge_index": ops.to_numpy(self.static_edge_index)
    }
    base_config = super(LinearMEGraphNodeBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
