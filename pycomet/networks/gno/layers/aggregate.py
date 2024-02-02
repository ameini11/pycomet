import tensorflow as tf

from pycomet.config import get_config
from pycomet import utils as pcm_utils
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Aggregate(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    fluxes="both",
    reduction="sum",
    **kwargs
  ):
    super(Aggregate, self).__init__(**kwargs)
    # Gather fluxes
    self.fluxes = fluxes
    self.fluxes_idx = {
      "in": [1],          # Incoming > gather receivers (index 1)
      "out": [0],         # Outgoing > gather senders (index 0)
      "both": [0,1]
    }[self.fluxes]
    # Reduction operation
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
    fluxes = [self.reduction_fun(
      data=edge_attr,
      segment_ids=edge_index[i],
      num_segments=node_attr.shape[-2]
    ) for i in self.fluxes_idx]
    # Move batches as first dimension
    if (rank == 3):
      fluxes = [tf.transpose(f, perm=[2,0,1]) for f in fluxes]
      edge_attr = tf.transpose(edge_attr, perm=[2,0,1])
    # Return graph
    if (len(fluxes) > 1):
      node_attr = tf.concat(fluxes, axis=-1)
    else:
      node_attr = fluxes[-1]
    return [node_attr, edge_index, edge_attr]

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "fluxes": self.fluxes,
      "reduction": self.reduction
    }
    base_config = super(Aggregate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
