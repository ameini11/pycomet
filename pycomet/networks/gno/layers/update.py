import tensorflow as tf

from pycomet.networks import gno
from keras import layers as tf_layers
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()


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
    # Merging 'outgoing' (index 0) and 'incoming' (index 1) fluxes
    self.reduction = reduction
    self.red_coeff = {
      "sum": [1,1],
      "diff": [-1,1]
    }[self.reduction]

  # Calling
  # ===================================
  def call(self, graph, training=False):
    # Unpack graph
    node_attr, *_ = gno.utils.unpack_graph(graph)
    # Combine fluxes on nodes
    if (node_attr.shape[-1] > 1):
      fluxes = tf.split(node_attr, 2, axis=-1)
      fluxes = [f * self.red_coeff[i] for (i, f) in enumerate(fluxes)]
      return tf_layers.Add()(fluxes)
    else:
      return node_attr

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "reduction": self.reduction
    }
    base_config = super(LinearUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
