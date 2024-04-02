import copy
import tensorflow as tf

from .utils import read_static_prop
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Correction(nn_elem.Layer):

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
    static_prop=None,
    **kwargs
  ):
    super(Correction, self).__init__(**kwargs)
    # Static state/group-specific physical properties
    self.static_prop = tf.constant(read_static_prop(static_prop))
    # Blocks for encoding 'time' and 'energy' spaces
    # > Hyperparameters
    self.layer_size = layer_size
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.regularizer = regularizer
    self.dropout = dropout
    self.layer_norm = layer_norm
    # > Initialization
    self.blocks = {}
    for name in ("time", "energy"):
      layer_size = copy.deepcopy(self.layer_size)
      if (name == "energy"):
        layer_size[-1] += 2
      self.blocks[name] = nn_elem.FFBlock(
        layer_size=layer_size,
        activation=self.activation,
        kernel_initializer=self.kernel_initializer,
        regularizer=self.regularizer,
        layer_norm=self.layer_norm,
        dropout=self.dropout
      )

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Encode 'time' inputs
    # -------------
    ft = self.blocks["time"](inputs, training=training)
    # Encode static 'energy' inputs
    # -------------
    fe = self.blocks["energy"](self.static_prop, training=training)
    fe, d, c = tf.split(fe, num_or_size_splits=[fe.shape[1]-2,1,1], axis=-1)
    # > Scaling
    fe *= d
    # > Reshape shifting
    c = tf.expand_dims(c, axis=0)
    c = tf.repeat(input=c, repeats=tf.shape(ft)[0], axis=0)
    # Linerly combine latent spaces
    # -------------
    y = tf.einsum("np,mp->nmp", ft, fe)
    y = tf.concat([y,c], axis=-1)
    return y

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
      "static_prop": ops.to_numpy(self.static_prop)
    }
    base_config = super(Correction, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
