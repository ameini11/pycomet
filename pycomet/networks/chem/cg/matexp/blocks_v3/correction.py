import copy
import tensorflow as tf

from .utils import read_prop
from pycomet import backend as bkd
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
    state_prop=None,
    **kwargs
  ):
    super(Correction, self).__init__(**kwargs)
    # Static state-specific properties
    self.state_prop = read_prop(state_prop)
    self.state_prop = tf.constant(self.state_prop, dtype=bkd.get_floatx())
    # Blocks for encoding 'time' and 'state' spaces
    # > Hyperparameters
    self.layer_size = layer_size
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.regularizer = regularizer
    self.dropout = dropout
    self.layer_norm = layer_norm
    # > Initialization
    self.blocks = {}
    for name in ("time", "state"):
      layer_size_i = copy.deepcopy(self.layer_size)
      if (name == "state"):
        layer_size_i[-1] += 2
      self.blocks[name] = nn_elem.FFBlock(
        layer_size=layer_size_i,
        activation=self.activation,
        kernel_initializer=self.kernel_initializer,
        regularizer=self.regularizer,
        layer_norm=self.layer_norm,
        dropout=self.dropout
      )
    # > Latent space
    self.latent = self.layer_size[-1]
    self.splits = [self.latent, 1, 1]

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Encode 'time' inputs > shape = [n,1,p]
    # -------------
    yt = self.blocks["time"](inputs, training=training)
    yt = tf.expand_dims(yt, axis=1)
    # Encode static state-specific properties > shape = [1,m,p+2]
    # -------------
    ys = self.blocks["state"](self.state_prop, training=training)
    ys = tf.expand_dims(ys, axis=0)
    # > Splitting
    ys, d, c = tf.split(ys, num_or_size_splits=self.splits, axis=-1)
    # > Scaling
    ys *= d
    # > Shifting
    c = tf.repeat(input=c, repeats=tf.shape(yt)[0], axis=0)
    # Linerly combine latent spaces
    # -------------
    return tf.concat([yt*ys,c], axis=-1)

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
      "state_prop": ops.to_numpy(self.state_prop)
    }
    base_config = super(Correction, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
