import tensorflow as tf

from .. import _NN_MOD
from pycomet import utils
from . import activations
from . import initializers
from . import regularizers
from . import layers as pcm_layers
from pycomet.config import get_config
from keras import layers as tf_layers
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class MLP(pcm_layers.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer='GlorotNormal',
    regularizer=None,
    layer_norm=None,
    dropout=None,
    **kwargs
  ):
    super(MLP, self).__init__(**kwargs)
    # Hyperparameters initialization
    # -----------------------
    self.units = units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.regularizer = regularizers.get(**regularizer)
    self.layer_norm = layer_norm
    self.dropout = dropout
    # Layers initialization
    # -----------------------
    self.lays = []
    # Dense
    self.lays.append(
      tf_layers.Dense(
        units=self.units,
        activation=self.activation,
        use_bias=self.use_bias,
        kernel_initializer=self.kernel_initializer,
        kernel_regularizer=self.regularizer,
        bias_regularizer=self.regularizer,
        name=self.name+'_dense'
      )
    )
    # Normalization
    if self.layer_norm:
      self.lays.append(
        tf_layers.LayerNormalization(
          axis=-1,
          epsilon=1e-5,
          center=True,
          scale=True,
          beta_initializer='zeros',
          gamma_initializer='ones',
          beta_regularizer=self.regularizer,
          gamma_regularizer=self.regularizer,
          beta_constraint=None,
          gamma_constraint=None,
          name=self.name+'_layerNorm'
        )
      )
    # Dropout
    if self.dropout:
      self.lays.append(
        pcm_layers.Dropout(**dropout, name=self.name+'_dropout')
      )

  # Calling
  # ===================================
  def call(self, x, training=None):
    y = x
    for lay in self.lays:
      y = lay(y, training=training)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'units': self.units,
      'activation': self.activation,
      'use_bias': self.use_bias,
      'kernel_initializer': self.kernel_initializer,
      'regularizer': self.regularizer,
      'layer_norm': self.layer_norm,
      'dropout': self.dropout
    }
    base_config = super(MLP, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class FFBlock(pcm_layers.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=['linear'],
    kernel_initializer=None,
    regularizer=None,
    layer_norm=None,
    dropout=None,
    interpolating=None,
    **kwargs
  ):
    super(FFBlock, self).__init__(**kwargs)
    # Hyperparameters initialization
    # -----------------------
    self.layer_size = layer_size
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.regularizer = regularizer
    self.layer_norm = layer_norm
    self.dropout = dropout
    self.interpolating = interpolating
    # Checking
    c1 = all([isinstance(k, (list, tuple)) for k in (activation, layer_size)])
    c2 = (len(activation) != len(layer_size)-1)
    if ((not c1) or c2):
      utils.raise_err(
        ValueError,
        f"'activation' and 'layer_size' inputs passed to {self.name} " \
        "block have to be a 'list'/'tuple' of lengths: " \
        "len(activation) = len(layer_size)-1"
      )
    # Layers initialization
    # -----------------------
    if (self.interpolating is not None):
      self.interpolating['name'] = self.name
      self.lays = [pcm_layers.Interpolating(**self.interpolating)]
    else:
      # Hidden Layers
      names = [f'hidden_{i}' for i in range(len(self.layer_size)-2)]
      names = names + ['output']
      nb_mlp = len(self.layer_size)-1
      self.lays = [
        MLP(
          units=self.layer_size[i+1],
          activation=self.activation[i],
          kernel_initializer=self.kernel_initializer,
          regularizer=self.regularizer,
          layer_norm=self.layer_norm if (i == nb_mlp-1) else False,
          dropout=self.dropout,
          name=self.name+'_'+names[i]
        ) for i in range(nb_mlp)
      ]

  # Calling
  # ===================================
  def call(self, x, training=None):
    y = x
    for lay in self.lays:
      y = lay(y, training=training)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    interp_cfg = None
    if (self.interpolating is not None):
      interp_cfg = self.interp_lay.get_config()
    config = {
      'layer_size': self.layer_size,
      'activation': self.activation,
      'kernel_initializer': self.kernel_initializer,
      'regularizer': self.regularizer,
      'layer_norm': self.layer_norm,
      'dropout': self.dropout,
      'interpolating': interp_cfg
    }
    base_config = super(FFBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class ResPIBlock(FFBlock):
  """An improved residual block architecture for PINNs.

  See:
    Sifan Wang, Yujun Teng, Paris Perdikaris
    'Understanding and mitigating gradient pathologies in physics-informed
    neural networks'
    https://arxiv.org/abs/2001.04536
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=['linear'],
    uv_activation=['sigmoid','sigmoid'],
    kernel_initializer=None,
    regularizer=None,
    layer_norm=None,
    dropout=None,
    **kwargs
  ):
    super(ResPIBlock, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    # Hyperparameters initialization
    # -----------------------
    for i in range(1, len(self.layer_size)-2):
      if (self.layer_size[i] != self.layer_size[i+1]):
        utils.raise_err(
          ValueError,
          "The number of units of each hidden layer " \
          f"must be the same in '{self.name}'."
        )
    self.uv_activation = uv_activation
    # Layers initialization
    # -----------------------
    self.u, self.v = [
      MLP(
        units=self.layer_size[1],
        activation=self.uv_activation[i],
        kernel_initializer=self.kernel_initializer,
        regularizer=self.regularizer,
        layer_norm=self.layer_norm,
        dropout=self.dropout,
        name=name
      ) for (i, name) in enumerate(('u', 'v'))
    ]

  # Calling
  # ===================================
  def call(self, x, training=None):
    # Input layer
    y = x
    # Encoder layers
    u = self.u(y, training=training)
    v = self.v(y, training=training)
    # Hidden layers
    y = self.lays[0](y, training=training)
    for i in range(1,len(self.layer_size)-2):
      z = self.lays[i](y, training=training)
      y = (1 - z) * u + z * v
    # Output layer
    y = self.lays[-1](y, training=training)
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'uv_activation': self.uv_activation
    }
    base_config = super(ResPIBlock, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class MixedLaysBlock(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    layers,
    **kwargs
  ):
    super(MixedLaysBlock, self).__init__(**kwargs)
    # Order layers by index
    indices = list(layers.keys())
    indices.sort()
    # Initialize layers
    self.layers = {i: self.get_lay(layers[i]) for i in indices}

  def get_lay(self, layer={}):
    if callable(layer):
      return layer
    else:
      lay_cls = utils.get_class(modules=_NN_MOD, name=layer['lay_id'])
      if ('kwargs' not in layer):
        layer['kwargs'] = {}
      return lay_cls(**layer['kwargs'])

  # Calling
  # ===================================
  def call(self, inputs, training=False, *args, **kwargs):
    y = inputs
    for layer in self.layers.values():
      y = layer(y, training=training, *args, **kwargs)
    return y

  def __call__(self, inputs, training=False, *args, **kwargs):
    return self.call(inputs, training=training, *args, **kwargs)

  def forward(self, inputs, training=False, *args, **kwargs):
    return self.call(inputs, training=training, *args, **kwargs)
