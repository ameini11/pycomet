import numpy as np
import tensorflow as tf

from keras import layers as tf_layers

from pycomet import utils
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()

# Hypernet (v.22.12)
from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
sp_th_mod = thermophysics.specie_thermo.specie_thermo


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Boltzmann(nn_elem.Layer):
  """Boltzmann Layer"""

  # Initialization
  # ===================================
  def __init__(
    self,
    specie_thermo=None,
    process=None,
    **kwargs
  ):
    super(Boltzmann, self).__init__(**kwargs)
    # Specie thermodynamics
    self.process = process
    if "config" in specie_thermo:
      args = specie_thermo
    else:
      args = {**specie_thermo, **dict(index=0, process=self.process)}
    self.sp_th = sp_th_mod.SpecieThermo(**args)

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract mass fractions and temperature
    ya, ym, T = inputs
    # Partition functions
    Q = self.sp_th.pf.int.spline["Q"].apply(points=T)
    # Boltzmann avarage
    ym = (ym * Q) / tf.math.reduce_sum(Q, axis=1, keepdims=True)
    # Return
    return ya, ym, T

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "specie_thermo": {"config": self.sp_th.get_config()},
      "process": self.process
    }
    base_config = super(Boltzmann, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class CGEquilibrium(nn_elem.Layer):
  """Coarse-graining equilibrium layer"""

  # Initialization
  # ===================================
  def __init__(
    self,
    output_dim=2,
    use_pf_int=False,
    specie_thermo=None,
    process=None,
    **kwargs
  ):
    super(CGEquilibrium, self).__init__(**kwargs)
    # Output dimensions
    self.output_dim = tf.nest.flatten(output_dim)
    # Partition functions
    self.process = process
    self.use_pf_int = use_pf_int
    if self.use_pf_int:
      # Specie thermodynamics
      if "config" in specie_thermo:
        args = specie_thermo
      else:
        args = {**specie_thermo, **dict(index=0, process=self.process)}
      self.sp_th = sp_th_mod.SpecieThermo(**args)

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    if isinstance(inputs, (list, tuple)):
      x, T = inputs
    else:
      utils.raise_err(
        ValueError,
        f"The inputs of the '{self.name}' layer have " \
        "to be a 'list' or a 'tuple': [x, T]"
      )
    y = tf.nest.flatten(x)
    if self.use_pf_int:
      # Partition functions
      Q = self.sp_th.pf.int.spline["Q"].apply(points=T)
      Q = nn_elem.Split(self.output_dim, axis=-1)(Q)
      # Simil-Softmax layer
      y = [tf.math.exp(-yi) * Qi for (yi, Qi) in zip(y, Q)]
      y = [yi / tf.math.reduce_sum(yi, axis=-1, keepdims=1) for yi in y]
    else:
      y = [tf_layers.Softmax(axis=-1)(yi) for yi in y]
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    if self.use_pf_int:
      specie_thermo_CFG = {"config": self.sp_th.get_config()}
    else:
      specie_thermo_CFG = None
    config = {
      "output_dim": self.output_dim,
      "use_pf_int": self.use_pf_int,
      "specie_thermo": specie_thermo_CFG,
      "process": self.process
    }
    base_config = super(CGEquilibrium, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class AddIC(nn_elem.Layer):
  """Initial condition add operation layer"""

  # Initialization
  # ===================================
  def __init__(
    self,
    out_tensor=None,
    out_indices=None,
    out_xform=None,
    in_tensor=None,
    in_indices=None,
    in_xform=None,
    **kwargs
  ):
    super(AddIC, self).__init__(**kwargs)
    # Index of the output tensor
    self.out_tensor = 0 if (out_tensor is None) else int(out_tensor)
    # Columns indices to be extracted from the output tensor
    self.out_indices = [0] if (out_indices is None) else out_indices
    self.out_indices = tf.nest.flatten(self.out_indices)
    # Extracted inoutputput transformation
    self.out_xform = out_xform
    # Index of the input tensor
    self.in_tensor = 0 if (in_tensor is None) else int(in_tensor)
    # Columns indices to be extracted from the input tensor
    self.in_indices = [0] if (in_indices is None) else in_indices
    self.in_indices = tf.nest.flatten(self.in_indices)
    # Extracted input transformation
    self.in_xform = in_xform

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Separate model inputs and outputs
    x, y = ops.unpack_list(inputs, n_elem=2)
    x, y = tf.nest.flatten(x), tf.nest.flatten(y)
    # Extract & tranform model outputs - QoI
    yi_qoi = tf.gather(y[self.out_tensor], self.out_indices, axis=1)
    if (self.out_xform is not None):
      yi_qoi = self.out_xform(yi_qoi)
    # Extract & tranform model inputs  - QoI IC
    xi_qoi = tf.gather(x[self.in_tensor], self.in_indices, axis=1)
    if (self.in_xform is not None):
      xi_qoi = self.in_xform(xi_qoi)
    # Add operation
    _y = []
    for (i, yi) in enumerate(y):
      if (i == self.out_tensor):
        yi = tf_layers.Add()([xi_qoi, yi_qoi])
      _y.append(yi)
    return _y if (len(_y) > 1) else _y[-1]

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "out_tensor": self.out_tensor,
      "out_indices": self.out_indices,
      "out_xform": self.out_xform,
      "in_tensor": self.in_tensor,
      "in_indices": self.in_indices,
      "in_xform": self.in_xform
    }
    base_config = super(AddIC, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class CGThermoEncoder(nn_elem.Layer):
  """Coarse-graining thermo encoder layer"""

  # Initialization
  # ===================================
  def __init__(
    self,
    tensor_toencode=0,
    nb_bins=1,
    indices_toencode=None,
    encoded_tensors_order=None,
    **kwargs
  ):
    super(CGThermoEncoder, self).__init__(**kwargs)
    # Tensor to encode
    self.tensor_toencode = tensor_toencode
    # Tensor"s indices to encode
    self.nb_bins = int(nb_bins)
    self.indices_toencode = indices_toencode
    if (self.indices_toencode == None):
      indices = np.arange(self.nb_bins)
      self.indices_toencode = {
        "rho_a": [0], "rho_m": 1+indices, "Th": [indices[-1]+2]
      }
    self.indices_toencode = {
      k: list(indices) for (k, indices) in self.indices_toencode.items()
    }
    # Encoded tensor"s order
    self.encoded_tensors_order = encoded_tensors_order
    if (self.encoded_tensors_order == None):
      self.encoded_tensors_order = {0: "rho", 1: "Y_m", 2: "Th"}

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extracting inputs quantities to encode
    x = {}
    for (k, index) in self.indices_toencode.items():
      x[k] = tf.gather(inputs[self.tensor_toencode], index, axis=1)
    # Encoded quantities
    y = {}
    # Encode mass quantities
    y["rho"] = tf.math.reduce_sum(x["rho_m"], axis=1, keepdims=1)
    y["rho"] = y["rho"] + x["rho_a"]
    y["Y_m"] = x["rho_m"] / y["rho"]
    # Encode temperature quantities
    y["Th"] = x["Th"]
    # Compose the encoded tensor
    y_en = [y[self.encoded_tensors_order[i]] for i in range(len(y))]
    y_en = tf_layers.Concatenate()(y_en) if (len(y_en) > 1) else y_en[-1]
    # Compose and return outputs
    outputs = []
    for (i, x) in enumerate(inputs):
      if (i == self.tensor_toencode):
        outputs.append(y_en)
      else:
        outputs.append(x)
    return outputs

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "tensor_toencode": self.tensor_toencode,
      "nb_bins": self.nb_bins,
      "indices_toencode": self.indices_toencode,
      "encoded_tensors_order": self.encoded_tensors_order
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class FastestToCurrentTS(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    fastest_to_current_ts,
    **kwargs
  ):
    super(FastestToCurrentTS, self).__init__(**kwargs)
    # Fastest-to-current timescale connectivity
    self.fastest_to_current_ts = fastest_to_current_ts

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    y = inputs
    if (self.fastest_to_current_ts is not None):
      _y = []
      for indices in self.fastest_to_current_ts:
        _y.append(
          tf.math.reduce_sum(
            tf.gather(y, indices, axis=1), axis=1, keepdims=1
          )
        )
      y = tf_layers.Concatenate()(_y) if (len(_y) > 1) else _y[-1]
    return y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "fastest_to_current_ts": self.fastest_to_current_ts
    }
    base_config = super(FastestToCurrentTS, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class CurrentToNextTS(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    current_to_next_ts,
    **kwargs
  ):
    super(CurrentToNextTS, self).__init__(**kwargs)
    # Current-to-next timescale connectivity
    self.current_to_next_ts = current_to_next_ts

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    y = inputs
    _y = [y[0]]
    for ts in range(1,len(y)):
      indices = self.current_to_next_ts[ts-1]
      if (indices is not None):
        # Split the previous slower timescale
        y_slow = nn_elem.Split(_y[ts-1].shape[1], axis=1)(_y[ts-1])
        # Gather the current faster timescale by indices
        y_fast = [tf.gather(y[ts], idx, axis=1) for idx in indices]
        if (len(y_slow) != len(y_fast)):
          y_slow = y_slow[-len(y_fast):]
        # Multiply slower and faster timescales
        y_ts = [tf_layers.Multiply()(z) for z in zip(y_slow, y_fast)]
        # Concatenate
        y_ts = tf_layers.Concatenate()(y_ts) if (len(y_ts) > 1) else y_ts[-1]
        # Store the solution
        _y.append(y_ts)
    return _y

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "current_to_next_ts": self.current_to_next_ts
    }
    base_config = super(CurrentToNextTS, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
