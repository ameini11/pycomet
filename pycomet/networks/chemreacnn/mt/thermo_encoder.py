import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import config as cfg
from keras import layers as tf_layers
from tensorflow.python.training.tracking.data_structures import NoDependency

# Hypernet (Version v.22.12)
from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
mix_mod = thermophysics.reaction_thermo.mixture
sp_th_mod = thermophysics.specie_thermo.specie_thermo


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class ThermoEncoderMT(tf_layers.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    process=[],
    specie_thermo={},
    inp_tensor=None,
    inp_indices=None,
    reordering={"mass": False, "temp": False},
    order=None,
    out_indices=None,
    **kwargs
  ):
    super(ThermoEncoderMT, self).__init__(**kwargs)
    # Thermodynamics
    # -----------------------
    # Do not track 'specie_thermo' dictionary
    self.specie_thermo = NoDependency(specie_thermo)
    # Specie thermodynamics
    self.sp_th = {}
    for (index, st) in self.specie_thermo.items():
      if ('config' in st):
        args = st
      else:
        args = {**st, **dict(index=index, process=process)}
      sp_th = sp_th_mod.SpecieThermo(**args)
      self.sp_th[sp_th.specie.name] = sp_th
    # Mixture
    self.mix = mix_mod.MultiComponent(self.sp_th)
    # Number of variables
    self.nb_comp = self.mix.n_comp
    self.nb_temp = 2  # Force 2T model
    if (self.nb_temp > 2):
      utils.raise_err(
        NotImplementedError,
        f"The current version of '{self.name}' " \
        "layer only supports Park's 2T model."
      )
    # QoIs indices to keep
    self.keep_indices = {
      "mass": np.arange(self.nb_comp-1).tolist(),
      "temp": np.arange(self.nb_temp-1).tolist(),
    }
    # Encoding parameters
    # -----------------------
    # Inputs encoding
    self.inp_tensor = inp_tensor
    self.inp_indices = inp_indices
    if (not self._is_inp_dict_correct(self.inp_indices)):
      utils.raise_err(
        ValueError,
        "Input 'inp_indices' must be a dictionary with " \
        "keys ('mass', 'temp') and values as lists of " \
        "integers indicating the indices of the tensor " \
        "to extract the corresponding physical quantities from."
      )
    self.inp_indices = {
      k: tf.nest.flatten(v) for (k, v) in self.inp_indices.items()
    }
    # Reordering quantities
    self.reordering = reordering
    if (not self._is_inp_dict_correct(self.reordering)):
      utils.raise_err(
        ValueError,
        "Input 'reordering' must be a dictionary with keys " \
        "('mass', 'temp') and values as boolean indicating " \
        "whether the physical quantities must be reordered or not."
      )
    self.order = order
    # Inputs decoding
    self.out_indices = out_indices
    if (self.out_indices is not None):
      self.out_index_to_key = {
        i: k for k, i in self.out_indices.items()
      }

  # Calling
  # ===================================
  def call(self, inputs):
    # Extract partial densities and temperatures
    x = {}
    xi = inputs if (self.inp_tensor is None) else inputs[self.inp_tensor]
    for (k, indices) in self.inp_indices.items():
      x[k] = tf.gather(xi, indices, axis=1)
    # Compute mass fractions
    rho = tf.math.reduce_sum(x["mass"], axis=1, keepdims=True)
    x["mass"] /= rho
    # Reorder mass fractions and temperatures
    for k in x.keys():
      if self.reordering[k]:
        x[k] = tf.gather(x[k], self.order[k], axis=1)
    # Update thermodynamics
    for st in self.sp_th.values():
      st.update(x["temp"])
    self.mix.update(x["mass"], var='Y')
    # Collect encoded input tensor
    y = {
      "Y": tf.gather(x["mass"], self.keep_indices["mass"], axis=1),
      "rho": rho,
      "T": tf.gather(x["temp"], self.keep_indices["temp"], axis=1),
      "e": self.mix._he()
    }
    # Compose encoded input tensor
    if (self.out_indices is not None):
      outputs_i = []
      for i in range(4):
        outputs_i.append(y[self.out_index_to_key[i]])
    else:
      outputs_i = list(y.values())
    outputs_i = tf.concat(outputs_i, axis=1)
    # Replace encoded input tensor
    outputs = []
    for i in range(len(inputs)):
      if (i == self.inp_tensor):
        outputs.append(outputs_i)
      else:
        outputs.append(inputs[i])
    return outputs

  # Utils
  # ===================================
  def _is_inp_dict_correct(self, idict):
    c1 = isinstance(idict, dict)
    c2 = False
    if c1:
      c2 = all([(k in idict) for k in ("mass", "temp")])
    return (c1 and c2)

  # Configuration
  # ===================================
  def get_config(self):
    specie_thermo_cfg = {
      st.specie.index: {'config': st.get_config()} \
        for st in self.sp_th.values()
    }
    config = {
      "specie_thermo": specie_thermo_cfg,
      "inp_tensor": self.inp_tensor,
      "inp_indices": self.inp_indices,
      "reordering": self.reordering,
      "order": self.order,
      "out_indices": self.out_indices
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
