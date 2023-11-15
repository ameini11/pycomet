import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import config as cfg
from pycomet.networks.elements import layers as pcm_layers
from tensorflow.python.training.tracking.data_structures import NoDependency

# Hypernet (Version v.22.12)
from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
mix_mod = thermophysics.reaction_thermo.mixture
sp_th_mod = thermophysics.specie_thermo.specie_thermo


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class ThermoEncoderMT(pcm_layers.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    process=[],
    specie_thermo={},
    inp_to_encode=0,
    inp_indices=None,
    inp_indices_ordered=None,
    inp_indices_to_model=None,
    out_qoi_order={"Y": 0, "rho": 1, "T": 2, "e": 3},
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
    # >> Species
    self.nb_comp = self.mix.n_comp
    # >> Temperatures: force 2T model
    self.nb_temp = 2
    if (self.nb_temp > 2):
      utils.raise_err(
        NotImplementedError,
        f"The current version of '{self.name}' " \
        "layer only supports Park's 2T model."
      )
    # Encoding parameters
    # -----------------------
    # Inputs encoding
    self.inp_to_encode = int(inp_to_encode)
    self.inp_indices = inp_indices
    if (self.inp_indices is None):
      self.inp_indices = {
        "mass": np.arange(self.nb_comp).tolist(),
        "temp": (self.nb_comp + np.arange(self.nb_temp)).tolist()
      }
    if (not self._is_inp_dict_correct(self.inp_indices)):
      utils.raise_err(
        ValueError,
        "The 'inp_indices' input should be a dictionary comprising " \
        "keys ('mass', 'temp') and values as lists of integers, " \
        "specifying the indices for extracting the corresponding " \
        "physical quantities, i.e., partial densities and temperatures."
      )
    # Reordering input quantities
    self.inp_indices_ordered = inp_indices_ordered
    if (self.inp_indices_ordered is not None):
      if (not self._is_inp_dict_correct(self.inp_indices_ordered)):
        utils.raise_value_err(
          "The 'inp_indices_ordered' input should be a dictionary " \
          "containing keys ('mass', 'temp') and values represented " \
          "as lists of integers, indicating the accurate sequence " \
          "of physical quantities (Y_i and T_i) to input to the model."
        )
    # Extracting input quantities to feed into the model
    self.inp_indices_to_model = inp_indices_to_model
    if (inp_indices_to_model is None):
      self.inp_indices_to_model = {
        "mass": np.arange(self.nb_comp - 1).tolist(),
        "temp": np.arange(self.nb_temp - 1).tolist()
      }
    if (not self._is_inp_dict_correct(self.inp_indices_to_model)):
      utils.raise_err(
        ValueError,
        "The 'inp_indices_to_model' input should be a dictionary " \
        "containing keys ('mass', 'temp') with corresponding values " \
        "as lists of integers. These lists indicate the indices of " \
        "the inputs (Yi and Ti) that you intend to input to the model."
      )
    # Flattening
    for k in (
      "inp_indices",
      "inp_indices_ordered",
      "inp_indices_to_model"
    ):
      d = getattr(self, k)
      if (d is not None):
        d = self._flatten_dict_values(d)
        setattr(self, k, d)
    # Inputs decoding
    self.out_qoi_order = out_qoi_order
    self.out_index_to_qoi = {
      i: k for k, i in self.out_qoi_order.items()
    }

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Flatten inputs
    inputs = tf.nest.flatten(inputs)
    # Extract partial densities and temperatures
    x = {}
    for (k, indices) in self.inp_indices.items():
      x[k] = tf.gather(inputs[self.inp_to_encode], indices, axis=1)
    # Compute mass fractions
    rho = tf.math.reduce_sum(x["mass"], axis=1, keepdims=True)
    x["mass"] /= rho
    # Reorder mass fractions and temperatures
    if (self.inp_indices_ordered is not None):
      for (k, indices) in self.inp_indices_ordered.items():
        x[k] = tf.gather(x[k], indices, axis=1)
    # Update thermodynamics
    for st in self.sp_th.values():
      st.update(x["temp"])
    self.mix.update(x["mass"], var='Y')
    # Collect encoded input tensor
    y = {
      "rho": rho,
      "e": self.mix._he()
    }
    for (k, name) in (("Y", "mass"), ("T", "temp")):
      y[k] = tf.gather(x[name], self.inp_indices_to_model[name], axis=1)
    # Compose encoded input tensor
    outputs_i = []
    for i in range(len(self.out_qoi_order)):
      outputs_i.append(y[self.out_index_to_qoi[i]])
    outputs_i = tf.concat(outputs_i, axis=1)
    # Replace encoded input tensor
    outputs = []
    for i in range(len(inputs)):
      if (i == self.inp_to_encode):
        outputs.append(outputs_i)
      else:
        outputs.append(inputs[i])
    return outputs if len(outputs) > 1 else outputs[-1]

  # Utils
  # ===================================
  def _is_inp_dict_correct(self, idict):
    c1 = isinstance(idict, dict)
    c2 = False
    if c1:
      c2 = all([(k in idict) for k in ("mass", "temp")])
    return (c1 and c2)

  def _flatten_dict_values(self, idict):
    return {k: tf.nest.flatten(v) for (k, v) in idict.items()}

  # Configuration
  # ===================================
  def get_config(self):
    specie_thermo_cfg = {
      st.specie.index: {'config': st.get_config()} \
        for st in self.sp_th.values()
    }
    config = {
      "specie_thermo": specie_thermo_cfg,
      "inp_to_encode": self.inp_to_encode,
      "inp_indices": self.inp_indices,
      "inp_indices_ordered": self.inp_indices_ordered,
      "inp_indices_to_model": self.inp_indices_to_model,
      "out_qoi_order": self.out_qoi_order
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
