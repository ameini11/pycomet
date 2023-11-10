import tensorflow as tf

from pycomet import utils
from pycomet import config as cfg
from pycomet import operations as ops
from keras import layers as tf_layers
from tensorflow.python.training.tracking.data_structures import NoDependency

# Hypernet (Version v.22.12)
from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
mix_mod = thermophysics.reaction_thermo.mixture
sp_th_mod = thermophysics.specie_thermo.specie_thermo


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class ThermoDecoderMT(tf_layers.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    process=[],
    specie_thermo={},
    inp_energy_tensor=0,
    inp_energy_index=-1,
    out_indices_in=None,
    out_indices_out=None,
    T_min=100.0,
    **kwargs
  ):
    super(ThermoDecoderMT, self).__init__(**kwargs)
    # Thermodynamics
    # -----------------------
    # Do not track 'specie_thermo' dictionary
    self.specie_thermo = NoDependency(specie_thermo)
    # Specie thermodynamics
    self.sp_th = {}
    for (index, s) in self.specie_thermo.items():
      if ('config' in s):
        args = s
      else:
        args = {**s, **dict(index=index, process=process)}
      sp_th = sp_th_mod.SpecieThermo(**args)
      self.sp_th[sp_th.specie.name] = sp_th
    # Mixture
    self.mix = mix_mod.MultiComponent(self.sp_th)
    # Temperature limiter
    self.T_min = T_min
    if (self.mix.nb_temp > 2):
      utils.raise_err(
        NotImplementedError,
        f"The current version of '{self.name}' " \
        "layer only supports Park's 2T model."
      )
    # Decoding parameters
    # -----------------------
    # Input total energy
    self.inp_energy_tensor = int(inp_energy_tensor)
    self.inp_energy_index = int(inp_energy_index)
    # Outputs decoding
    self.out_indices_in = out_indices_in
    self.out_indices_out = out_indices_out
    if (self.out_indices_out is not None):
      self.out_index_to_key = {
        i: k for k, i in self.out_indices_out.items()
      }
    if (not self._is_inp_dict_correct(self.out_indices_in)):
      utils.raise_err(
        ValueError,
        "The 'out_indices_in' input must be a dictionary with keys " \
        "('mass', 'temp') and values as integers indicating the " \
        "order of the output tensors to decode."
      )
    if (not self._is_inp_dict_correct(self.out_indices_out)):
      utils.raise_err(
        ValueError,
        "The 'out_indices_out' input must be a dictionary with keys " \
        "('mass', 'temp') and values as integers indicating the " \
        "order of the decoded output tensors."
      )

  # Calling
  # ===================================
  def call(self, inputs):
    # Extract inputs/outputs quantities
    y, e = self._extract_io_qoi(inputs)
    # Enforce conservation of mass
    y["mass"] = self._decode_mass(y["mass"])
    # Enforce conservation of energy
    y["temp"] = self._decode_temp(y["temp"], e)
    # Compose outputs
    if (self.out_indices_out is not None):
      y = [y[self.out_index_to_key[i]] for i in range(2)]
    # Return outputs
    return y

  def _extract_io_qoi(self, inputs):
    # Separate inputs from outputs
    x, y = ops.unpack_list(inputs, n_elem=2)
    # Extract model outputs QoIs:
    #  - Mass fractions
    #  - Vibro-electronic temperature
    y = {k: y[self.out_indices_in[k]] for k in ("mass", "temp")}
    # Extract model inputs QoIs:
    #  - Total energy
    e = tf.gather(
      x[self.inp_energy_tensor], [self.inp_energy_index], axis=1
    )
    return y, e

  def _decode_mass(self, x):
    # Enforce mass fractions to be between 0 and 1
    yi = tf.math.sigmoid(x)
    # Enforce conservation of mass
    if (self.mix.nb_comp > 2):
      # Simple and normalized sum
      ss = tf.math.reduce_sum(yi, axis=1, keepdims=True)
      sn = tf.math.sigmoid(ss)
      # Mass fractions
      y0 = 1.0 - sn
      yi = sn * yi / ss
    else:
      y0 = 1.0 - yi
    # Compose mass fractions
    y = tf.concat([y0, yi], 1)
    # Update mixture
    self.mix.update(y, var='Y')
    # Return mass fractions
    return y

  def _decode_temp(self, x, e):
    # Enforce temperature to be positive
    T_ve = tf.nn.relu(x) + self.T_min
    # Formation energy
    e_f = self.mix._e_f()
    # Vibro-electronic energy mode
    e_ve = self._e_ve(T_ve)
    # Roto-translational energy mode
    cv_rt = self._cv_rt()
    e_rt_min = cv_rt * self.T_min
    e_rt = tf.nn.relu(e - e_f - e_ve - e_rt_min) + e_rt_min
    # Compose temperatures
    T = tf.concat([e_rt / cv_rt, T_ve], 1)
    # Return temperatures
    return T

  def _e_ve(self, T):
    e_ve = 0.0
    for (name, st) in self.sp_th.items():
      if (name == "e"):
        # Electronic energy
        e_ve = e_ve + st.thermo._cv_tr() * T * st.specie.Y
      elif (st.specie.n_at > 1):
        # Vibrational energy
        e_ve = e_ve + st.thermo.th_int.ho.e(T) * st.specie.Y
    return e_ve

  def _cv_rt(self):
    cv_rt = 0.0
    for (name, st) in self.sp_th.items():
      if (name != "e"):
        # Translational energy
        cv_rt = cv_rt + st.thermo._cv_tr() * st.specie.Y
        if (st.specie.n_at > 1):
          # Rotational energy
          cv_rt = cv_rt + st.thermo.th_int.rr._cv() * st.specie.Y
    return cv_rt

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
      "inp_energy_tensor": self.inp_energy_tensor,
      "inp_energy_index": self.inp_energy_index,
      "out_indices_in": self.out_indices_in,
      "out_indices_out": self.out_indices_out,
      "T_min": self.T_min
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
