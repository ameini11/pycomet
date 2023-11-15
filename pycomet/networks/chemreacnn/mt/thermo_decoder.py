import tensorflow as tf

from pycomet import utils
from pycomet import config as cfg
from pycomet import operations as ops
from pycomet.networks.elements import layers as pcm_layers
from tensorflow.python.training.tracking.data_structures import NoDependency

# Hypernet (Version v.22.12)
from hypernet.src.general import utils as hyp_utils
thermophysics = hyp_utils.get_thermophysics_mod()
mix_mod = thermophysics.reaction_thermo.mixture
sp_th_mod = thermophysics.specie_thermo.specie_thermo


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class ThermoDecoderMT(pcm_layers.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    process=[],
    specie_thermo={},
    mix_model_param=None,
    inp_encoded=0,
    inp_encoded_indices=None,
    out_qoi_order_in={"mass": 0, "temp": 1},
    out_qoi_order_out={"mass": 0, "temp": 1},
    **kwargs
  ):
    super(ThermoDecoderMT, self).__init__(**kwargs)
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
    # Mixture model parameters
    # -----------------------
    self.mix_model_param = mix_model_param
    if (self.mix_model_param is None):
      # The default one is Air-5
      mass = ("N", "O", "N2", "NO", "O2")
      temp = ("Th", "Tv")
      M = {k: self.sp_th[k].specie.M for k in ("N", "O")}
      self.mix_model_param = {
        "name": "air5",
        "order": {
          "mass": {i: k for (i, k) in enumerate(mass)},
          "temp": {i: k for (i, k) in enumerate(temp)}
        },
        "temp_lim": (200.0, 30000.0),
        "no_ratio": float(M["N"] / (M["O"] + M["N"]))
      }
    # Decoding parameters
    # -----------------------
    # Encoded inputs
    self.inp_encoded = int(inp_encoded)
    self.inp_encoded_indices = inp_encoded_indices
    if (self.inp_encoded_indices is None):
      self.inp_encoded_indices = {
        "rho": self.nb_comp,
        "Th": self.nb_comp + 1,
        "e": self.nb_comp + 2
      }
      order = self.mix_model_param["order"]["mass"]
      for (i, k) in enumerate(order.keys()):
        if (i != self.nb_comp-1):
          self.inp_encoded_indices[f"Y_{k}"] = i
    self.inp_encoded_indices = self._flatten_dict_values(
      self.inp_encoded_indices
    )
    # Outputs decoding
    # >> From model inference
    self.out_qoi_order_in = out_qoi_order_in
    # >> To external codes (i.e., CFD)
    self.out_qoi_order_out = out_qoi_order_out
    self.out_index_to_qoi_out = {
      i: k for k, i in self.out_qoi_order_out.items()
    }
    if (not self._is_inp_dict_correct(self.out_qoi_order_in)):
      utils.raise_err(
        ValueError,
        "The 'out_qoi_order_in' input must be a dictionary " \
        "with keys ('mass', 'temp') and values as integers " \
        "indicating the order of the output tensors to decode."
      )
    if (not self._is_inp_dict_correct(self.out_qoi_order_out)):
      utils.raise_err(
        ValueError,
        "The 'out_qoi_order_out' input must be a dictionary " \
        "with keys ('mass', 'temp') and values as integers " \
        "indicating the order of the decoded output tensors."
      )

  # Calling
  # ===================================
  def call(self, inputs):
    y = {}
    # Extract inputs/outputs quantities
    qoi = self._extract_io_qoi(inputs)
    # Enforce conservation of mass
    y["mass"] = self._conserve_mass(qoi)
    # Enforce conservation of energy
    y["temp"] = self._conserve_energy(qoi)
    # Compose outputs
    if (self.out_qoi_order_out is not None):
      y = [y[self.out_index_to_key[i]] for i in range(2)]
    # Return outputs
    return y

  # Input/Output QoIs
  # -----------------------------------
  def _extract_io_qoi(self, inputs):
    qoi = {}
    # Separate model inputs/outputs
    x, y = ops.unpack_list(inputs, n_elem=2)
    # Extract model inputs QoIs
    qoi["inp"] = self._extract_inp_qoi(x)
    # Extract model outputs QoIs
    qoi["out"] = self._extract_out_qoi(y)
    return qoi

  def _extract_inp_qoi(self, inputs):
    qoi = {}
    inputs = tf.nest.flatten(inputs)
    # >> Total energy of the mixture
    qoi["e"] = tf.gather(
      inputs[self.inp_encoded],
      self.inp_encoded_indices["e"],
      axis=1
    )
    # >> Total masses of chemical elements
    if (self.mix_model_param["name"].lower() == "air5"):
      # Total nitrogen mass
      yn_tot = 0.0
      for k in ("N", "N2", "NO"):
        yi = tf.gather(
          inputs[self.inp_encoded],
          self.inp_encoded_indices[f"Y_{k}"],
          axis=1
        )
        if (k == "NO"):
          yi *= self.mix_model_param["no_ratio"]
        yn_tot = yn_tot + yi
      qoi["yn_tot"] = yn_tot
    return qoi

  def _extract_out_qoi(self, outputs):
    qoi = {}
    outputs = tf.nest.flatten(outputs)
    # Mass fractions and vibro-electronic temperature
    for k in ("mass", "temp"):
      qoi[k] = outputs[self.out_qoi_order_in[k]]
    return qoi

  # Mass conservation
  # -----------------------------------
  def _conserve_mass(self, qoi):
    # Enforce mass fractions to be between 0 and 1
    y = qoi["out"]["mass"]
    y = tf.math.sigmoid(y)
    # Enforce conservation of mass
    if (self.nb_comp > 2):
      y = self._conserve_mass_multi_comp(y, qoi)
    else:
      y = tf.concat([1.0 - y, y], 1)
    # Update mixture
    self.mix.update(y, var='Y')
    # Return mass fractions
    return y

  def _conserve_mass_multi_comp(self, y, qoi):
    if (self.mix_model_param["name"].lower() == "air5"):
      # Total nitrogen mass
      yn_tot = qoi["inp"]["yn_tot"]
      # Extract predicted species
      y = tf.split(y, num_or_size_splits=3, axis=1)
      y = {k: y[i] for (i, k) in enumerate(("N", "N2", "O"))}
      # Enforce conservation of oxygen
      y["O"] *= (1.0 - yn_tot)
      # Enforce conservation of nitrogen
      yn = y["N"] + y["N2"]
      yn = (yn_tot - tf.nn.relu(yn_tot - yn)) / yn
      for k in ("N", "N2"):
        y[k] *= yn
      # Compute the remaining species
      # >> NO
      y["NO"] = yn_tot - y["N"] - y["N2"]
      y["NO"] *= (1 / self.mix_model_param["no_ratio"])
      # >> O2
      y_tot = 0.0
      for yi in y.values():
        y_tot = y_tot + yi
      y["O2"] = 1.0 - y_tot
      # Compose mass fractions
      y = [y[k] for k in self.mix_model_param["order"]["mass"]]
      y = tf.concat(y, axis=1)
    # Return mass fractions
    return y

  # Energy conservation
  # -----------------------------------
  def _conserve_energy(self, qoi):
    # Get useful parameters
    e = qoi["inp"]["e"]
    T_lim = self.mix_model_param["temp_lim"]
    # Enforce temperature to be positive
    T_ve = qoi["out"]["temp"]
    T_ve = tf.math.sigmoid(T_ve) * (T_lim[1] - T_lim[0]) + T_lim[0]
    # Formation energy
    e_f = self.mix._e_f()
    # Vibro-electronic energy mode
    e_ve = self._e_ve(T_ve)
    # Roto-translational energy mode
    cv_rt = self._cv_rt()
    e_rt_min = cv_rt * T_lim[0]
    e_rt = tf.nn.relu(e - e_f - e_ve - e_rt_min) + e_rt_min
    # Compose temperatures
    T = {"Th": e_rt / cv_rt, "Tv": T_ve}
    T = [T[k] for k in self.mix_model_param["order"]["temp"]]
    T = tf.concat(T, axis=1)
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
      "mix_model_param": self.mix_model_param,
      "inp_encoded": self.inp_encoded,
      "inp_encoded_indices": self.inp_encoded_indices,
      "out_qoi_order_in": self.out_qoi_order_in,
      "out_qoi_order_out": self.out_qoi_order_out,
      "T_min": self.T_min
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))
