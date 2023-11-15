import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import config as cfg
from pycomet import operations as ops
from .thermo_layer import ThermoLayerMT


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class ThermoDecoderMT(ThermoLayerMT):

  # Initialization
  # ===================================
  def __init__(
    self,
    process=[],
    specie_thermo={},
    apply_cons={"mass": True, "temp": True},
    apply_mass_limiter=True,
    mix_model="air5",
    mix_model_param=None,
    Tv_lim=(200.0, 15000.0),
    inp_encoded=0,
    inp_encoded_indices=None,
    out_qoi_order_in={"mass": 0, "temp": 1},
    out_qoi_order_out={"mass": 0, "temp": 1},
    **kwargs
  ):
    super(ThermoDecoderMT, self).__init__(
      process=process,
      specie_thermo=specie_thermo,
      **kwargs
    )
    self.apply_cons = apply_cons
    self.apply_mass_limiter = apply_mass_limiter
    # Mixture model parameters
    # -----------------------
    self.mix_model = mix_model.lower()
    self._set_mix_model_param(mix_model_param, Tv_lim)
    # Decoding parameters
    # -----------------------
    # Encoded inputs
    self.inp_encoded = int(inp_encoded)
    self._set_inp_encoded_indices(inp_encoded_indices)
    # Outputs decoding
    # >> From model inference
    self.out_qoi_order_in = out_qoi_order_in
    # >> To external codes (e.g., CFD)
    self.out_qoi_order_out = out_qoi_order_out
    self.out_index_to_qoi_out = {
      i: k for k, i in self.out_qoi_order_out.items()
    }
    self._check_inp_dicts()

  def _set_mix_model_param(
    self,
    mix_model_param=None,
    Tv_lim=(200.0, 15000.0)
  ):
    if (mix_model_param is None):
      mix_model_param = {}
      # Temperatures list
      temp = ("Th", "Tv")
      if (self.mix_model == "air5"):
        # Species list
        mass = ("N", "O", "N2", "NO", "O2")
        # Mass fraction of nitrogen in NO
        m = {k: self.sp_th[k].specie.M for k in ("N", "O")}
        mn = float(m["N"] / (m["O"] + m["N"]))
        mix_model_param["mn_in_no"] = mn
        # Weights for computing input total mass fraction of nitrogen
        wn = np.array([1,0,1,mn]).reshape(1,-1)
        mix_model_param["weigths_n"] = wn
        # Predicted species
        mix_model_param["pred_species"] = ("N2", "NO", "O2")
      elif (self.mix_model == "oxygen"):
        # Species list
        mass = ("O", "O2")
        # Predicted species
        mix_model_param["pred_species"] = ("O2",)
      else:
        utils.raise_err(
          ValueError,
          f"The specified mixture model '{self.mix_model}' is " \
          "unrecognized. The available options are: ['air5', 'oxygen']."
        )
      # Shared parameters
      mix_model_param["order"] = {
        "mass": {i: k for (i, k) in enumerate(mass)},
        "temp": {i: k for (i, k) in enumerate(temp)}
      }
      mix_model_param["Tv_lim"] = Tv_lim
    # Set 'mix_model_param'
    self.mix_model_param = mix_model_param

  def _set_inp_encoded_indices(
    self,
    inp_encoded_indices=None
  ):
    if (inp_encoded_indices is None):
      inp_encoded_indices = {
        "rho": self.nb_comp - 1,
        "Th": self.nb_comp,
        "e": self.nb_comp + 1
      }
      inp_encoded_indices["Y"] = []
      order = self.mix_model_param["order"]["mass"]
      for (i, k) in order.items():
        if (i != self.nb_comp-1):
          inp_encoded_indices[f"Y_{k}"] = i
          inp_encoded_indices["Y"].append(i)
    inp_encoded_indices = self._flatten_dict_values(inp_encoded_indices)
    # Set 'inp_encoded_indices'
    self.inp_encoded_indices = inp_encoded_indices

  def _check_inp_dicts(self):
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
  def call(self, inputs, training=None):
    y = {}
    # Extract inputs/outputs quantities
    qoi = self._extract_io_qoi(inputs)
    # Get predicted variables
    y["mass"] = self._get_mass(qoi)
    y["temp"] = self._get_temp(qoi)
    # Enforce conservation of mass
    if self.apply_cons["mass"]:
      y["mass"] = self._conserve_mass(y["mass"], qoi)
    # Enforce conservation of energy
    if self.apply_cons["temp"]:
      y["temp"] = self._conserve_energy(y["temp"], qoi)
    # Compose outputs
    if (self.out_qoi_order_out is not None):
      y = [y[self.out_index_to_qoi_out[i]] for i in range(2)]
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
    if self.apply_cons["temp"]:
      qoi["e"] = tf.gather(
        inputs[self.inp_encoded],
        self.inp_encoded_indices["e"],
        axis=1
      )
    # >> Total masses of chemical elements
    if self.apply_cons["mass"]:
      if (self.mix_model == "air5"):
        # Get input mass fractions
        y = tf.gather(
          inputs[self.inp_encoded],
          self.inp_encoded_indices["Y"],
          axis=1
        )
        # Get weights for computing total nitrogen mass
        w = self.mix_model_param["weigths_n"]
        # Compute total nitrogen mass
        qoi["yn_tot"] = tf.math.reduce_sum(y*w, axis=1, keepdims=True)
    return qoi

  def _extract_out_qoi(self, outputs):
    qoi = {}
    outputs = tf.nest.flatten(outputs)
    # Mass fractions and vibro-electronic temperature
    for k in ("mass", "temp"):
      qoi[k] = outputs[self.out_qoi_order_in[k]]
    return qoi

  # Predicted variables
  # -----------------------------------
  def _get_mass(self, qoi):
    # Enforce mass fractions to be between 0 and 1
    return tf.math.sigmoid(qoi["out"]["mass"])

  def _get_temp(self, qoi):
    # Get useful parameters
    Tv_lim = self.mix_model_param["Tv_lim"]
    # Enforce temperature to be positive and limited
    Tv = tf.math.sigmoid(qoi["out"]["temp"])
    Tv = Tv * (Tv_lim[1] - Tv_lim[0]) + Tv_lim[0]
    return Tv

  # Mass conservation
  # -----------------------------------
  def _conserve_mass(self, y, qoi):
    if (self.nb_comp > 2):
      y = self._conserve_mass_multi_comp(y, qoi)
    elif (self.nb_comp == 2):
      y = tf.concat([1.0 - y, y], 1)
    # Update mixture
    self.mix.update(y, var='Y')
    # Return mass fractions
    return y

  def _conserve_mass_multi_comp(self, y, qoi):
    if (self.mix_model == "air5"):
      # Get useful parameters
      mn = self.mix_model_param["mn_in_no"]
      sp = self.mix_model_param["pred_species"]
      # Conserved masses
      yn_tot = qoi["inp"]["yn_tot"]
      yo_tot = 1.0 - yn_tot
      # Extract predicted species
      y = tf.split(y, num_or_size_splits=3, axis=1)
      y = {k: y[i] for (i, k) in enumerate(sp)}
      # Enforce conservation of mass
      # >> Limit NO
      if self.apply_mass_limiter:
        yn_tot_no = yn_tot * (1/mn)
        yo_tot_no = yo_tot * (1/(1-mn))
        y["NO"] = yn_tot_no - tf.nn.relu(yn_tot_no - y["NO"])
        y["NO"] = yo_tot_no - tf.nn.relu(yo_tot_no - y["NO"])
      # Enforce conservation of nitrogen
      dyn = yn_tot - mn * y["NO"]
      # >> Limit N2
      if self.apply_mass_limiter:
        y["N2"] *= dyn
      y["N"] = dyn - y["N2"]
      # Enforce conservation of oxygen
      dyo = yo_tot - (1 - mn) * y["NO"]
      # >> Limit O2
      if self.apply_mass_limiter:
        y["O2"] *= dyo
      y["O"] = dyo - y["O2"]
      # Compose mass fractions
      y = self._compose_outputs(y, self.nb_comp, "mass")
    # Return mass fractions
    return y

  # Energy conservation
  # -----------------------------------
  def _conserve_energy(self, Tv, qoi):
    # Get useful parameters
    Tv_lim = self.mix_model_param["Tv_lim"]
    # Conserved total energy
    e = qoi["inp"]["e"]
    # Formation energy
    e_f = self.mix._e_f()
    # Vibro-electronic energy mode
    e_ve = self._e_ve(Tv)
    # Roto-translational energy mode
    cv_rt = self._cv_rt()
    e_rt_min = cv_rt * Tv_lim[0]
    e_rt = tf.nn.relu(e - e_f - e_ve - e_rt_min) + e_rt_min
    # Compose temperatures
    T = {"Th": e_rt / cv_rt, "Tv": Tv}
    T = self._compose_outputs(T, self.nb_temp, "temp")
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
  def _compose_outputs(self, y, dim, identifier="mass"):
    yl = []
    for i in range(dim):
      k = self.mix_model_param["order"][identifier][i]
      yl.append(y[k])
    return tf.concat(yl, axis=1)

  # Configuration
  # ===================================
  def get_config(self):
    config = {
      "apply_cons": self.apply_cons,
      "apply_mass_limiter": self.apply_mass_limiter,
      "mix_model": self.mix_model,
      "mix_model_param": self.mix_model_param,
      "inp_encoded": self.inp_encoded,
      "inp_encoded_indices": self.inp_encoded_indices,
      "out_qoi_order_in": self.out_qoi_order_in,
      "out_qoi_order_out": self.out_qoi_order_out
    }
    base_config = super(ThermoDecoderMT, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
