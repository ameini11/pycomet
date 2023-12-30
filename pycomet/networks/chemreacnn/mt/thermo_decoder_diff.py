import numpy as np
import tensorflow as tf

from pycomet import utils
from pycomet import config as cfg
from pycomet import operations as ops
from .thermo_decoder import ThermoDecoderMT


@tf.keras.utils.register_keras_serializable(package=cfg._NAME)
class ThermoDecoderMTDiff(ThermoDecoderMT):

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
    super(ThermoDecoderMTDiff, self).__init__(
      process=process,
      specie_thermo=specie_thermo,
      apply_cons=apply_cons,
      apply_mass_limiter=apply_mass_limiter,
      mix_model=mix_model,
      mix_model_param=mix_model_param,
      Tv_lim=Tv_lim,
      inp_encoded=inp_encoded,
      inp_encoded_indices=inp_encoded_indices,
      out_qoi_order_in=out_qoi_order_in,
      out_qoi_order_out=out_qoi_order_out,
      **kwargs
    )

  # def _set_mix_model_param(
  #   self,
  #   mix_model_param=None,
  #   Tv_lim=(200.0, 15000.0)
  # ):
  #   if (mix_model_param is None):
  #     mix_model_param = {}
  #     # Temperatures list
  #     temp = ("Th", "Tv")
  #     if (self.mix_model == "air5"):
  #       # Species list
  #       mass = ("N", "O", "N2", "NO", "O2")
  #       # Mass fraction of nitrogen in NO
  #       m = {k: self.sp_th[k].specie.M for k in ("N", "O")}
  #       mn = float(m["N"] / (m["O"] + m["N"]))
  #       mix_model_param["mn_in_no"] = mn
  #       # Weights for computing input total mass fraction of nitrogen
  #       wn = np.array([1,0,1,mn]).reshape(1,-1)
  #       mix_model_param["weigths_n"] = wn
  #       # Predicted species
  #       mix_model_param["pred_species"] = ("N", "O", "NO")
  #     elif (self.mix_model == "oxygen"):
  #       # Species list
  #       mass = ("O", "O2")
  #       # Predicted species
  #       mix_model_param["pred_species"] = ("O",)
  #     else:
  #       utils.raise_value_err(
  #         f"The specified mixture model '{self.mix_model}' is " \
  #         "unrecognized. The available options are: ['air5', 'oxygen']."
  #       )
  #     # Shared parameters
  #     mix_model_param["order"] = {
  #       "mass": {i: k for (i, k) in enumerate(mass)},
  #       "temp": {i: k for (i, k) in enumerate(temp)}
  #     }
  #     mix_model_param["Tv_lim"] = Tv_lim
  #   # Set 'mix_model_param'
  #   self.mix_model_param = mix_model_param

  # def _set_inp_encoded_indices(
  #   self,
  #   inp_encoded_indices=None
  # ):
  #   if (inp_encoded_indices is None):
  #     inp_encoded_indices = {
  #       "rho": self.nb_comp - 1,
  #       "Tv": self.nb_comp,
  #       "e": self.nb_comp + 1,
  #       "Y": [],
  #       "Y_pred": []
  #     }
  #     order = self.mix_model_param["order"]["mass"]
  #     for (i, k) in order.items():
  #       if (i != self.nb_comp-1):
  #         inp_encoded_indices[f"Y_{k}"] = i
  #         inp_encoded_indices["Y"].append(i)
  #     for k in self.mix_model_param["pred_species"]:
  #       i = inp_encoded_indices[f"Y_{k}"]
  #       inp_encoded_indices["Y_pred"].append(i)
  #   inp_encoded_indices = self._flatten_dict_values(inp_encoded_indices)
  #   # Set 'inp_encoded_indices'
  #   self.inp_encoded_indices = inp_encoded_indices

  # Calling
  # ===================================
  # Input/Output QoIs
  # -----------------------------------
  def _extract_inp_qoi(self, inputs):
    qoi = {}
    inputs = tf.nest.flatten(inputs)
    # >> Total energy of the mixture
    qoi["Tv"] = tf.gather(
      inputs[self.inp_encoded],
      self.inp_encoded_indices["Tv"],
      axis=1
    )
    if self.apply_cons["temp"]:
      qoi["e"] = tf.gather(
        inputs[self.inp_encoded],
        self.inp_encoded_indices["e"],
        axis=1
      )
    # >> Total masses of chemical elements
    # # Get initial mass fractions of predicted species
    # qoi["y_pred"] = tf.gather(
    #   inputs[self.inp_encoded],
    #   self.inp_encoded_indices["Y_pred"],
    #   axis=1
    # )
    if self.apply_cons["mass"]:
      if (self.mix_model == "air5"):
        # Get input mass fractions
        y = tf.gather(
          inputs[self.inp_encoded],
          self.inp_encoded_indices["Y"],
          axis=1
        )
        # Compute total nitrogen mass
        qoi["yn_tot"] = tf.math.reduce_sum(
          y * self.mix_model_param["weigths_n"],
          axis=1,
          keepdims=True
        )
    return qoi

  # Predicted variables
  # -----------------------------------
  # def _get_mass(self, qoi):
  #   # Enforce mass fractions to be between 0 and 1
  #   dy = tf.math.tanh(qoi["out"]["mass"])
  #   y = qoi["inp"]["y_pred"] + dy
  #   return tf.clip_by_value(y, 0, 1)

  def _get_temp(self, qoi):
    # Get useful parameters
    Tv_lim = self.mix_model_param["Tv_lim"]
    dTv_lim = Tv_lim[1] - Tv_lim[0]
    # Enforce temperature to be positive and limited
    dTv = tf.math.tanh(qoi["out"]["temp"]) * dTv_lim
    Tv = qoi["inp"]["Tv"] + dTv
    return tf.clip_by_value(Tv, *Tv_lim)

  # # Mass conservation
  # # -----------------------------------
  # def _conserve_mass_multi_comp(self, y, qoi):
  #   if (self.mix_model == "air5"):
  #     # Get useful parameters
  #     mn = self.mix_model_param["mn_in_no"]
  #     sp = self.mix_model_param["pred_species"]
  #     # Conserved masses
  #     yn_tot = qoi["inp"]["yn_tot"]
  #     yo_tot = 1.0 - yn_tot
  #     # Extract predicted species
  #     y = tf.split(y, num_or_size_splits=3, axis=1)
  #     y = {k: y[i] for (i, k) in enumerate(sp)}
  #     # Enforce conservation of mass
  #     # >> Limit NO
  #     if self.apply_mass_limiter:
  #       yn_tot_no = yn_tot * (1/mn)
  #       yo_tot_no = yo_tot * (1/(1-mn))
  #       y["NO"] = yn_tot_no - tf.nn.relu(yn_tot_no - y["NO"])
  #       y["NO"] = yo_tot_no - tf.nn.relu(yo_tot_no - y["NO"])
  #     # Enforce conservation of nitrogen
  #     dyn = yn_tot - mn * y["NO"]
  #     # >> Limit N
  #     if self.apply_mass_limiter:
  #       y["N"] *= dyn
  #     # >> Compute N2
  #     y["N2"] = dyn - y["N"]
  #     # Enforce conservation of oxygen
  #     dyo = yo_tot - (1 - mn) * y["NO"]
  #     # >> Limit O
  #     if self.apply_mass_limiter:
  #       y["O"] *= dyo
  #     # >> Compute O2
  #     y["O2"] = dyo - y["O"]
  #     # Compose mass fractions
  #     y = self._compose_outputs(y, self.nb_comp, "mass")
  #   # Return mass fractions
  #   return y
