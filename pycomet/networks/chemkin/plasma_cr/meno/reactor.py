import tensorflow as tf

from pycomet import utils
from pycomet import networks
from pycomet import backend as bkd
from pycomet import operations as ops
from pycomet.networks.basenet import BaseNet
from pycomet.networks import elements as nn_elem
from keras import layers as tf_layers

# Blocks
from .blocks import PlasmaCRMoleMass, PlasmaCRSolUpdate
from pycomet.networks.meno.blocks import StateCorrection

# Constants
from scipy.constants import Boltzmann as UKB
UKB = tf.constant(UKB, dtype=bkd.get_floatx())


class MENOPlasmaCRReactor(BaseNet):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    qoi_indices={"mass": 0, "temp": 1},
    callable_lays=None,
    lays_name=None,
    lays_propty=None,
    noisy_io=None,
    inference=None,
    masking=False,
    force_siso=False,
    *args,
    **kwargs
  ):
    super(MENOPlasmaCRReactor, self).__init__(
      dim=dim,
      lays_name=lays_name,
      lays_propty=lays_propty,
      noisy_io=noisy_io,
      inference=inference,
      masking=masking,
      force_siso=force_siso,
      *args,
      **kwargs
    )
    # Time-dependent inputs
    # -------------
    # Only for training the model for the distribution function (optional)
    self.qoi_indices = tf.nest.map_structure(
      lambda x: ops.to_numpy(x, int).reshape(-1),
      qoi_indices
    )
    # Layers
    # -------------
    # Layers names
    if (lays_name is None):
      self.lays_name = self.lays_name + (
        "lin_autoenc", "log_t_ph", "ic_xform", "mole_mass", \
        "em", "Ar1", "Th", "Te", "fcorr_compute", "states_update"
      )
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("lin_autoenc", "log_t_ph", "ic_xform", "mole_mass"),
      default=self.default_lays_propty
    )
    # Initializing
    self.active_lays = self.init_layers()
    self.init_preinput_lay()
    # Callable layers
    self.callable_lays = callable_lays
    if (self.callable_lays is None):
      self.callable_lays = {
        k: (k in self.active_lays) for k in ("em", "Ar1", "Th", "Te")
      }
      self.callable_lays["states"] = ("states_update" in self.active_lays)
    # Mixture indices
    mix = self.active_lays["states_update"].operators.mix["h"]
    self.mix_indices = mix.indices
    self.mix_indices["em"] = mix.species["em"].indices
    self.mix_indices["Ar1"] = mix.species["Ar"].indices[0]
    self.mix_indices["lin"] = mix.indices["h"][1:]
    self.mix_indices = tf.nest.map_structure(
      lambda x: ops.to_numpy(x, int).reshape(-1),
      self.mix_indices
    )

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier in ("lin_autoenc", "log_t_ph", "ic_xform")):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier in ("em", "Ar1", "Th", "Te")):
        lays[identifier] = self.init_kernel_lay(propty)
      elif (identifier in ("mole_mass", "fcorr_compute", "states_update")):
        propty["lay_id"] = {
          "mole_mass": PlasmaCRMoleMass,
          "fcorr_compute": StateCorrection,
          "states_update": PlasmaCRSolUpdate
        }[identifier]
        lays[identifier] = self.init_lay(identifier, propty)
    return lays

  def init_kernel_lay(self, propty={}):
    opnn = "OpNN" if ("lay_id" not in propty) else propty["lay_id"]
    opnn = opnn if callable(opnn) else utils.get_class(networks, opnn)
    return opnn(
      dim=propty["dim"],
      lays_propty=propty["lays_propty"]
    )

  # Calling
  # ===================================
  def forward(self, inputs, training=None):
    # Set outputs
    # -------------
    y = {
      "em": None,
      "Ar1": None,
      "Ari": None,
      "Th": None,
      "Te": None
    }
    self.states_only = False
    # Extract inputs
    # -------------
    rho_0, Th_0, Te_0, t, qoi_t = utils.unpack_list(inputs, nb_elem=5)
    # Compose inputs
    # -------------
    # Initial conditions
    # > Get total density and mass fractions
    rho, w_0 = self.active_lays["mole_mass"](rho_0, method=1)
    # > Encode mass fractions
    z_0 = self.active_lays["lin_autoenc"](w_0, method=1)
    # > Concatenating
    y_0 = tf_layers.Concatenate()([rho, z_0, Th_0, Te_0])
    y_0_xform = self.active_lays["ic_xform"](y_0)
    # Time
    # > Get number densities
    n_0 = self.active_lays["mole_mass"](rho_0, method=3)
    nh_0 = nn_elem.Gather(indices=self.mix_indices["h"], axis=-1)(n_0)
    nh_0 = nn_elem.Reduction(indices=None, reduction="sum")(
      nh_0, axis=-1, keepdims=True
    )
    # > Scale time by the heavy-particle pressure
    ln_t_ph_0 = self.active_lays["log_t_ph"](t * nh_0 * UKB * Th_0)
    # Inputs
    x = [y_0_xform, ln_t_ph_0]
    # Predict nonlinear variables
    # -------------
    qoi_0 = {"Th": Th_0, "Te": Te_0}
    for k in ("em", "Ar1"):
      qoi_0[k] = nn_elem.Gather(indices=self.mix_indices[k], axis=-1)(n_0)
    for k in y.keys():
      if (k != "Ari"):
        y = self.call_nlin_lay(k, x, y, qoi_0, qoi_t, training)
    # Predict linear variables (states)
    # -------------
    if self.callable_lays["states"]:
      # > Compute state-wise latent space
      x = tf_layers.Concatenate()(x)
      f = self.active_lays["fcorr_compute"](x, training=training)
      # > Update states
      xnlin = [y[k] for k in ("em", "Ar1", "Th", "Te")]
      xlin_0 = nn_elem.Gather(indices=self.mix_indices["lin"], axis=-1)(n_0)
      y["Ari"] = self.active_lays["states_update"](
        (t, xlin_0, xnlin, f), training=training
      )
    # Return predictions
    # -------------
    if self.states_only:
      return self.convert_n_to_w(y)
    else:
      if self.return_full:
        return [self.convert_n_to_w(y), y["Th"], y["Te"]]
      else:
        return [yk for yk in y.values() if (yk is not None)]

  def call_nlin_lay(self, key, x, y, qoi_0, qoi_t, training):
    if self.callable_lays[key]:
      y[key] = qoi_0[key] * self.active_lays[key](x, training=training)
    elif self.callable_lays["states"]:
      y[key] = nn_elem.Gather(indices=self.qoi_indices[key], axis=-1)(qoi_t)
      self.states_only = True
    return y

  def convert_n_to_w(self, y):
    y = tf_layers.Concatenate()([y[k] for k in ("em", "Ar1", "Ari")])
    return self.active_lays["mole_mass"](y, method=4)
