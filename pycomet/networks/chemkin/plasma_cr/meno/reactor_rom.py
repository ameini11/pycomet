import tensorflow as tf

from pycomet import utils
from pycomet import networks
from pycomet import operations as ops
from pycomet.networks.basenet import BaseNet
from pycomet.networks import elements as nn_elem
from keras import layers as tf_layers

from . import const
from .blocks import PlasmaCRMoleMass
from .blocks_rom import PlasmaCRSolUpdateROM
from pycomet.networks.meno.blocks import StateCorrection


class MENOPlasmaCRReactorROM(BaseNet):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    decode=False,
    return_full=False,
    qoi_indices=None,
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
    super(MENOPlasmaCRReactorROM, self).__init__(
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
    self.decode = decode
    self.return_full = return_full
    # Time-dependent inputs
    # -------------
    # Only for training the model for the electronic states (optional)
    self.qoi_indices = qoi_indices
    if (self.qoi_indices is not None):
      self.qoi_indices = tf.nest.map_structure(
        lambda x: ops.to_numpy(x, int).reshape(-1),
        qoi_indices
      )
    # Nonlinear variables
    self.nlin_vars = ("n_em", "n_Ar1", "Th", "Te")
    # Layers
    # -------------
    # Layers names
    if (lays_name is None):
      self.lays_name += ("lin_prom", "log_t_ph", "ic_xform", "mole_mass")
      self.lays_name += self.nlin_vars + ("fcorr_compute", "states_update")
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("lin_prom", "log_t_ph", "ic_xform", "mole_mass"),
      default=self.default_lays_propty
    )
    # Initializing
    self.active_lays = self.init_layers()
    self.init_preinput_lay()
    # Callable layers
    self.callable_lays = callable_lays
    if (self.callable_lays is None):
      self.callable_lays = {k: (k in self.active_lays) for k in self.nlin_vars}
      self.callable_lays["states"] = ("states_update" in self.active_lays)
    # Mixture indices
    mix = self.active_lays["mole_mass"].mix
    self.mix_indices = mix.indices
    self.mix_indices["n_em"] = mix.species["em"].indices
    self.mix_indices["n_Ar1"] = mix.species["Ar"].indices[0]
    self.mix_indices = tf.nest.map_structure(
      lambda x: ops.to_numpy(x, int).reshape(-1),
      self.mix_indices
    )

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier in ("lin_prom", "log_t_ph", "ic_xform")):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier in self.nlin_vars):
        lays[identifier] = self.init_kernel_lay(propty)
      elif (identifier in ("mole_mass", "fcorr_compute", "states_update")):
        propty["lay_id"] = {
          "mole_mass": PlasmaCRMoleMass,
          "fcorr_compute": StateCorrection,
          "states_update": PlasmaCRSolUpdateROM
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
    y = {k: None for k in self.nlin_vars + ("z",)}
    self.states_only = False
    # Extract inputs
    # -------------
    rhoz_0, Th_0, Te_0, t, qoi_t = utils.unpack_list(inputs, nb_elem=5)
    # Compose inputs
    # -------------
    # > Process initial latent space
    w_0, z_0, rho = self.active_lays["lin_prom"](rhoz_0, method=5)
    # > Get number densities
    n_0 = self.active_lays["mole_mass"](rho*w_0, method=3)
    # > Get heavy-particle pressure
    nh_0 = nn_elem.Gather(indices=self.mix_indices["h"], axis=-1)(n_0)
    nh_0 = nn_elem.Reduction(indices=None, reduction="sum")(
      nh_0, axis=-1, keepdims=True
    )
    ph_0 = nh_0 * const.UKB * Th_0
    # Initial conditions
    # > Latent space
    zh_0, we_0 = nn_elem.Split([z_0.shape[1]-1, 1], axis=-1)(z_0)
    log_we_0 = tf_layers.Lambda(lambda x: tf.math.log(tf.math.abs(x)))(we_0)
    # > Concatenating
    y_0 = tf_layers.Concatenate()([rho, zh_0, log_we_0, Th_0, Te_0])
    y_0_xform = self.active_lays["ic_xform"](y_0)
    # Time
    # > Scale time by the heavy-particle pressure
    log_t_ph_0 = self.active_lays["log_t_ph"](t * ph_0)
    # Inputs
    x_logt = [y_0_xform, log_t_ph_0]
    # Predict nonlinear variables
    # -------------
    # > Initial value
    qoi_0 = {"Th": Th_0, "Te": Te_0}
    for k in ("n_em", "n_Ar1"):
      qoi_0[k] = nn_elem.Gather(indices=self.mix_indices[k], axis=-1)(n_0)
    # > Predict value
    for k in self.nlin_vars:
      if self.callable_lays[k]:
        y[k] = self.active_lays[k](x_logt, training=training) * qoi_0[k]
      elif self.callable_lays["states"]:
        y[k] = nn_elem.Gather(indices=self.qoi_indices[k], axis=-1)(qoi_t)
        self.states_only = True
    # Predict linear variables (states)
    # -------------
    if self.callable_lays["states"]:
      # > Compute state-wise latent space
      x = tf_layers.Concatenate()(x_logt)
      f = self.active_lays["fcorr_compute"](x, training=training)
      # > Update states
      u = [y[k] for k in self.nlin_vars]
      y["z"] = self.active_lays["states_update"]((t, zh_0, u, f), training)
      if self.decode:
        y["z"] = self.active_lays["lin_prom"](y["z"], method=2)
    # Return predictions
    # -------------
    if self.states_only:
      return y["z"]
    else:
      if self.return_full:
        return [y["z"], y["Th"], y["Te"]]
      else:
        return [yk for yk in y.values() if (yk is not None)]
