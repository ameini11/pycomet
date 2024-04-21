import tensorflow as tf

from .blocks import *
from pycomet import utils
from pycomet.networks.basenet import BaseNet
from hypernet import const
from pycomet.networks.don.opnn import _init_input_lay, _manipulate_input


class LinNonlinME(BaseNet):
  """
  Master equation integral solution using:
  - Nonlinear: DeepONet-based surrogate for the nonlinear variables (na and T)
  - Linear:    Matrix exponential-based integral solution for the
               distribution function with a state-specific corrective factor.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    lays_name=None,
    lays_propty=None,
    noisy_io=None,
    inference=None,
    masking=False,
    force_siso=False,
    *args,
    **kwargs
  ):
    super(LinNonlinME, self).__init__(
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
    # Layers names
    default_lays_name = (
      "pf_update", "fcorr_xform", "fcorr_compute", "operators", "sol_update"
    )
    if (lays_name is None):
      self.lays_name = self.lays_name + default_lays_name
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=default_lays_name,
      default=self.default_lays_propty
    )
    # Initializing layers
    self.active_lays = self.init_layers()
    self.init_preinput_lay()

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == "fcorr_xform"):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier in (
        "pf_update", "fcorr_compute", "operators", "sol_update"
      )):
        lays[identifier] = self.init_kernel_lay(identifier, propty)
    self.thermo_lay = self.active_lays["thermo_update"]
    return lays

  def init_kernel_lay(self, identifier, propty):
    propty["lay_id"] = {
      "thermo_update": ComposeInputs,
      "correction": Correction,
      "dist_update": SolUpdate
    }[identifier]
    return self.init_lay(identifier, propty)

  # Calling
  # ===================================
  def forward(self, inputs, training=None):
    # Extract inputs
    # -------------
    na_0, nmi_0, T_0, t, qoi_t = utils.unpack_list(inputs, nb_elem=5)
    # Update thermodynamics
    # -------------
    # Compute state energies
    self.thermo_lay(T_0, species="molecule", qois="e")
    emi_0 = self.thermo_lay.get_updated(species="molecule", qoi="e")
    # Compute moments of the distribution
    nm_0 = tf.math.reduce_sum(nmi_0, axis=-1, keepdims=True)
    em_0 = tf.math.reduce_sum(nmi_0*emi_0, axis=-1, keepdims=True) / nm_0
    # Compose inputs
    # -------------
    n_0 = na_0 + 2*nm_0
    y_0 = tf.concat([na_0, n_0, em_0, T_0], axis=-1)
    x = _manipulate_input(self, [y_0, t])
    # Predict species
    # -------------
    # > Atom number density
    if ("mass" in self.active_lays):
      na = n_0 * self.active_lays["mass"](x, training=training)
    else:
      na = tf.gather(qoi_t, [self.mass_index], axis=-1)
    # > Molecule number density
    nm = 0.5*(n_0-na)
    # Predict temperature
    # -------------
    if ("temp" in self.active_lays):
      T = T_0 * self.active_lays["temp"](x, training=training)
    else:
      T = tf.gather(qoi_t, [self.temp_index], axis=-1)
    # Predict distribution
    # -------------
    if ("dist_update" in self.active_lays):
      # > Update partition functions
      self.thermo_lay(T, qois="q")
      q = [self.thermo_lay.get_updated(s, "q") for s in ("atom", "molecule")]
      # > Compute state-wise latent space
      t_pa = t * na * const.UKB * T
      f = tf.concat([y_0, tf.math.log(t_pa+1e-10)], axis=-1)
      f = self.active_lays["fcorr_xform"](f, training=training)
      f = self.active_lays["fcorr_compute"](f, training=training)
      # > Update distribution
      nmi = self.active_lays["dist_update"](
        (t, na, nm, T, *q, nmi_0, f), training=training
      )
    else:
      nmi = None
    # Return predictions
    # -------------
    y = [na]
    if (nmi is not None):
      y.append(nmi)
    y.append(T)
    return y
