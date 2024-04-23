import tensorflow as tf

from .blocks import *
from pycomet import utils
from hypernet import const
from pycomet import networks
from keras import layers as tf_layers
from pycomet.networks.basenet import BaseNet
from pycomet.networks import elements as nn_elem


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
    t_pa_eps=1e-10,
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
    self.t_pa_eps = t_pa_eps
    self.qoi_indices = qoi_indices
    # Layers
    # -------------
    # Layers names
    if (lays_name == None):
      self.lays_name = self.lays_name + (
        "ic_xform", "mass", "temp", "thermo_update", \
        "fcorr_xform", "fcorr_compute", "dist_update"
      )
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("ic_xform", "thermo_update"),
      default=self.default_lays_propty
    )
    # Initializing
    self.active_lays = self.init_layers()
    self.init_preinput_lay()
    # Callable layers
    self.callable_lays = callable_lays
    if (self.callable_lays is None):
      self.callable_lays = []
      for k in ("mass", "temp", "dist_update"):
        if k in self.active_lays:
          self.callable_lays.append(k[:4])

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {"log": nn_elem.Log(base="e", offset=self.t_pa_eps)}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == "ic_xform"):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier in ("mass", "temp")):
        lays[identifier] = self.init_kernel_lay(propty)
      elif (identifier == "fcorr_xform"):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier in ("thermo_update", "fcorr_compute", "dist_update")):
        propty["lay_id"] = {
          "thermo_update": ThermoUpdate,
          "fcorr_compute": Correction,
          "dist_update": SolUpdate
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
      "na": None,  # Atom n.d. [1/m^3]:                  shape = [n,1]
      "nmi": None, # Molecule distribution n.d. [1/m^3]: shape = [n,g]
      "T": None    # Translation temperature [K]:        shape = [n,1]
    }
    _dist_only = False
    # Extract inputs
    # -------------
    na_0, nmi_0, T_0, t, qoi_t = utils.unpack_list(inputs, nb_elem=5)
    # Constants
    ONE, UKB = [tf.constant(c, dtype=na_0.dtype) for c in (1.0, const.UKB)]
    # Update thermodynamics
    # -------------
    # Compute state energies
    emi_0 = self.active_lays["thermo_update"](
      T_0, species="molecule", qois="e_int"
    )
    # Compute moments of the distribution
    nm_0 = nn_elem.Reduction(indices=None, reduction="sum")(
      nmi_0, axis=-1, keepdims=True
    )
    em_0 = nn_elem.Reduction(indices=None, reduction="sum")(
      nmi_0*emi_0, axis=-1, keepdims=True
    ) / nm_0
    # Compose inputs
    # -------------
    n_0 = na_0 + tf_layers.Rescaling(scale=2.0)(nm_0)
    y_0 = tf_layers.Concatenate()([na_0, n_0, em_0, T_0])
    y_0_xform = self.active_lays["ic_xform"](y_0)
    ln_t_pa_0 = self.active_lays["log"](t * na_0 * UKB * T_0)
    x = [y_0_xform, ln_t_pa_0]
    # Predict species
    # -------------
    # > Atom number density
    if ("mass" in self.callable_lays):
      y["na"] = n_0 * (ONE + self.active_lays["mass"](x, training=training))
    elif ("dist" in self.callable_lays):
      y["na"] = nn_elem.Gather([self.qoi_indices["mass"]], axis=-1)(qoi_t)
      _dist_only = True
    # > Molecule number density
    if (y["na"] is not None):
      nm = tf_layers.Rescaling(scale=0.5)(n_0-y["na"])
    # Predict temperature
    # -------------
    if ("temp" in self.callable_lays):
      y["T"] = T_0 * (ONE + self.active_lays["temp"](x, training=training))
    elif ("dist" in self.callable_lays):
      y["T"] = nn_elem.Gather([self.qoi_indices["temp"]], axis=-1)(qoi_t)
      _dist_only = True
    # Predict distribution
    # -------------
    if ("dist" in self.callable_lays):
      # > Update partition functions
      q = self.active_lays["thermo_update"](y["T"], qois="q_tot")
      # > Compute state-wise latent space
      ln_t_pa = self.active_lays["log"](t * y["na"] * UKB * y["T"])
      f = tf_layers.Concatenate()([y_0, ln_t_pa])
      f = self.active_lays["fcorr_xform"](f, training=training)
      f = self.active_lays["fcorr_compute"](f, training=training)
      # > Update distribution
      y["nmi"] = self.active_lays["dist_update"](
        (t, y["na"], nm, y["T"], *q, nmi_0, f), training=training
      )
    # Return predictions
    # -------------
    if _dist_only:
      return y["nmi"]
    else:
      return [y[k] for k in ("na", "nmi", "T") if (y[k] is not None)]
