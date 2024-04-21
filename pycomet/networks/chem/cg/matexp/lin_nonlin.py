import tensorflow as tf

from .blocks import *
from pycomet import utils
from hypernet import const
from pycomet import networks
from keras import layers as tf_layers
from pycomet.networks.basenet import BaseNet
from pycomet.networks import elements as nn_elem
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
    # Input
    # -------------
    if (not isinstance(self.input_dim, (list, tuple))):
      utils.raise_err(
        ValueError,
        f"'dim['inp']' passed into '{self.name}' has to be a "
        "'list' or a 'tuple' of length 2: [u_dim, y_dim]"
      )
    self.qoi_indices = qoi_indices
    # Layers
    # -------------
    # Default layers
    self.default_lays_propty.update({
      "input": {
        "branch": {
          "index": 0,
          "xform": None
        },
        "trunk": {
          "index": 1,
          "xform": None
        }
      },
      "output": {
        "xform": None
      }
    })
    # Layers names
    if (lays_name == None):
      self.lays_name = self.lays_name + (
        "mass", "temp", "thermo_update", \
        "fcorr_xform", "fcorr_compute", "dist_update"
      )
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("thermo_update",),
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
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == "input"):
        lays[identifier] = _init_input_lay(self, identifier, propty)
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
    self.thermo_lay = self.active_lays["thermo_update"]
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
    # Extract inputs
    # -------------
    na_0, nmi_0, T_0, t, qoi_t = utils.unpack_list(inputs, nb_elem=5)
    # Update thermodynamics
    # -------------
    # Compute state energies
    self.thermo_lay(T_0, species="molecule", qois="e_int")
    emi_0 = self.thermo_lay.get_updated(species="molecule", qoi="e_int")
    # Compute moments of the distribution
    nm_0 = nn_elem.Reduction(indices=None, reduction="sum")(
      nmi_0, axis=-1, keepdims=True
    )
    em_0 = nn_elem.Reduction(indices=None, reduction="sum")(
      nmi_0*emi_0, axis=-1, keepdims=True
    ) / nm_0
    # Compose inputs
    # -------------
    n_0 = na_0 + 2*nm_0
    y_0 = tf_layers.Concatenate()([na_0, n_0, em_0, T_0])
    x = _manipulate_input(self, [y_0, t])
    # Predict species
    # -------------
    # > Atom number density
    if ("mass" in self.callable_lays):
      na = n_0 * self.active_lays["mass"](x, training=training)
      y["na"] = na
    else:
      na = nn_elem.Gather(indices=[self.qoi_indices["mass"]], axis=-1)(qoi_t)
    # > Molecule number density
    nm = 0.5*(n_0-na)
    # Predict temperature
    # -------------
    if ("temp" in self.callable_lays):
      T = T_0 * self.active_lays["temp"](x, training=training)
      y["T"] = T
    else:
      T = nn_elem.Gather(indices=[self.qoi_indices["temp"]], axis=-1)(qoi_t)
    # Predict distribution
    # -------------
    if ("dist" in self.callable_lays):
      # > Update partition functions
      self.thermo_lay(T, qois="q_tot")
      q = []
      for s in ("atom", "molecule"):
        q.append(self.thermo_lay.get_updated(s, "q_tot"))
      # > Compute state-wise latent space
      ln_t_pa = nn_elem.Log(base="e", offset=1e-10)(t * na * const.UKB * T)
      f = tf_layers.Concatenate()([y_0, ln_t_pa])
      f = self.active_lays["fcorr_xform"](f, training=training)
      f = self.active_lays["fcorr_compute"](f, training=training)
      # > Update distribution
      y["nmi"] = self.active_lays["dist_update"](
        (t, na, nm, T, *q, nmi_0, f), training=training
      )
    # Return predictions
    # -------------
    return [y[k] for k in ("na", "nmi", "T") if (y[k] is not None)]
