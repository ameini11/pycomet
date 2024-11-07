import tensorflow as tf

from pycomet import utils
from pycomet import networks
from pycomet import backend as bkd
from pycomet.networks.basenet import BaseNet
from pycomet.networks import elements as nn_elem
from keras import layers as tf_layers

# Blocks
from ..meno.blocks import MoleToMassUpdate, NonEqThermoUpdate

# Constants
from scipy.constants import Boltzmann as UKB
UKB = tf.constant(UKB, dtype=bkd.get_floatx())


class DONonEqReac(BaseNet):
  """
  Master equation integral solution using vanilla DeepONet
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
    super(DONonEqReac, self).__init__(
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
    self.qoi_indices = qoi_indices
    # Layers
    # -------------
    # Layers names
    if (lays_name is None):
      self.lays_name = self.lays_name + (
        "log_t_pa", "ic_xform", "mass", "temp", "mole_to_mass", "thermo_update"
      )
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("log_t_pa", "ic_xform", "mole_to_mass", "thermo_update"),
      default=self.default_lays_propty
    )
    # Initializing
    self.active_lays = self.init_layers()
    self.init_preinput_lay()
    # Callable layers
    self.callable_lays = callable_lays
    if (self.callable_lays is None):
      self.callable_lays = {
        "mass": True if ("mass" in self.active_lays) else False,
        "temp": True if ("temp" in self.active_lays) else False
      }

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier in ("log_t_pa", "ic_xform")):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier in ("mass", "temp")):
        lays[identifier] = self.init_kernel_lay(propty)
      elif (identifier in ("mole_to_mass", "thermo_update")):
        propty["lay_id"] = {
          "mole_to_mass": MoleToMassUpdate,
          "thermo_update": NonEqThermoUpdate
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
    y = {"n": None, "T": None}
    # Extract inputs
    # -------------
    na_0, nmi_0, T_0, t, qoi_eq = utils.unpack_list(inputs, nb_elem=5)
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
    ln_t_pa_0 = self.active_lays["log_t_pa"](t * na_0 * UKB * T_0)
    x = [y_0_xform, ln_t_pa_0]
    # Predict species
    # -------------
    # > Atom number density
    if self.callable_lays["mass"]:
      # Compute density
      rho = self.active_lays["mole_to_mass"]([na_0, nmi_0], qoi="rho")
      # Mass fractions
      w = self.active_lays["mass"](x, training=training)
      w = nn_elem.Split(num_or_size_splits=[1, w.shape[-1]-1], axis=-1)(w)
      # Number densities
      n = self.active_lays["mole_to_mass"](w, rho, qoi="n")
      y["n"] = tf_layers.Concatenate()(n)
    # Predict temperature
    # -------------
    if self.callable_lays["temp"]:
      T_eq = nn_elem.Gather([self.qoi_indices["temp"]], axis=-1)(qoi_eq)
      y["T"] = self.active_lays["temp"](x, training=training)
      y["T"] = T_0 + (T_eq-T_0) * y["T"]
    # Return predictions
    # -------------
    return [y[k] for k in ("n", "T") if (y[k] is not None)]
