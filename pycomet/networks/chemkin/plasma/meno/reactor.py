import tensorflow as tf

from .blocks import *
from pycomet import utils
from pycomet import networks
from pycomet import backend as bkd
from pycomet.networks.basenet import BaseNet
from pycomet.networks.meno.blocks import StateCorrection
from keras import layers as tf_layers

from scipy.constants import Boltzmann as UKB
UKB = tf.constant(UKB, dtype=bkd.get_floatx())


class MENOPlasmaReac(BaseNet):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
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
    super(MENOPlasmaReac, self).__init__(
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
    # Layers
    # -------------
    # Layers names
    if (lays_name == None):
      self.lays_name = self.lays_name + (
        "log_t_pe", "ic_xform", "electron", "fcorr_compute", "argon_update"
      )
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("log_t_pe", "ic_xform"),
      default=self.default_lays_propty
    )
    # Initializing
    self.active_lays = self.init_layers()
    self.init_preinput_lay()
    # Callable layers
    self.callable_lays = callable_lays
    if (self.callable_lays is None):
      self.callable_lays["electron"] = True \
        if ("electron" in self.active_lays) else False
      self.callable_lays["argon"] = True \
        if ("argon_update" in self.active_lays) else False

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier in ("log_t_pe", "ic_xform")):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier == "electron"):
        lays[identifier] = self.init_kernel_lay(propty)
      elif (identifier in ("fcorr_compute", "argon_update")):
        propty["lay_id"] = {
          "fcorr_compute": StateCorrection,
          "argon_update": SolUpdate
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
      "nar": None, # Argon n.d.    [1/m^3]: shape = [n,g]
      "ne": None   # Electron n.d. [1/m^3]: shape = [n,1]
    }
    argon_only = False
    # Extract inputs
    # -------------
    nar_0, ne_0, Te, t, ne_t = utils.unpack_list(inputs, nb_elem=5)
    # Compose inputs
    # -------------
    y_0 = tf_layers.Concatenate()([nar_0, ne_0])
    y_0_xform = self.active_lays["ic_xform"](y_0)
    ln_t_pe_0 = self.active_lays["log_t_pe"](t * ne_0 * UKB * Te)
    x = [y_0_xform, ln_t_pe_0]
    # Predict species
    # -------------
    # > Electron number density
    if self.callable_lays["electron"]:
      y["ne"] = ne_0 * self.active_lays["electron"](x, training=training)
    elif self.callable_lays["argon"]:
      y["ne"] = ne_t
      argon_only = True
    # > Argon number density
    if self.callable_lays["argon"]:
      # > Compute state-wise latent space
      x = tf_layers.Concatenate()(x)
      f = self.active_lays["fcorr_compute"](x, training=training)
      # > Update Argon distribution
      y["nar"] = self.active_lays["argon_update"](
        (t, nar_0, y["ne"], f), training=training
      )
    # Return predictions
    # -------------
    if argon_only:
      return y["nar"]
    else:
      return [y[k] for k in ("nar", "ne") if (y[k] is not None)]
