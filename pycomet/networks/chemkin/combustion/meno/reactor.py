from pycomet import utils
from pycomet import networks
from pycomet.networks.basenet import BaseNet
from pycomet.networks import elements as nn_elem
from keras import layers as tf_layers

# Blocks
from .blocks import POLLUSolUpdate
from pycomet.networks.meno.blocks import StateCorrection


class MENOPOLLUReac(BaseNet):

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
    super(MENOPOLLUReac, self).__init__(
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
    if (lays_name is None):
      self.lays_name = self.lays_name + (
        "log_t", "ic_xform", "y1", "y2", "y6", "fcorr_compute", "y_lin_update"
      )
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("log_t", "ic_xform"),
      default=self.default_lays_propty
    )
    # Initializing
    self.active_lays = self.init_layers()
    self.init_preinput_lay()
    # Callable layers
    self.callable_lays = callable_lays
    if (self.callable_lays is None):
      self.callable_lays = {}
      if any([(f"y{i}" in self.active_lays) for i in (1, 2, 6)]):
        self.callable_lays["y_nlin"] = True
      else:
        self.callable_lays["y_nlin"] = False
      if ("y_lin_update" in self.active_lays):
        self.callable_lays["y_lin"] = True
      else:
        self.callable_lays["y_lin"] = False

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier in ("log_t", "ic_xform")):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier in ("y1", "y2", "y6")):
        lays[identifier] = self.init_kernel_lay(propty)
      elif (identifier in ("fcorr_compute", "y_lin_update")):
        propty["lay_id"] = {
          "fcorr_compute": StateCorrection,
          "y_lin_update": POLLUSolUpdate
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
      "y_lin": None,  # Linear state variables
      "y_nlin": None  # Nonlinear state variables
    }
    lin_only = False
    # Extract inputs
    # -------------
    y_l_0, y_nl_0, y_inp, t, y_nl_t = utils.unpack_list(inputs, nb_elem=5)
    print(y_l_0.shape, y_nl_0.shape, y_inp.shape, t.shape, y_nl_t.shape)
    # Compose inputs
    # -------------
    y_tot = nn_elem.Reduction(indices=None, reduction="sum")(
      y_inp, axis=-1, keepdims=True
    )
    x = [
      self.active_lays["ic_xform"](y_inp),
      self.active_lays["log_t"](t*y_tot)
    ]
    # Predict species
    # -------------
    # > Nonlinear state variables
    if self.callable_lays["y_nlin"]:
      y_nl_t = []
      for (i, j) in enumerate((1, 2, 6)):
        name = f"y{j}"
        if (name in self.active_lays):
          yj = nn_elem.Gather([i], axis=-1)(y_nl_0)
          yj += t * self.active_lays[name](x, training=training)
          y_nl_t.append(yj)
      y["y_nlin"] = tf_layers.Concatenate()(y_nl_t)
    elif self.callable_lays["y_lin"]:
      y["y_nlin"] = y_nl_t
      lin_only = True
    # > Linear state variables
    if self.callable_lays["y_lin"]:
      # > Compute state-wise latent space
      x = tf_layers.Concatenate()(x)
      f = self.active_lays["fcorr_compute"](x, training=training)
      # > Update Argon distribution
      y["y_lin"] = self.active_lays["y_lin_update"](
        (t, y_l_0, y["y_nlin"], f), training=training
      )
    # Return predictions
    # -------------
    if lin_only:
      return y["y_lin"]
    else:
      return [y[k] for k in ("y_lin", "y_nlin") if (y[k] is not None)]
