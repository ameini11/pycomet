import tensorflow as tf

from pycomet import utils
from pycomet import networks
from pycomet.networks.basenet import BaseNet
from pycomet.networks.don.opnn import _init_input_lay, _manipulate_input


class NonLinME(BaseNet):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    mass_index=1,
    lays_name=None,
    lays_propty=None,
    noisy_io=None,
    inference=None,
    masking=False,
    force_siso=False,
    return_init_cond=None,
    *args,
    **kwargs
  ):
    super(NonLinME, self).__init__(
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
    self.mass_index = mass_index
    self.return_init_cond = return_init_cond
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
      self.lays_name = self.lays_name + ("temp", "mass")
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      default=self.default_lays_propty
    )
    # Initializing
    self.active_lays = self.init_layers()
    self.init_preinput_lay()

  # Layers initialization
  # ===================================
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == "input"):
        lays[identifier] = _init_input_lay(self, identifier, propty)
      elif (identifier == ("mass", "temp")):
        lays[identifier] = self.init_kernel_lay(propty)
      elif (identifier == "output"):
        lays[identifier] = self.init_inout_lay(propty)
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
  # def forward(self, inputs, training=None):
  #   # Inputs manipulation
  #   # -------------
  #   # [y0, t] = inputs
  #   # >> y0: initial conditions
  #   # >> t:  time
  #   y = _manipulate_input(self, inputs)
  #   # Kernel - Mass / Number density (OpNN)
  #   # -------------
  #   n0 = tf.gather(inputs[0], [self.mass_index], axis=-1)
  #   na = n0 * self.active_lays["mass"](y, training=training)
  #   nt = tf.concat([na, 0.5*(n0-na)], axis=-1)
  #   # Kernel - Temperature (OpNN)
  #   # -------------
  #   if (self.active_lays["temp"].return_latent_space == "branch"):
  #     pa, f_lat = self.active_lays["temp"](y, training=training)
  #   else:
  #     pa = self.active_lays["temp"](y, training=training)
  #   T = pa / (const.UKB * na)
  #   # Return number densities and temperature
  #   # -------------
  #   if (self.active_lays["temp"].return_latent_space == "branch"):
  #     return nt, T, pa, f_lat
  #   else:
  #     return nt, T

  def forward(self, inputs, training=None):
    # Inputs manipulation
    # -------------
    # [y0, t] = inputs
    # >> y0: initial conditions
    # >> t:  time
    y = _manipulate_input(self, inputs)
    # Kernel - Mass / Number density (OpNN)
    # -------------
    n0 = tf.gather(inputs[0], [self.mass_index], axis=-1)
    na = n0 * tf.math.sigmoid(self.active_lays["mass"](y, training=training))
    nm = 0.5*(n0-na)
    # Kernel - Temperature (OpNN)
    # -------------
    T = tf.math.exp(self.active_lays["temp"](y, training=training))
    # Return number densities and temperature
    # -------------
    return na, nm, T
