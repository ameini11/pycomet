from pycomet import utils
from pycomet.networks.basenet import BaseNet


class MEIntCorrFE(BaseNet):
  """Master equation integral using a forward Euler-type update 
  combined with a state or group-specific corrective factor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    lays_name=None,
    lays_propty=None,
    noisy_io=None,
    inference=None,
    *args,
    **kwargs
  ):
    super(MEIntCorrFE, self).__init__(
      dim=dim,
      lays_name=lays_name,
      lays_propty=lays_propty,
      noisy_io=noisy_io,
      inference=inference,
      masking=False,
      force_siso=False,
      *args,
      **kwargs
    )
    self.process_layers(lays_name)

  def process_layers(self, lays_name):
    # Layers names
    if (lays_name is None):
      self.lays_name = self.lays_name + ("corr_factor", "fwd_euler")
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("corr_factor", "fwd_euler"),
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
      if (identifier in ("input", "output")):
        lays[identifier] = self.init_inout_lay(propty)
      elif (identifier == "corr_factor"):
        lays[identifier] = self.init_lay(identifier, propty)
    lays.update(self.init_kernel_layers())
    return lays

  def init_kernel_layers(self):
    lay_ids = {
      "fwd_euler": "FwdEulerME"
    }
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      lay_id = lay_ids.get(identifier, None)
      if (lay_id is not None):
        lays[identifier] = self.init_lay(
          identifier, propty, lay_id=lay_id, default_lay_id=lay_id
        )
    return lays

  # Calling
  # ===================================
  def forward(self, inputs, training=None):
    # Unpack inputs
    qt, x, f = utils.unpack_list(inputs, nb_elem=3)
    # Input nodes manipulation
    f = self._manipulate_inout(
      inputs=dict(x=f, y=f), identifier="input", training=training
    )
    # State/group-specific corrective factor
    f = self.active_lays["corr_factor"](f, training=training)
    # Forward Euler-type update
    y = self.active_lays["fwd_euler"]((qt, x, f), training=training)
    # Output nodes manipulation
    y = self._manipulate_inout(
      inputs=dict(x=x, y=y), identifier="output", training=training
    )
    return y
