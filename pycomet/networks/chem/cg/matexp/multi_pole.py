from pycomet import utils
from pycomet.networks.basenet import BaseNet


class MultiPoleME(BaseNet):
  """Master equation solution using a multi-pole matrix exponential-based
  neural operator with a state-specific corrective factor."""

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
    super(MultiPoleME, self).__init__(
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
      self.lays_name = self.lays_name + ("corr_factor", "update")
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("corr_factor", "update"),
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
      elif (identifier in ("corr_factor", "update")):
        lays[identifier] = self.init_lay(identifier, propty)
    # lays.update(self.init_kernel_layers())
    return lays

  # def init_kernel_layers(self):
  #   lay_ids = {
  #     "update": "FwdEulerME"
  #   }
  #   lays = {}
  #   for (identifier, propty) in self.lays_propty.items():
  #     lay_id = lay_ids.get(identifier, None)
  #     if (lay_id is not None):
  #       lays[identifier] = self.init_lay(
  #         identifier, propty, lay_id=lay_id, default_lay_id=lay_id
  #       )
  #   return lays

  # Calling
  # ===================================
  def forward(self, inputs, training=None):
    # Preprocess inputs
    x = self.active_lays["preproc"](inputs, training=training)
    *qoi_t, nm, nm_0, f_in = x
    # Transform inputs for state-wise corrective factors
    f_in = self.active_lays["xform_fcorr"](f_in, training=training)
    # Loop over timescales
    y = [nm]
    for ts_lay in self.active_lays["timescales"]:
      y.append(ts_lay((*qoi_t, nm_0, f_in), training=training))
    # Merge timescales
    return self.active_lays["connect"](y)
