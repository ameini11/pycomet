from .blocks import *
from pycomet.networks.basenet import BaseNet


class MultiPoleME(BaseNet):
  """Master equation integral solution using multi-pole matrix exponential
  neural operator with a state-specific corrective factor."""

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    groups_graph,
    lays_name=None,
    lays_propty=None,
    noisy_io=None,
    inference=None,
    masking=False,
    force_siso=False,
    *args,
    **kwargs
  ):
    super(MultiPoleME, self).__init__(
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
    # Groups connectivity
    self.grp_conn = GroupsConnectivity(groups_graph)
    # Layers names
    if (lays_name is None):
      self.lays_name = self.lays_name + ("xform_fcorr", "timescales")
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("xform_fcorr", "timescales"),
      default=self.default_lays_propty
    )
    # Initializing layers
    self.active_lays = self.init_layers()
    self.init_preinput_lay()

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {
      "preproc": PreProcessor(),
      "connect": CurrentToLast(self.grp_conn.current_to_last)
    }
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == "xform_fcorr"):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier == "timescales"):
        lays[identifier] = self.init_timescales_lay(propty)
    return lays

  def init_timescales_lay(self, propty):
    lays = []
    for (i, ipropty) in enumerate(propty):
      lays.append(
        TimeScale(
          dim=ipropty["dim"],
          lays_propty=ipropty["lays_propty"],
          current_indices=self.grp_conn.current_to_last[str(i)],
          last_to_current=self.grp_conn.last_to_current[str(i+1)]
        )
      )
    return lays

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
    for lay in self.active_lays["timescales"]:
      y.append(lay((*qoi_t, nm_0, f_in), training=training))
    # Merge timescales
    return self.active_lays["connect"](y)
