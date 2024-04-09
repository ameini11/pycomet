from .blocks import *
from pycomet import utils
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
    # Extract inputs
    # -------------
    # > t:    time
    # > na:   atom's number density
    # > nm:   molecule's number density
    # > T:    translational temperature
    # > nm_0: molecule's initial distribution function
    # > f:    inputs for state-wise corrective factors
    t, na, nm, T, nm_0, f = utils.unpack_list(inputs, nb_elem=6)
    # Update partition functions
    # -------------
    qa, qm = self.active_lays["update_pf"](T)
    # Compute state-wise latent space
    # -------------
    # > Transforming
    f = self.active_lays["xform_fcorr"](f, training=training)
    # > Computing
    f = self.active_lays["compute_fcorr"](f, training=training)
    # Compute operators
    # -------------
    k, w = self.active_lays["operators"]((na, T, qa, qm, f))
    # Loop over timescales
    # -------------
    y = [nm]
    for lay in self.active_lays["update_sols"]:
      y.append(lay((t, nm_0, k, w, qm)))
    # Merge timescales
    # -------------
    return self.active_lays["connect"](y)
