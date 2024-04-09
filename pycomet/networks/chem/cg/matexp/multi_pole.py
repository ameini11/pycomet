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
    default_lays_name = (
      "pf_update", "fcorr_xform", "fcorr_compute", \
        "operators", "sol_updates", "connect"
    )
    if (lays_name is None):
      self.lays_name = self.lays_name + default_lays_name
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=default_lays_name,
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
      if (identifier == "fcorr_xform"):
        lays[identifier] = self.init_lay(identifier, propty)
      elif (identifier in ("pf_update", "fcorr_compute", "operators")):
        lays[identifier] = self.init_kernel_layers(identifier, propty)
      elif (identifier == "sol_updates"):
        lays[identifier] = self.init_sol_update_lays(identifier, propty)
    lays["connect"] = CurrentToNext(self.grp_conn.current_to_next["indices"])
    return lays

  def init_kernel_layers(self, identifier, propty):
    propty["lay_id"] = {
      "pf_update": PFUpdate,
      "fcorr_compute": Correction,
      "operators": Operators
    }[identifier]
    return self.init_lay(identifier, propty)

  def init_sol_update_lays(self, identifier, propty):
    lays = []
    for (i, ipropty) in enumerate(propty):
      isplit = self.grp_conn.last_to_current["conn"][str(i)]
      icompr = self.grp_conn.last_to_current["indices"][str(i+1)]
      ipropty["lay_id"] = SolUpdate
      ipropty["kwargs"]["indices_split"] = isplit if (i != 0) else None
      ipropty["kwargs"]["indices_compress"] = icompr
      lays.append(self.init_lay(identifier, ipropty))
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
    qa, qm = self.active_lays["pf_update"](T)
    # Compute state-wise latent space
    # -------------
    # > Transforming
    f = self.active_lays["fcorr_xform"](f, training=training)
    # > Computing
    f = self.active_lays["fcorr_compute"](f, training=training)
    # Compute operators
    # -------------
    k, w = self.active_lays["operators"]((na, T, qa, qm, f))
    # Update solutions
    # -------------
    y = [nm]
    for lay in self.active_lays["sol_updates"]:
      y.append(lay((t, nm_0, k, w, qm)))
    # Merge solutions
    # -------------
    return self.active_lays["connect"](y)
