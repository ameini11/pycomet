from .correction import Correction
from .connectivity import LastToCurrent
from .update import PFUpdate, MultiSolUpdate
from pycomet.networks.basenet import BaseNet


class TimeScale(BaseNet):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    lays_name=None,
    lays_propty=None,
    current_indices=None,
    last_to_current=None,
    *args,
    **kwargs
  ):
    # Layers names
    super(TimeScale, self).__init__(
      dim=dim,
      lays_name=lays_name,
      lays_propty=lays_propty,
      *args,
      **kwargs
    )
    # Timescale connectivity
    self.current_indices = current_indices
    self.last_to_current = last_to_current
    # Layers names
    if (lays_name is None):
      self.lays_name = self.lays_name + ("pf_update", "correct", "sol_update")
    else:
      self.lays_name = lays_name
    # Layers properties
    self.lays_propty = self.check_layers(
      layers=self.lays_propty,
      not_opt=("pf_update", "correct", "sol_update"),
      default=self.default_lays_propty
    )
    # Initializing
    self.active_lays = self.init_layers()
    self.init_preinput_lay()

  # Layers initialization
  # -----------------------------------
  def init_layers(self):
    lays = {}
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == "pf_update"):
        lays[identifier] = self.init_lay_kernel(identifier, propty)
      elif (identifier == "correct"):
        lays[identifier] = self.init_lay_kernel(identifier, propty)
      elif (identifier == "sol_update"):
        propty["kwargs"]["split_indices"] = self.current_indices
        lays[identifier] = self.init_lay_kernel(identifier, propty)
    lays["connect"] = LastToCurrent(self.last_to_current)
    return lays

  def init_lay_kernel(self, identifier, propty):
    propty["lay_id"] = {
      "pf_update": PFUpdate,
      "correct": Correction,
      "sol_update": MultiSolUpdate
    }[identifier]
    return self.init_lay(identifier, propty)

  # Calling
  # ===================================
  def forward(self, inputs, training=None):
    # Extract inputs:
    # > t:    time
    # > na:   atom's number density
    # > Ts:   precomputed temperature vectors (T, ln_T, ov_T)
    # > nm_0: molecule's initial distribution function
    # > f_in: inputs for state-wise corrective factors
    t, na, Ts, nm_0, f_in = inputs
    # Update partition functions
    qa, qm = self.active_lays["pf_update"](Ts, training=training)
    # Merge states
    nm_0 = self.active_lays["connect"](nm_0, training=training)
    # Compute corrections
    f = self.active_lays["correct"](f_in, training=training)
    # Compute solution
    inputs = (t, na, Ts, qa, qm, nm_0, f)
    return self.active_lays["sol_update"](inputs, training=training)
