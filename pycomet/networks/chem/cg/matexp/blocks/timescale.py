from .correction import Correction
from .update import PFUpdate, MultiSolUpdate
from pycomet.networks.basenet import BaseNet
from pycomet.networks import elements as nn_elem


class TimeScale(BaseNet):

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    lays_name=None,
    lays_propty=None,
    indices_split=None,
    indices_reduce=None,
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
    self.indices_split = indices_split
    self.indices_reduce = indices_reduce
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
        propty["kwargs"]["indices_split"] = self.indices_split
        lays[identifier] = self.init_lay_kernel(identifier, propty)
    lays["reduce"] = nn_elem.Reduction(self.indices_reduce)
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
    # Extract inputs
    t, na, T, qa, nm_0, f = inputs
    # Merge states
    nm_0 = self.active_lays["reduce"](nm_0, axis=1)
    # Update partition functions
    qm = self.active_lays["pf_update"](T, species="molecule")
    # Compute state-wise latent space
    f = self.active_lays["correct"](f, training=training)
    # Compute solution
    return self.active_lays["sol_update"]((t, na, T, qa, qm, nm_0, f))
