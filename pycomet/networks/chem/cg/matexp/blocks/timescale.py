import copy
import tensorflow as tf

from .utils import split_rates
from .correction import Correction
from .update import PFUpdate, SolUpdate
from .connectivity import LastToCurrent
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
    lays = {
      "connect": LastToCurrent(self.last_to_current),
      "gather": nn_elem.Gather(indices=self.current_indices, axis=1)
    }
    for (identifier, propty) in self.lays_propty.items():
      if (identifier == "pf_update"):
        lays[identifier] = self.init_lay_kernel(identifier, propty)
      elif (identifier == "correct"):
        lays[identifier] = self.init_lay_kernel(identifier, propty)
      elif (identifier == "sol_update"):
        lays[identifier] = self.init_lay_sol_update(identifier, propty)
    return lays

  # def init_lay_correct(self, identifier, propty):
  #   static_prop = split_static_prop(
  #     prop=propty["kwargs"]["static_prop"],
  #     indices=self.current_indices
  #   )
  #   lays = []
  #   for istatic_prop in static_prop:
  #     ipropty = tf.nest.map_structure(copy.deepcopy, propty)
  #     ipropty["kwargs"]["static_prop"] = istatic_prop
  #     lays.append(self.init_lay_kernel(identifier, ipropty))
  #   return lays

  def init_lay_sol_update(self, identifier, propty):
    rates = split_rates(
      rates=propty["kwargs"]["rates"],
      indices=self.current_indices
    )
    lays = []
    for irates in rates:
      ipropty = tf.nest.map_structure(copy.deepcopy, propty)
      ipropty["kwargs"]["rates"] = irates
      lays.append(self.init_lay_kernel(identifier, ipropty))
    return lays

  def init_lay_kernel(self, identifier, propty):
    propty["lay_id"] = {
      "pf_update": PFUpdate,
      "correct": Correction,
      "sol_update": SolUpdate
    }[identifier]
    return self.init_lay(identifier, propty)

  # Calling
  # ===================================
  def forward(self, inputs, training=None):
    # Extract inputs
    # -------------
    # > t:    time
    # > na:   atom's number density
    # > Ts:   precomputed temperature vectors (T, ln_T, ov_T)
    # > nm_0: molecule's initial distribution function
    # > f_in: inputs for state-wise corrective factors
    t, na, Ts, nm_0, f_in = inputs
    # Update partition functions
    # -------------
    T = nn_elem.Gather(indices=[0], axis=-1)(Ts)
    qa, qm = self.active_lays["pf_update"](T, training=training)
    # Merge states
    # -------------
    # Micro-states (fastest/last timescale) to macro-states (current timescale)
    nm_0 = self.active_lays["connect"](nm_0, training=training)
    # Compute corrections
    # -------------
    f = self.active_lays["correct"](f_in, training=training)
    # Compute solution
    # -------------
    qm, nm_0, f = [self.active_lays["gather"](z) for z in (qm, nm_0, f)]
    # Relative molar fractions (molecule)
    xm_t = []
    for (i, nm_i) in enumerate(nm_0):
      # Compose inputs
      inputs = (t, na, Ts, qa, qm[i], nm_i, f[i])
      # Compute solution update
      xm_i = self.active_lays["sol_update"][i](inputs, training=training)
      # Store solution update
      xm_t.append(xm_i)
    return tf.concat(xm_t, axis=-1) if (len(xm_t) > 1) else xm_t[-1]
