import tensorflow as tf

from pycomet import utils
from .operators import Operators
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()

# HyperNet library
from hypernet.thermophysics import specie_thermo as st_mod


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class ThermoUpdate(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    specie_thermos={
      "filename": None,
      "config": None
    },
    **kwargs
  ):
    super(ThermoUpdate, self).__init__(**kwargs)
    # Species thermodynamics
    specie_thermos.update({
      "thermo_process": 1,
      "eos_model": "PerfectGas"
    })
    self.sp_th = st_mod.init_specie_thermos(**specie_thermos)
    st_mod.build_specie_thermos(self.sp_th)
    # Species identifiers
    self.species = {}
    for st in self.sp_th.values():
      if (st.specie.nb_comp > 1):
        self.species["molecule"] = st.specie.name
      else:
        self.species["atom"] = st.specie.name

  # Calling
  # ===================================
  def call(self, T, species=None, qois=["e_int", "q_tot"], training=None):
    T = tf.gather(T, indices=[0], axis=-1)
    if (species is None):
      sp = [self.species["atom"], self.species["molecule"]]
    elif (species in ("atom", "molecule")):
      sp = self.species[species]
    else:
      sp = species
    for s in tf.nest.flatten(sp):
      for qoi in tf.nest.flatten(qois):
        self.update(T, s, qoi)

  def update(self, T, species, qoi):
    if (qoi == "e_int"):
      # Internal energy
      self.sp_th[species].thermo.reset()
      self.sp_th[species].thermo.update(T)
    else:
      # Partition functions
      self.sp_th[species].pf.reset()
      self.sp_th[species].pf.update(T)

  def get_updated(self, species, qoi):
    if (species in ("atom", "molecule")):
      species = self.species[species]
    if (qoi == "e_int"):
      # Internal energy
      return self.sp_th[species].thermo.int.e
    elif (qoi == "q_tot"):
      # Partition functions
      return self.sp_th[species].pf.tot.Q
    else:
      utils.raise_err(ValueError)

  # Configuration update
  # ===================================
  def get_config(self):
    sp_th_cfg = tf.nest.map_structure(lambda st: st.get_config(), self.sp_th)
    config = {
      "specie_thermos": {"config": sp_th_cfg}
    }
    base_config = super(ThermoUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class SolUpdate(nn_elem.Layer):
  """Matrix exponential-based update for the master equation."""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates=None,
    use_corr=True,
    use_pade=False,
    use_expmv=False,
    max_degree=False,
    max_nb_squarings=False,
    add_inv_loss=False,
    inv_loss_weight=1e-3,
    **kwargs
  ):
    super(SolUpdate, self).__init__(**kwargs)
    # Operators layer
    self.operators = Operators(
      rates=rates,
      use_corr=use_corr
    )
    # LinODE Layer
    self.linode = nn_elem.LinODE(
      use_pade=use_pade,
      use_expmv=use_expmv,
      max_degree=max_degree,
      max_nb_squarings=max_nb_squarings,
      add_inv_loss=add_inv_loss,
      inv_loss_weight=inv_loss_weight
    )

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    t, na, nm, T, *q, nmi_0, f = inputs
    # Compute operators
    k, w = self.operators((na, T, *q, f), training=training)
    # Compute state-wise number densities
    nmi = self.linode((t, nmi_0, k, w), training=training)
    # Ensure mass conservation
    nmi *= (nm / tf.math.reduce_sum(nmi, axis=-1, keepdims=True))
    return nmi

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.operators.rates),
      "use_corr": self.operators.use_corr,
      "use_pade": self.linode.use_pade,
      "use_expmv": self.linode.use_expmv,
      "max_degree": self.linode.max_degree,
      "max_nb_squarings": self.linode.max_nb_squarings,
      "add_inv_loss": self.linode.add_inv_loss,
      "inv_loss_weight": self.linode.inv_loss_weight
    }
    base_config = super(SolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
