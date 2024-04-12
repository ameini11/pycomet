import tensorflow as tf

from .operators import Operators
from pycomet import operations as ops
from pycomet.config import get_config
from .utils import read_rates, split_rates
from pycomet.networks import elements as nn_elem
_CFG = get_config()

# HyperNet library
from hypernet.thermophysics import specie_thermo as st_mod


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class TempUpdate(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    **kwargs
  ):
    super(TempUpdate, self).__init__(**kwargs)

  # Calling
  # ===================================
  def call(self, T, training=None):
    # Precomputed temperature vectors
    return tf.concat([T, tf.math.log(T), 1/T], axis=-1)


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PFUpdate(nn_elem.Layer):
  """Update total partition functions for triatomic systems."""

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
    super(PFUpdate, self).__init__(**kwargs)
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
  def call(self, T, species=None, training=None):
    T = tf.gather(T, indices=[0], axis=-1)
    if (species is None):
      sp = [self.species["atom"], self.species["molecule"]]
    elif (species in ("atom", "molecule")):
      sp = self.species[species]
    else:
      sp = species
    q = [self.update(T, s) for s in tf.nest.flatten(sp)]
    return q[0] if (len(q) == 1) else q

  def update(self, T, species):
    self.sp_th[species].pf.reset()
    self.sp_th[species].pf.update(T)
    return self.sp_th[species].pf.tot.Q

  # Configuration update
  # ===================================
  def get_config(self):
    sp_th_cfg = tf.nest.map_structure(lambda st: st.get_config(), self.sp_th)
    config = {
      "specie_thermos": {"config": sp_th_cfg}
    }
    base_config = super(PFUpdate, self).get_config()
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
    use_eig=False,
    use_pinv=False,
    expm_use_pade=False,
    add_det_loss=False,
    det_loss_weight=1e-3,
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
      use_eig=use_eig,
      use_pinv=use_pinv,
      real_only=True,
      expm_use_pade=expm_use_pade,
      expm_max_degree=None,
      expm_max_nb_squarings=None,
      add_det_loss=add_det_loss,
      det_loss_weight=det_loss_weight
    )

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    t, na, T, *q, nm_0, f = inputs
    # Compute operators
    k, w = self.operators((na, T, *q, f))
    # Compute relative molar fractions
    nm_t = self.linode((t, nm_0, k, w))
    return (1/tf.math.reduce_sum(nm_t, axis=-1, keepdims=True)) * nm_t

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.operators.rates),
      "use_corr": self.operators.use_corr,
      "use_eig": self.linode.use_eig,
      "use_pinv": self.linode.use_pinv,
      "expm_use_pade": self.linode.expm_use_pade,
      "add_det_loss": self.linode.add_det_loss,
      "det_loss_weight": self.linode.det_loss_weight
    }
    base_config = super(SolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class MultiSolUpdate(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    rates=None,
    use_corr=True,
    use_eig=False,
    use_pinv=False,
    add_det_loss=False,
    det_loss_weight=1e-3,
    expm_use_pade=False,
    indices_split=None,
    **kwargs
  ):
    super(MultiSolUpdate, self).__init__(**kwargs)
    # Inputs
    self.rates = read_rates(rates=rates)
    self.use_corr = use_corr
    self.use_eig = use_eig
    self.use_pinv = use_pinv
    self.add_det_loss = add_det_loss
    self.det_loss_weight = det_loss_weight
    self.expm_use_pade = expm_use_pade
    self.indices_split = indices_split
    # Sub-layers
    # -------------
    self.update = []
    rates_split = split_rates(rates=self.rates, indices=self.indices_split)
    for irates in rates_split:
      self.update.append(
        SolUpdate(
          rates=irates,
          use_corr=self.use_corr,
          use_eig=self.use_eig,
          use_pinv=self.use_pinv,
          expm_use_pade=self.expm_use_pade,
          add_det_loss=self.add_det_loss,
          det_loss_weight=self.det_loss_weight
        )
      )
    self.gather = nn_elem.Gather(indices=self.indices_split, axis=1)

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    t, na, T, qa, qm, nm_0, f = inputs
    # Split QoI
    qm, nm_0, f = [self.gather(z) for z in (qm, nm_0, f)]
    # Compute molecule's relative molar fractions
    xm_t = []
    for (i, nm_i) in enumerate(nm_0):
      # Compute solution update
      xm_t.append(self.update[i]((t, na, T, qa, qm[i], nm_i, f[i])))
    # Concatenate final solution
    return tf.concat(xm_t, axis=-1) if (len(xm_t) > 1) else xm_t[-1]

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates),
      "use_corr": self.use_corr,
      "use_eig": self.use_eig,
      "use_pinv": self.use_pinv,
      "add_det_loss": self.add_det_loss,
      "det_loss_weight": self.det_loss_weight,
      "expm_use_pade": self.expm_use_pade,
      "indices_split": self.indices_split
    }
    base_config = super(MultiSolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
