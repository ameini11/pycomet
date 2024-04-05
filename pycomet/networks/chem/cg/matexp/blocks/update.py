import tensorflow as tf

from pycomet import utils
from .operators import Operators
from pycomet import operations as ops
from pycomet.config import get_config
from .utils import read_rates, split_rates
from pycomet.networks import elements as nn_elem
_CFG = get_config()

# HyperNet library
from hypernet.thermophysics import specie_thermo as st_mod


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PreProcessor(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    **kwargs
  ):
    super(PreProcessor, self).__init__(**kwargs)

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs:
    # > qoi_t: time-dependant QoIs
    #   > t:   time
    #   > na:  atom's number density
    #   > nm:  molecule's number density
    #   > T:   translational temperature
    # > nm_0:  molecule's initial distribution function
    # > f_in:  inputs for state-wise corrective factors
    qoi_t, nm_0, f_in = utils.unpack_list(inputs, nb_elem=3)
    t, na, nm, T = tf.split(qoi_t, num_or_size_splits=4, axis=-1)
    # Precomputed temperature vectors
    Ts = tf.concat([T, tf.math.log(T), 1/T], axis=-1)
    return t, na, Ts, nm, nm_0, f_in

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

  # Calling
  # ===================================
  def call(self, Ts, training=None):
    T = tf.gather(Ts, indices=[0], axis=-1)
    # Update partition functions: q = [qa, qm]
    q = []
    for st in self.sp_th.values():
      st.pf.reset()
      st.pf.update(T)
      q.append(st.pf.tot.Q)
    return q

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
    use_pinv=False,
    expm_use_pade=False,
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
      use_eig=False,
      use_pinv=use_pinv,
      real_only=True,
      expm_use_pade=expm_use_pade,
      expm_max_degree=None,
      expm_max_nb_squarings=None
    )

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    # -------------
    # > t:    time
    # > na:   atom's number density
    # > Ts:   precomputed temperature vectors (T,ln_T,ov_T)
    # > q:    updated partition functions at temperature T
    # > nm_0: molecule's initial distribution function
    # > f:    state-wise corrective factors
    t, na, Ts, q, nm_0, f = inputs
    # Compute operators
    # -------------
    k, wr = self.operators((na, Ts, q, f), training=training)
    # Compute updated solution
    # -------------
    nm_t = self.linode((t, nm_0, k, wr), training=training)
    # Compute relative molar fractions
    return (1/tf.math.reduce_sum(nm_t, axis=-1, keepdims=True)) * nm_t

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.operators.rates),
      "use_corr": self.operators.use_corr,
      "use_pinv": self.linode.use_pinv,
      "expm_use_pade": self.linode.expm_use_pade
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
    use_pinv=False,
    expm_use_pade=False,
    split_indices=None,
    **kwargs
  ):
    super(MultiSolUpdate, self).__init__(**kwargs)
    # Inputs
    self.rates = read_rates(rates=rates)
    self.use_corr = use_corr
    self.use_pinv = use_pinv
    self.expm_use_pade = expm_use_pade
    self.split_indices = split_indices
    # Sub-layers
    # -------------
    self.update = []
    multi_rates = split_rates(rates=self.rates, indices=self.split_indices)
    for irates in multi_rates:
      self.update.append(
        SolUpdate(
          rates=irates,
          use_corr=self.use_corr,
          use_pinv=self.use_pinv,
          expm_use_pade=self.expm_use_pade
        )
      )
    self.gather = nn_elem.Gather(indices=self.split_indices, axis=1)

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    t, na, Ts, qa, qm, nm_0, f = inputs
    # Split QoI
    qm, nm_0, f = [self.gather(z) for z in (qm, nm_0, f)]
    # Compute molecule's relative molar fractions
    xm_t = []
    for (i, nm_i) in enumerate(nm_0):
      # Compose inputs
      q_i = tf.concat([qa, qm[i]], axis=1)
      inputs = (t, na, Ts, q_i, nm_i, f[i])
      # Compute solution update
      xm_t.append(self.update[i](inputs, training=training))
    # Concatenate final solution
    return tf.concat(xm_t, axis=-1) if (len(xm_t) > 1) else xm_t[-1]

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates),
      "use_corr": self.use_corr,
      "use_pinv": self.use_pinv,
      "expm_use_pade": self.expm_use_pade,
      "split_indices": self.split_indices
    }
    base_config = super(MultiSolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
