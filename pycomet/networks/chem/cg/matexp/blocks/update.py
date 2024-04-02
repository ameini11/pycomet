import tensorflow as tf

from pycomet import utils
from .utils import read_rates
from pycomet import operations as ops
from pycomet.config import get_config
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
  def call(self, T, training=None):
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
class Operators(nn_elem.Layer):
  """Compute the master equation operators for triatomic systems."""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates=None,
    use_corr=True,
    **kwargs
  ):
    super(Operators, self).__init__(**kwargs)
    # Reaction rates
    self.rates = tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.constant(x), axis=0),
      read_rates(rates)
    )
    # Corrections
    self.use_corr = use_corr

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs:
    # > na: atom number density
    # > Ts: precomputed temperature vectors (T,ln_T,ov_T)
    # > q:  updated partition functions at temperature T
    # > f:  state-wise corrective factors
    na, Ts, q, f = inputs
    # Compute reaction rates
    rates = self.compute_rates(Ts, q, f)
    # Compute operators
    return self.compute_ops(rates, na)

  def compute_rates(self, Ts, q, f):
    # Compute fwd reaction rates
    rates = self.compute_fwd_rates(Ts)
    # Correct fwd reaction rates
    if self.use_corr:
      fe, fd = self.compose_corr(f)
      rates["kef"] *= fe
      rates["kd"] *= fd
    # Compute bwd reaction rates
    rates = self.compute_bwd_rates(rates, q)
    return rates

  def compute_ops(self, rates, na):
    # > Excitation-Relaxation-Dissociation processes
    ke = rates["kef"] + tf.transpose(rates["keb"], perm=[0,2,1])
    ke_diag = tf.linalg.diag(tf.math.reduce_sum(ke, axis=-1))
    ke = tf.transpose(ke, perm=[0,2,1]) - ke_diag
    k = (ke - tf.linalg.diag(rates["kd"])) * tf.expand_dims(na, axis=-1)
    # > Recombination processes
    wr = rates["kr"] * na**3
    return k, wr

  # Reaction rates corrections
  # -----------------------------------
  def compose_corr(self, f):
    # Excitation processes
    fe = tf.math.exp(tf.einsum("nik,njk->nij", f, -f))
    # Dissociation processes
    fd = tf.math.exp(tf.einsum("nik->ni", f))
    return fe, fd

  # Reaction rates
  # -----------------------------------
  def compute_fwd_rates(self, Ts):
    # Precomputed temperature vectors
    T, ln_T, ov_T = tf.split(Ts, num_or_size_splits=3, axis=-1)
    # Loop over processes
    rates = {}
    for k in ("kef", "kd"):
      # Extract Arrhenius law parameters
      param = [self.rates[k][p] for p in ("A", "beta", "Ta")]
      # Apply Arrhenius law
      rates[k] = self._arrhenius(ln_T, ov_T, *param)
    return rates

  def _arrhenius(self, ln_T, ov_T, A, beta, Ta):
    # Reshape temperature vectors
    shape = [-1] + [1 for _ in A.shape[1:]]
    ln_T, ov_T = [tf.reshape(x, shape) for x in (ln_T, ov_T)]
    # Apply Arrhenius law
    return A * tf.math.exp(beta*ln_T - Ta*ov_T)

  def compute_bwd_rates(self, rates, q):
    # Unpack partition functions
    qa, qm = tf.split(q, num_or_size_splits=[1,q.shape[1]-1], axis=-1)
    # Excitation processes
    rates["keb"] = tf.einsum("nij,ni,nj->nij", rates["kef"], qm, 1/qm)
    # Dissociation processes
    rates["kr"] = rates["kd"] * qm * (1/(qa**2))
    return rates

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates),
      "use_corr": self.use_corr
    }
    base_config = super(Operators, self).get_config()
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
    **kwargs
  ):
    super(SolUpdate, self).__init__(**kwargs)
    # Operators layer
    self.operators = Operators(
      rates=rates,
      use_corr=use_corr
    )
    self.rates = self.operators
    # LinODE Layer
    self.linode = nn_elem.LinODE(
      use_eig=False,
      use_pinv=use_pinv,
      real_only=True,
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
    t, na, Ts, *q, nm_0, f = inputs
    q = tf.concat(q, axis=-1)
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
      "use_pinv": self.linode.use_pinv
    }
    base_config = super(SolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
