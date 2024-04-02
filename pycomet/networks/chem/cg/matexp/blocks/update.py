import tensorflow as tf

from .utils import read_rates
from pycomet import operations as ops
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
_CFG = get_config()

# HyperNet library
from hypernet.thermophysics import specie_thermo as st_mod


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
    split_indices=None,
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
    # Indices for splitting states into different clusters
    self.split_indices = split_indices

  # Calling
  # ===================================
  def call(self, Ts, training=None):
    # Extract temperature T from precomputed temperature vectors (T,ln_T,ov_T)
    T = tf.gather(Ts, [0], axis=-1)
    # Update partition functions: q = [qa, qm]
    q = []
    for st in self.sp_th.values():
      st.pf.reset()
      st.pf.update(T)
      q.append(st.pf.tot.Q)
    # Split partition functions
    qa, qm = q
    qm = [tf.gather(qm, i, axis=-1) for i in self.split_indices]
    return qa, *qm

  # Configuration update
  # ===================================
  def get_config(self):
    sp_th_cfg = tf.nest.map_structure(lambda st: st.get_config(), self.sp_th)
    config = {
      "specie_thermos": {"config": sp_th_cfg},
      "split_indices": self.split_indices
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
    **kwargs
  ):
    super(Operators, self).__init__(**kwargs)
    # Reaction rates
    self.rates = tf.nest.map_structure(
      lambda x: tf.expand_dims(tf.constant(x), axis=0),
      read_rates(rates)
    )

  # Calling
  # ===================================
  def call(self, inputs, use_corr=True, training=None):
    # Extract inputs:
    # > na: atom number density
    # > Ts: precomputed temperature vectors (T,ln_T,ov_T)
    # > q:  updated partition functions at temperature T
    # > f:  state-wise corrective factors
    na, Ts, q, f = inputs
    # Compute reaction rates
    rates = self.compute_rates(Ts, q, f, use_corr)
    # Compute operators
    return self.compute_ops(rates, na)

  def compute_rates(self, Ts, q, f, use_corr):
    # Compute fwd reaction rates
    rates = self.compute_fwd_rates(Ts)
    # Correct fwd reaction rates
    if use_corr:
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
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates)
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
    use_pinv=False,
    **kwargs
  ):
    super(SolUpdate, self).__init__(**kwargs)
    # Operators layer
    self.operators = Operators(
      rates=rates
    )
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
    t, na, Ts, q, nm_0, f = inputs
    # Compute operators
    # -------------
    k, wr = self.operators((na, Ts, q, f), training=training)
    # Compute updated solution
    # -------------
    nm_t = self.linode((t, nm_0, k, wr), training=training)
    # Compute relative molar fractions
    return (1/tf.math.reduce_sum(nm_t, axis=-1, keepdims=True)) * nm_t
