import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from pycomet import utils
from pycomet.networks import gno
from pycomet import operations as ops
from pycomet.config import get_config
from silx.io.dictdump import h5todict
from pycomet.networks import elements as nn_elem
_CFG = get_config()

# HyperNet library
from hypernet import const
from hypernet.thermophysics import specie_thermo as st_mod


def _compose_feat(inputs, static_inputs, nb_elem):
  # Compose input features
  features = []
  # > Conditioning inputs (e.g., [t, na, T]])
  x = tf.expand_dims(inputs, axis=1)
  features.append(tf.repeat(x, repeats=nb_elem, axis=1))
  # > Static inputs
  n = tf.shape(inputs)[0]
  features.append(tf.repeat(static_inputs, repeats=n, axis=0))
  features = tf.concat(features, axis=-1)
  return features


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PFCorr(nn_elem.FFBlock):

  # Initialization
  # ===================================
  def __init__(
    self,
    layer_size=None,
    activation=["linear"],
    kernel_initializer=None,
    regularizer=None,
    dropout=dict(rate=0),
    layer_norm=None,
    static_prop=None,
    **kwargs
  ):
    super(PFCorr, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    # Static state/group-specific physical properties
    self.static_prop = gno.utils.read_attr(
      data=static_prop, identifier="static_prop"
    )
    self.nb_comp = self.static_prop.shape[-2]

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    # Compose input features
    features = _compose_feat(inputs, self.static_prop, self.nb_comp)
    # Predict corrective factor
    f = super(PFCorr, self).call(features, training=training)
    return (f,)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_prop": ops.to_numpy(self.static_prop)
    }
    base_config = super(PFCorr, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Operators(nn_elem.Layer):
  """Compute the master equation operators for triatomic systems."""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates=None,
    specie_thermos={
      "filename": None,
      "config": None
    },
    use_corr=True,
    **kwargs
  ):
    super(Operators, self).__init__(**kwargs)
    # Species thermodynamics
    self._sp_th_common = {
      "thermo_process": 1,
      "eos_model": "PerfectGas"
    }
    self.sp_th = self._build_sp_th(specie_thermos)
    # Reaction rates
    self.rates = rates
    if isinstance(self.rates, str):
      self.rates = h5todict(h5file=self.rates)
    self.rates = tf.nest.map_structure(
      lambda x: tf.constant(np.expand_dims(ops.to_numpy(x).squeeze(), axis=0)),
      self.rates
    )
    # Corrections
    self.use_corr = use_corr

  def _build_sp_th(self, specie_thermos):
    specie_thermos.update(self._sp_th_common)
    sp_th = st_mod.init_specie_thermos(**specie_thermos)
    st_mod.build_specie_thermos(sp_th)
    return sp_th

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    # Extract inputs
    na, T, f = utils.unpack_list(inputs, nb_elem=3)
    # Update species thermodynamics
    st_mod.update_specie_thermos(self.sp_th, T)
    # Compute reaction rates
    rates = self.compute_rates(T, f)
    # Compute operators
    return self.compute_ops(rates, na)

  def compute_rates(self, T, f):
    # Compute fwd reaction rates
    rates = self.compute_fwd_rates(T)
    # Correct fwd reaction rates
    if self.use_corr:
      fe, fd = self.compose_corr(f)
      rates["kef"] *= fe
      rates["kd"] *= fd
    # Compute bwd reaction rates
    rates = self.compute_bwd_rates(rates)
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
  def compute_fwd_rates(self, T):
    # Precompute temperature vectors
    ln_T, ov_T = tf.math.log(T), 1/T
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

  def compute_bwd_rates(self, rates):
    qa, qm = [st.pf.tot.Q for st in self.sp_th.values()]
    # Excitation processes
    rates["keb"] = tf.einsum("nij,ni,nj->nij", rates["kef"], qm, 1/qm)
    # Dissociation processes
    rates["kr"] = rates["kd"] * qm * (1/(qa**2))
    return rates

  # Configuration update
  # ===================================
  def get_config(self):
    sp_th_cfg = tf.nest.map_structure(lambda st: st.get_config(), self.sp_th)
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates),
      "specie_thermos": {"config": sp_th_cfg},
      "use_corr": self.use_corr
    }
    base_config = super(Operators, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class Solution(nn_elem.Layer):
  """Matrix exponential-based solution for the master equation."""

  # Initialization
  # ===================================
  def __init__(
    self,
    # Operators Layer
    rates=None,
    specie_thermos={
      "filename": None,
      "config": None
    },
    use_corr=True,
    use_tril_corr=False,
    use_state_corr=False,
    use_state_corr_lin=True,
    # LinODE Layer
    use_eig=False,
    use_pinv=False,
    real_only=True,
    expm_max_degree=None,
    expm_max_nb_squarings=None,
    **kwargs
  ):
    super(Solution, self).__init__(**kwargs)
    # Operators layer
    self.operators = Operators(
      rates=rates,
      specie_thermos=specie_thermos,
      use_corr=use_corr,
      use_tril_corr=use_tril_corr,
      use_state_corr=use_state_corr,
      use_state_corr_lin=use_state_corr_lin
    )
    self.linode = nn_elem.LinODE(
      use_eig=use_eig,
      use_pinv=use_pinv,
      real_only=real_only,
      expm_max_degree=expm_max_degree,
      expm_max_nb_squarings=expm_max_nb_squarings
    )

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    qt, nm_0, f = utils.unpack_list(inputs, nb_elem=3)
    # Extract time-dependent QoI
    # t, na, T = tf.split(qt, num_or_size_splits=3, axis=1)
    t, na, nm, T, em = tf.split(qt, num_or_size_splits=5, axis=1)
    # Compute operators
    k, wr = self.operators((na, T, f), training=training)
    # Compute solution
    nm_t = self.linode((t, nm_0, k, wr), training=False)
    # Mass conservation
    # nm_t *= nm / tf.math.reduce_sum(nm_t, axis=-1, keepdims=True)
    return nm_t
