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
    f = tf.squeeze(f, axis=-1)
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
class FwdExcRateCorr(nn_elem.FFBlock):

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
    super(FwdExcRateCorr, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    # Static state/group-specific physical properties
    if isinstance(static_prop, str):
      static_prop = np.loadtxt(static_prop)
    self.static_prop = ops.to_numpy(static_prop).squeeze()
    # Number of states/groups
    nb_comp, np_prop = self.static_prop.shape
    # Forward rates
    # > Create tensor
    fwd_rates = np.zeros((nb_comp,nb_comp,np_prop*2))
    # > Get ij lower triangular indices only
    i, j = np.tril_indices(nb_comp, k=-1)
    # > Fill tensor with ij properties
    pi = self.static_prop[i]
    pj = self.static_prop[j]
    fwd_rates[i,j,:] = np.hstack([pi, pj])
    # > Convert to tensorflow object
    fwd_rates = tf.constant(fwd_rates[1:,:-1])
    fwd_rates = tf.transpose(fwd_rates, perm=[2,0,1])
    fwd_rates_vec = tfp.math.fill_triangular_inverse(fwd_rates)
    fwd_rates_vec = tf.transpose(fwd_rates_vec)
    self.fwd_rates_vec = tf.expand_dims(fwd_rates_vec, axis=0)
    self.nb_fwd_rates = self.fwd_rates_vec.shape[1]
    # > Create padding operator
    self.q = tf.constant(np.eye(nb_comp)[:nb_comp-1])

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    # Compose input features
    features = _compose_feat(inputs, self.fwd_rates_vec, self.nb_fwd_rates)
    # Predict corrective factor
    f = super(FwdExcRateCorr, self).call(features, training=training)
    f = self.make_tril(f)
    return f

  def make_tril(self, f):
    """Make lower triangular."""
    f = tf.squeeze(f, axis=-1)
    # Fill lower triangular (diagonal included)
    f = tfp.math.fill_triangular(f)
    # Padding last row/column
    f = tf.einsum("...ik,...ij,...jl->...kl", self.q, f, self.q)
    # Rolling last row up
    f = tf.roll(f, shift=[1], axis=[1])
    return f

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_prop": ops.to_numpy(self.static_prop)
    }
    base_config = super(FwdExcRateCorr, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class FwdDisRateCorr(nn_elem.FFBlock):

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
    super(FwdDisRateCorr, self).__init__(
      layer_size=layer_size,
      activation=activation,
      kernel_initializer=kernel_initializer,
      regularizer=regularizer,
      dropout=dropout,
      layer_norm=layer_norm,
      **kwargs
    )
    # Static state/group-specific physical properties
    if isinstance(static_prop, str):
      static_prop = np.loadtxt(static_prop)
    self.static_prop = ops.to_numpy(static_prop).squeeze()
    # Forward rates
    fwd_rates_vec = tf.constant(self.static_prop)
    self.fwd_rates_vec = tf.expand_dims(fwd_rates_vec, axis=0)
    self.nb_fwd_rates = self.fwd_rates_vec.shape[1]

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    # Compose input features
    features = _compose_feat(inputs, self.fwd_rates_vec, self.nb_fwd_rates)
    # Predict corrective factor
    f = super(FwdDisRateCorr, self).call(features, training=training)
    return tf.squeeze(f, axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "static_prop": ops.to_numpy(self.static_prop)
    }
    base_config = super(FwdDisRateCorr, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class FwdRateCorr(object):

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
    trainable=True,
    **kwargs
  ):
    self.lays = {}
    for (k, lay) in (("exc", FwdExcRateCorr), ("dis", FwdDisRateCorr)):
      self.lays[k] = lay(
        layer_size=layer_size[k],
        activation=activation[k],
        kernel_initializer=kernel_initializer,
        regularizer=regularizer,
        dropout=dropout,
        layer_norm=layer_norm,
        static_prop=static_prop,
        trainable=trainable if isinstance(trainable, bool) else trainable[k],
        **kwargs
      )

  # Calling
  # ===================================
  def __call__(self, inputs, training=False):
    return self.call(inputs, training=training)

  def call(self, inputs, training=False):
    f = []
    for k in ("exc", "dis"):
      f.append(self.lays[k](inputs, training=training))
    return tuple(f)


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class MEOperators(nn_elem.Layer):
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
    use_state_corr=False,
    **kwargs
  ):
    super(MEOperators, self).__init__(**kwargs)
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
    # Use correction
    self.use_corr = use_corr
    # Use node-wise correction
    self.use_state_corr = use_state_corr

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
      rates = self.correct_fwd_rates(rates, f)
    # Compute bwd reaction rates
    return self.compute_bwd_rates(rates)

  def compute_ops(self, rates, na):
    # > Excitation-Relaxation-Dissociation processes
    ke = rates["kef"] + tf.transpose(rates["keb"], perm=[0,2,1])
    ke_diag = tf.linalg.diag(tf.math.reduce_sum(ke, axis=-1))
    ke = tf.transpose(ke, perm=[0,2,1]) - ke_diag
    k = (ke - tf.linalg.diag(rates["kd"])) * tf.expand_dims(na, axis=-1)
    # > Recombination processes
    wr = rates["kr"] * na**3
    return k, wr

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

  def correct_fwd_rates(self, rates, f):
    # Excitation processes
    fe = tf.einsum("ni,nj->nij", f[0], -f[0]) if self.use_state_corr else f[0]
    rates["kef"] = rates["kef"] * tf.math.exp(fe)
    # Dissociation processes
    fd = f[0] if self.use_state_corr else f[1]
    rates["kd"] = rates["kd"] * tf.math.exp(fd)
    return rates

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
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates),
      "specie_thermos": {"config": self.sp_th.get_config()},
      "use_state_corr": self.use_state_corr
    }
    base_config = super(MEOperators, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class FwdEulerME(nn_elem.Layer):
  """Forward Euler-type update for the master equation."""

  # Initialization
  # ===================================
  def __init__(
    self,
    rates=None,
    molecule=None,
    specie_thermos={
      # N-groups model
      "Ng": {
        "filename": None,
        "config": None
      },
      # 1-group model
      "1g": {
        "filename": None,
        "config": None
      }
    },
    use_corr=True,
    use_state_corr=False,
    boltz_rescale_order=0,
    **kwargs
  ):
    super(FwdEulerME, self).__init__(**kwargs)
    # Operators layer
    self.ops_lay = MEOperators(
      rates=rates,
      specie_thermos=specie_thermos["Ng"],
      use_corr=use_corr,
      use_state_corr=use_state_corr
    )
    # Species thermodynamics
    self.molecule = molecule
    self.sp_th = {
      "Ng": self.ops_lay.sp_th,
      "1g": self.ops_lay._build_sp_th(specie_thermos["1g"])
    }
    self.th_int = {
      k: self.sp_th[k][self.molecule].thermo.int for k in ("Ng", "1g")
    }
    self.mk = self.sp_th["Ng"][self.molecule].specie.m / const.UKB
    # Boltzmann-based rescaling order
    self.boltz_rescale_order = boltz_rescale_order

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    qt, nm0, f = utils.unpack_list(inputs, nb_elem=3)
    # Extract time-dependent QoI
    t, na, nm, T, em = tf.split(qt, num_or_size_splits=5, axis=1)
    # Compute operators
    k, wr = self.ops_lay((na, T, f))
    # Boltzmann-based rescaling
    nm0 *= self.boltz_rescale(nm0, nm, em)
    # Compute source term
    w = tf.einsum("nij,nj->ni", k, nm0) + wr
    # Return linear update
    return nm0 + t * w

  # Boltzmann-based rescaling
  # -----------------------------------
  def boltz_rescale(self, ni0, nt, et):
    th = {k: self.sp_th[k][self.molecule].thermo for k in ("Ng", "1g")}
    # Compute total density and energy at t=0
    # -----------------------
    n0 = tf.math.reduce_sum(ni0, axis=-1, keepdims=True)
    if (self.boltz_rescale_order > 0):
      # Get state-specific energies
      ei = th["Ng"].int.e
      # Get total energy
      e0 = tf.math.reduce_sum(ni0 * ei, axis=-1, keepdims=True) / n0
      # Compute temperatures
      T0, Tt = [th["1g"].int._update_interp_inv(e) for e in (e0, et)]
    # Compute rescaling factor
    # -----------------------
    dB = nt/n0
    if (self.boltz_rescale_order > 0):
      dB = dB * tf.math.exp((self.mk * (1/T0-1/Tt)) * ei)
    return dB

  # Configuration update
  # ===================================
  def get_config(self):
    sp_th_cfg = tf.nest.map_structure(lambda st: st.get_config(), self.sp_th)
    sp_th = {k: {"config": stk_cfg} for (k, stk_cfg) in sp_th_cfg.items()}
    config = {
      "rates": tf.nest.map_structure(ops.to_numpy, self.rates),
      "specie_thermos": sp_th,
      "use_state_corr": self.use_state_corr,
      "boltz_rescale_order": self.boltz_rescale_order
    }
    base_config = super(FwdEulerME, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class MatExpME(nn_elem.Layer):
  """Matrix exponential-type update for the master equation."""

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
    use_state_corr=False,
    **kwargs
  ):
    super(MatExpME, self).__init__(**kwargs)
    # Operators layer
    self.compute_ops = MEOperators(
      rates=rates,
      specie_thermos=specie_thermos,
      use_corr=use_corr,
      use_state_corr=use_state_corr
    )

  # # Calling
  # # ===================================
  # def call(self, inputs, training=False):
  #   qt, nm0, f = utils.unpack_list(inputs, nb_elem=3)
  #   # Extract time-dependent QoI
  #   # t, na, T = tf.split(qt, num_or_size_splits=3, axis=1)
  #   t, na, _nm, T, _em = tf.split(qt, num_or_size_splits=5, axis=1)
  #   # Compute operators
  #   k, wr = self.compute_ops((na, T, f))
  #   # Construct solution operators
  #   # > f = exp(K)
  #   exp_k = tf.linalg.expm(k * tf.expand_dims(t, axis=-1))
  #   # > f = K^(-1)
  #   k_inv = tf.linalg.inv(k)
  #   # > f = exp(K) - I
  #   exp_k_I = tf.linalg.set_diag(
  #     input=exp_k,
  #     diagonal=tf.linalg.diag_part(exp_k) - 1.0
  #   )
  #   # Compute solution
  #   # > f = exp(K) @ nm0 + (exp(K) - I) @ K^(-1) @ wr
  #   nm = tf.einsum("nij,nj->ni", exp_k, nm0)
  #   nm += tf.einsum("nij,njk,nk->ni", exp_k_I, k_inv, wr)
  #   return nm

  # Calling
  # ===================================
  def call(self, inputs, training=False):
    qt, nm0, f = utils.unpack_list(inputs, nb_elem=3)
    # Extract time-dependent QoI
    # t, na, T = tf.split(qt, num_or_size_splits=3, axis=1)
    t, na, _nm, T, _em = tf.split(qt, num_or_size_splits=5, axis=1)
    # Compute operators
    k, wr = self.compute_ops((na, T, f))
    # Eigen-decomposition of the linear operator
    # > K = V @ S @ V^(-1)
    lam_k = self._decompose(k)
    # Construct solution operators
    # > f = exp(K)
    exp_k = self._reconstruct(tf.math.exp(t*lam_k))
    # > f = K^(-1)
    k_inv = self._reconstruct(1/lam_k)
    # > f = exp(K) - I
    exp_k_I = tf.linalg.set_diag(
      input=exp_k,
      diagonal=tf.linalg.diag_part(exp_k) - 1.0
    )
    # Compute solution
    # > f = exp(K) @ nm0 + (exp(K) - I) @ K^(-1) @ wr
    nm = tf.einsum("nij,nj->ni", exp_k, nm0)
    nm += tf.einsum("nij,njk,nk->ni", exp_k_I, k_inv, wr)
    return nm

  def _decompose(self, x):
    lam, v = tf.linalg.eig(x)
    v_inv = tf.linalg.inv(v)
    lam, self.v, self.v_inv = [tf.math.real(x) for x in (lam, v, v_inv)]
    return lam

  def _reconstruct(self, x):
    # > f = V @ diag(x) @ V^(-1)
    return tf.einsum("nij,nj,njk->nik", self.v, x, self.v_inv)
