import tensorflow as tf

from pycomet import operations as ops
from pycomet.config import get_config
from .connectivity import Compressing
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
class SolUpdate(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    use_eig=False,
    use_pinv=False,
    expm_use_pade=False,
    indices_split=None,
    indices_compress=None,
    **kwargs
  ):
    super(SolUpdate, self).__init__(**kwargs)
    # Linear ODE layer
    self.linode = nn_elem.LinODE(
      use_eig=use_eig,
      use_pinv=use_pinv,
      real_only=True,
      expm_use_pade=expm_use_pade,
      expm_max_degree=None,
      expm_max_nb_squarings=None
    )
    # Compressing layer
    self.indices_compress = indices_compress
    if (self.indices_compress is not None):
      self.compress = Compressing(mapping=self.indices_compress)
    # Splitting layer
    self.indices_split = indices_split
    if (self.indices_split is not None):
      self.indices_split = [
        tf.reshape(tf.constant(indices, dtype="int32"), shape=[-1]) \
          for indices in self.indices_split
      ]

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    # -------------
    # > t:    time
    # > nm_0: molecule's initial distribution function
    # > k:    linear matrix operator
    # > w:    linear source operator
    # > qm:   updated molecule's partition functions (temperature T)
    t, nm_0, k, w, qm = inputs
    # Compress
    # -------------
    if (self.indices_compress is not None):
      nm_0, k, w = self.compress_vars(nm_0, k, w, qm)
    # Split and update solution
    # -------------
    if (self.indices_split is not None):
      xm_t = []
      for indices in self.indices_split:
        # Slice
        si, ei = tf.math.reduce_min(indices), tf.math.reduce_max(indices)
        xm_t.append(
          self.update_sol(
            t, nm_0[:,si:ei], k[:,si:ei,si:ei], w[:,si:ei]
          )
        )
        # Concatenate final solution
        return tf.concat(xm_t, axis=-1) if (len(xm_t) > 1) else xm_t[-1]
    else:
      return self.update_sol(t, nm_0, k, w)

  # Compress variables
  # -----------------------------------
  def compress_vars(self, nm_0, k, w, qm):
    nm_0, w = [self.compress(z) for z in (nm_0, w)]
    # Compress linear operator
    # > Compress first dimension + Scaling
    k = self.compress(k, axis=1) * tf.expand_dims(qm, axis=1)
    # > Compress second dimension + Normalizing
    k = self.compress(k, axis=2) * tf.expand_dims(1/self.compress(qm), axis=1)
    return nm_0, k, w

  # Update solution
  # -----------------------------------
  def update_sol(self, t, nm_0, k, w):
    # Compute updated solution
    nm_t = self.linode((t, nm_0, k, w))
    # Compute relative molar fractions
    return nm_t * (1/tf.math.reduce_sum(nm_t, axis=-1, keepdims=True))

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "use_eig": self.linode.use_eig,
      "use_pinv": self.linode.use_pinv,
      "expm_use_pade": self.linode.expm_use_pade,
      "indices_split": tf.nest.map_structure(ops.to_numpy, self.indices_split),
      "indices_compress": self.compress.mapping
    }
    base_config = super(SolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
