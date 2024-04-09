import numpy as np
import tensorflow as tf

from pycomet import backend as bkd
from pycomet.config import get_config
from .connectivity import LastToCurrent
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
    self.prob_mat = None
    self.indices_compress = indices_compress
    if (self.indices_compress is not None):
      self.compress = LastToCurrent(indices=self.indices_compress)
      self.prob_mat = self.construct_prob_mat(indices_compress.squeeze())
    # Splitting layer
    self.indices_split = indices_split
    if (self.indices_split is not None):
      self.indices_compress = [
        tf.reshape(tf.constant(indices, dtype="int32"), shape=[-1]) \
          for indices in self.indices_split
      ]

  def construct_prob_mat(self, mapping):
    # Define dimensions and indices
    mapping = (mapping - np.amin(mapping)).astype(int)
    nb_comp_in, nb_comp_out = len(mapping), np.amax(mapping)+1
    indices = (mapping, np.arange(nb_comp_in))
    # Sparse tensor
    prob_mat = tf.sparse.SparseTensor(
      indices=np.vstack(indices).T,
      values=tf.ones(nb_comp_in, dtype=bkd.get_floatx()),
      dense_shape=(nb_comp_out, nb_comp_in)
    )
    return prob_mat

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
      nm_0 = self.compress(nm_0)
      k, w = self.compress_ops(k, w, qm)
    # Solution
    # -------------
    if (self.indices_split is not None):
      xm_t = []
      for indices in self.indices_split:
        # Slicing
        si, ei = tf.math.reduce_min(indices), tf.math.reduce_max(indices)
        xm_t.append(self.update(
          t, nm_0[:,si:ei], k[:,si:ei,si:ei], w[:,si:ei], training
        ))
        # Concatenate final solution
        return tf.concat(xm_t, axis=-1) if (len(xm_t) > 1) else xm_t[-1]
    else:
      return self.update(t, nm_0, k, w, training)

  # Compress operators
  # -----------------------------------
  def compress_ops(self, k, w, q):
    # Compress linear operator
    # > First dimension
    kc = self._spmm_3d(self.prob_mat, k)
    # > Second dimension
    kc *= tf.expand_dims(q, axis=1)
    kc = tf.transpose(
      self._spmm_3d(
        self.prob_mat,
        tf.transpose(kc, perm=[0,2,1])
      ),
      perm=[0,2,1]
    )
    # > Normalize
    qc = self._spmm_2d(self.prob_mat, q)
    kc *= tf.expand_dims(1/qc, axis=1)
    # Compress linear source
    wc = self._spmm_2d(self.prob_mat, w)
    return kc, wc

  def _spmm_2d(self, a, b):
    b = tf.transpose(b)
    y = tf.sparse.sparse_dense_matmul(a, b)
    return tf.transpose(y)

  def _spmm_3d(self, a, b):
    b = tf.transpose(b, perm=[1,2,0])
    ash, bsh = tf.shape(a), tf.shape(b)
    b = tf.reshape(b, (bsh[0], -1))
    y = tf.sparse.sparse_dense_matmul(a, b)
    y = tf.reshape(y, (ash[0], bsh[1], bsh[2]))
    return tf.transpose(y, perm=[2,0,1])

  # Update solution
  # -----------------------------------
  def update(self, t, nm_0, k, w, training):
    # Compute updated solution
    nm_t = self.linode((t, nm_0, k, w), training=training)
    # Compute relative molar fractions
    return nm_t * (1/tf.math.reduce_sum(nm_t, axis=-1, keepdims=True))

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "use_eig": self.linode.use_eig,
      "use_pinv": self.linode.use_pinv,
      "expm_use_pade": self.linode.expm_use_pade,
      "indices_split": self.compress.indices,
      "indices_compress": self.indices_compress
    }
    base_config = super(SolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
