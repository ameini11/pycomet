import numpy as np
import tensorflow as tf

from .blocks import PlasmaCROperators
from pycomet import backend as bkd
from pycomet.config import get_config
from pycomet.networks import meno as meno_elem
from pycomet.networks import elements as nn_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PlasmaCRSolUpdateROM(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    kwargs_ops={},
    kwargs_rom={},
    kwargs_ode={},
    **kwargs
  ):
    super(PlasmaCRSolUpdateROM, self).__init__(**kwargs)
    # PlasmaCROperators layer
    kwargs_ops["use_mass_frac"] = True
    self.lay_ops = PlasmaCROperators(**kwargs_ops)
    # LinPROM Layer
    self.lay_rom = meno_elem.blocks.LinPROM(**kwargs_rom)
    # LinODE Layer
    kwargs_ode["nb_eqs"] = self.lay_rom.ztot_dim
    self.lay_ode = meno_elem.layers.LinODE(**kwargs_ode)
    # Mixture
    self.mix = self.lay_ops.mix["h"]
    # Mixture
    self._build_ions_dec()

  def _build_ions_dec(self):
    # Ions indices
    iions = self.mix.species["Arp"].indices - 1
    # Mask vector
    imask = np.zeros(len(self.mix.indices["h"]))
    imask[iions] = 1.0
    imask = tf.constant(imask, dtype=bkd.get_floatx())
    # > Scaling
    imask *= self.mix.species["em"].m * self.mix.mh_inv
    imask = tf.reshape(imask, (1,-1))
    # Decoder and reference values
    self.idec = tf.linalg.matvec(imask, self.lay_rom.decoder)
    self.iref = tf.tensordot(imask, self.lay_rom.xref, axes=1)

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    t, zh_0, u, f = inputs
    # Compute operators
    A = self.lay_ops((*u, f), training=training)
    # Evolve latent space
    # > Heavy-particle only
    Ar, br = self.lay_rom((A,), method=4, training=training)
    zh = self.lay_ode((t, zh_0, Ar, br), training=training)
    # > Electron (charge conservation)
    z = self._enforce_charge_cons(zh)
    return z

  def _enforce_charge_cons(self, zh):
    # Compute electron mass fraction
    # > we = me * imask^T @ M_inv @ (D @ z + wref)
    we = tf.linalg.matmul(zh, self.idec) + self.iref
    return tf.concat([zh, we], axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      "kwargs_ops": self.lay_ops.get_config(),
      "kwargs_ode": self.lay_ode.get_config(),
      "kwargs_rom": self.lay_rom.get_config()
    }
    base_config = super(PlasmaCRSolUpdateROM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
