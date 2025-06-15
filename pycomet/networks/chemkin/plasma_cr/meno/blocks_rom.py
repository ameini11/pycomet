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
    cfg_ops={},
    cfg_rom={},
    cfg_ode={},
    **kwargs
  ):
    super(PlasmaCRSolUpdateROM, self).__init__(**kwargs)
    cfg_ops["use_mass_frac"] = True
    self.lays = {
      # PlasmaCROperators layer
      "ops": PlasmaCROperators(**cfg_ops),
      # LinPROM Layer
      "rom": meno_elem.blocks.LinPROM(**cfg_rom),
      # LinODE Layer
      "ode": meno_elem.layers.LinODE(**cfg_ode)
    }
    # Mixture
    self.mix = self.lays["ops"].mix["h"]

  # Building
  # ===================================
  def build(self, *args, **kwargs):
    for lay in self.lays.values():
      if (not lay.built):
        lay.build(*args, **kwargs)
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
    self.idec = tf.linalg.matvec(imask, self.lays["rom"].decoder)
    self.iref = tf.tensordot(imask, self.lays["rom"].xref, axes=1)

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    t, zh_0, u, f = inputs
    # Compute operators
    A = self.lays["ops"]((*u, f), training=training)
    # Evolve latent space
    # > Heavy-particle only
    Ar, br = self.lays["rom"]((A,), method=4, training=training)
    zh = self.lays["ode"]((t, zh_0, Ar, br), training=training)
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
      "cfg_ops": self.lays["ops"].get_config(),
      "cfg_ode": self.lays["ode"].get_config(),
      "cfg_rom": self.lays["rom"].get_config()
    }
    base_config = super(PlasmaCRSolUpdateROM, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
