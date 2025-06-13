import numpy as np
import tensorflow as tf

from pycomet import backend as bkd
from .blocks import PlasmaCROperators
from pycomet.config import get_config
from pycomet.networks import elements as nn_elem
from pycomet.networks import meno as meno_elem
_CFG = get_config()


@tf.keras.utils.register_keras_serializable(package=_CFG.name)
class PlasmaCRROMSolUpdate(nn_elem.Layer):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    rates,
    phi,
    psi,
    xref,
    xscale,
    use_corr=False,
    scale_corr=False,
    epsilon=1e-15,
    use_pade=False,
    use_expmv=False,
    max_degree=None,
    max_nb_squarings=None,
    expm_dtype=None,
    add_inv_loss=False,
    inv_loss_weight=1e-3,
    **kwargs
  ):
    super(PlasmaCRROMSolUpdate, self).__init__(**kwargs)
    # PlasmaCROperators layer
    self.operators = PlasmaCROperators(
      species=species,
      rates=rates,
      use_corr=use_corr,
      scale_corr=scale_corr
    )
    # LinODE Layer
    self.linode = meno_elem.layers.LinODE(
      nb_eqs=self.operators.nb_comp,
      epsilon=epsilon,
      use_pade=use_pade,
      use_expmv=use_expmv,
      max_degree=max_degree,
      max_nb_squarings=max_nb_squarings,
      expm_dtype=expm_dtype,
      add_inv_loss=add_inv_loss,
      inv_loss_weight=inv_loss_weight
    )
    self.linprom = meno_elem.blocks.LinPROM(
      phi=phi,
      psi=psi,
      xref=xref,
      xscale=xscale
    )
    # Mixture
    self.mix = self.operators.mix["h"]
    # Mixture
    self.build_ions_dec()

  def build_ions_dec(self):
    # Ions indices
    iions = self.mix.species["Arp"].indices - 1
    # Mask vector
    imask = np.zeros(len(self.mix.indices["h"]))
    imask[iions] = 1.0
    imask = tf.constant(imask, dtype=bkd.get_floatx())
    # > Scaling
    imask *= self.mix.species["em"].m * self.mix.mh_inv
    # Decoder and reference values
    self.idec = tf.linalg.matvec(self.linprom.decoder, imask)
    self.iref = tf.tensordot(imask, self.linprom.xref, axes=1)

  # Calling
  # ===================================
  def call(self, inputs, training=None):
    # Extract inputs
    t, zh_0, u, f = inputs
    # Compute operators
    A = self.operators((*u, f), training=training)
    # > From number densities to mass fractions
    A = self.convert_a(A)
    # Evolve latent space
    # > Heavy-particle only
    Ar, br = self.linprom((A,), method=4)
    zh = self.linode((t, zh_0, Ar, br), training=training)
    # > Electron (charge conservation)
    z = self.enforce_charge_cons(zh)
    return z

  def convert_a(self, A):
    # Equation: Aw = M @ An @ M_inv
    A = tf.transpose(A*self.mix.mh_inv, perm=(0,2,1))
    A = tf.transpose(A*self.mix.mh, perm=(0,2,1))
    return A

  def enforce_charge_cons(self, zh):
    # Compute electron mass fraction
    # > we = me * imask^T @ M_inv @ (D @ z + wref)
    we = tf.linalg.matvec(zh, self.idec) + self.iref
    return tf.concat([zh, we], axis=-1)

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      **self.operators.get_config(),
      **self.linode.get_config()
    }
    base_config = super(PlasmaCRROMSolUpdate, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))
