import json
import numpy as np
import tensorflow as tf

from . import const
from pycomet import backend as bkd


class Species(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    properties=None,
    config=None
  ):
    if (config is not None):
      self.config = config
      for (k, v) in self.config.items():
        setattr(self, k, v)
    else:
      # Load properties
      if (not isinstance(properties, dict)):
        with open(properties, "r") as file:
          properties = json.load(file)
      # Set properties
      for (k, v) in properties.items():
        setattr(self, k, v)
      # Levels
      self.lev = {k: np.array(v).reshape(1,-1) for (k, v) in self.lev.items()}
      self.nb_comp = self.lev["E"].size
      if (self.nb_comp > 1):
        self.comp_names = [self.name + f"_{i+1}" for i in range(self.nb_comp)]
      else:
        self.comp_names = [self.name]
      self.lev = tf.nest.map_structure(
        lambda z: tf.constant(z, dtype=bkd.get_floatx()), self.lev
      )
      # Configuration
      self.config = {}
      for (k, v) in properties.items():
        self.config[k] = v
      self.config["lev"] = self.lev
      self.config["nb_comp"] = self.nb_comp
      self.config["comp_names"] = self.comp_names
    # Indexing
    self.indices = []

  # Properties
  # ===================================
  @property
  def w(self):
    return self._w

  @w.setter
  def w(self, value):
    self._w = value

  @property
  def x(self):
    return self._x

  @x.setter
  def x(self, value):
    self._x = value

  @property
  def rho(self):
    return self._rho

  @rho.setter
  def rho(self, value):
    self._rho = value

  @property
  def n(self):
    return self._n

  @n.setter
  def n(self, value):
    self._n = value

  # Build and update
  # ===================================
  def build(self):
    # Thermo properties
    # > Charge number
    self.Z = int(self.Z)
    # > Mass [kg]
    self.m = self.M / const.UNA
    # > Specific gas constants [J/(kg K)]
    self.R = const.URG / self.M
    # Translational partition function factor
    self.q_tr_fac = 2.0 * np.pi * self.m * const.UKB / (const.UH * const.UH)

  def update(self, T):
    self.ov_kT = 1.0/(const.UKB*T)
    self.q = self._q(T)

  # Partition functions
  # ===================================
  def _q(self, T):
    return self._q_zero() * self._q_tr(T) * self._q_int()

  def _q_zero(self):
    return tf.math.exp(-self.Hf * self.ov_kT)

  def _q_tr(self, T):
    return tf.math.pow(self.q_tr_fac * T, 1.5)

  def _q_int(self):
    return self.lev["g"] * tf.math.exp(-self.lev["E"] * self.ov_kT)

  # Configuration
  # ===================================
  def get_config(self):
    return self.config
