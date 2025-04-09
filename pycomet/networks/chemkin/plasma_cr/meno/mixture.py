import json
import numpy as np
import tensorflow as tf

from . import const
from pycomet import backend as bkd


class Mixture(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species
  ):
    self.species_order = ("em", "Ar", "Arp")
    self._init_species(species)
    self._build_mass_vec()

  def _init_species(self, species):
    # Initialize species
    self.species = {}
    for k in species.keys():
      if isinstance(species[k], str):
        species[k] = {"properties": species[k]}
      self.species[k] = Species(**species[k])
      self.species[k].build()
    # Index species
    si = 0
    for k in self.species_order:
      ei = si + self.species[k].nb_comp
      self.species[k].indices = tf.constant(np.arange(si, ei), dtype="int32")
      si = ei
    self.nb_comp = ei
    # Species names
    self.species_names = []
    for k in self.species_order:
      self.species_names.extend(self.species[k].comp_names)

  def _build_mass_vec(self):
    # Build vector of masses
    self.m = np.ones((1,self.nb_comp))
    for k in self.species_order:
      s = self.species[k]
      self.m[0,s.indices] *= s.m
    self.m = tf.constant(self.m, dtype=bkd.get_floatx())
    self.m_inv = 1.0/self.m

  # Update
  # ===================================
  def update_species_thermo(self, Th, Te):
    self.q = []
    for k in self.species_order:
      sp = self.species[k]
      Ti = Te if (sp.name == "em") else Th
      sp.update(Ti)
      self.q.append(sp.q)
    self.q = tf.concat(self.q, axis=-1)

  # Conversions
  # ===================================
  def get_n(self, rho):
    return rho * self.m_inv

  def get_rho(self, n):
    return n * self.m

  def get_Te(self, pe, ne):
    return pe / (ne * const.UKB)

  def get_pe(self, Te, ne):
    return ne * const.UKB * Te

  # Configuration
  # ===================================
  def get_config(self):
    config = {"species": {}}
    for (k, sp) in self.species.items():
      config["species"][k] = {
        "properties": None,
        "config": sp.get_config(),
      }
    return config


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
