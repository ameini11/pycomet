import numpy as np
import tensorflow as tf

from . import const
from .species import Species
from pycomet import backend as bkd


class Mixture(object):

  # Initialization
  # ===================================
  def __init__(
    self,
    species,
    species_order
  ):
    self.species_order = tuple(species_order)
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
    config = {
      "species": {},
      "species_order": self.species_order
    }
    for (k, sp) in self.species.items():
      config["species"][k] = {
        "properties": None,
        "config": sp.get_config(),
      }
    return config
