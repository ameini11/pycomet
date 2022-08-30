import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend
from pycomet.physics.general import const
from pycomet.physics.thermochemistry.specie_thermo.thermo import Basic


class CoupledEnergyModes(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        int_pf,
        process='isochoric',
        *args,
        **kwargs
    ):
        super(CoupledEnergyModes, self).__init__(specie, process=process)
        self.int_pf = int_pf

    # Methods
    ###########################################################################
    # Internal energy
    def _e_int(self, T):
        # Get partition functions
        q, Q = self.int_pf.q, self.int_pf.Q
        # Get energy of the levels
        E = tf.transpose(self.specie.rv_lev['E'])
        # Evaluate internal energy [J]
        e_int = []
        for c in range(self.specie.n_comp):
            lev_mask = self.specie.mapping['lev'] == c
            e_int.append(
                tf.math.reduce_sum(
                    tf.boolean_mask(E * self.int_pf.q, lev_mask, axis=1),
                    axis=1,
                    keepdims=True
                )
            )
        e_int = tf.concat(e_int, 1) / self.int_pf.Q
        # Return the specific internal energy per unit mass [J/kg]
        return e_int / self.specie.m

    def _cv_int(self, T):
        # Get internal energy value [J]
        e_int = self.e_int * self.specie.m
        # Get energy of the levels
        E = tf.transpose(self.specie.rv_lev['E'])
        # Evaluate internal heat capacities [J/K]
        cv_int = []
        for c in range(self.specie.n_comp):
            lev_mask = self.specie.mapping['lev'] == c
            cv_int.append(
                tf.math.reduce_sum(
                    tf.boolean_mask(E * self.int_pf.dqdT, mask, axis=1),
                    axis=1,
                    keepdims=True
                )
            )
        cv_int = tf.concat(cv_int, 1)
        cv_int = (cv_int - e_int * self.int_pf.dQdT) / self.int_pf.Q
        # Return the specific internal heat capacities per unit mass [J/kg]
        return cv_int / self.specie.m
