import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend
from prode.physics.general import const
from prode.physics.thermochemistry.specie.thermo import Basic


class CoupledEnergyModes(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        internalPF,
        constVP='V',
        *args,
        **kwargs
    ):
        super(CoupledEnergyModes, self).__init__(specie, constVP=constVP)
        self.intPF = internalPF

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, T):
        self.e_tr = self.e_tr_(T)
        self.cv_tr = self.cv_tr_()
        self.e_int = self.e_int_(T)
        self.cv_int = self.cv_int_(T)
        super(CoupledEnergyModes, self).update(T)

    # Energy ------------------------------------------------------------------
    def e_(self, T):
        # [J/kg]
        return self.e_tr + self.e_int + self.e_f_()

    def cv_(self, T):
        # [J/(kg K)]
        return self.cv_tr + self.cv_int

    # Translational
    def e_tr_(self, T):
        # [J/kg]
        return 3./2. * self.specie.R * T

    def cv_tr_(self):
        # [J/(kg K)]
        return 3./2. * self.specie.R

    # Ro-vibrational
    def e_int_(self, T):
        # [J/kg]
        if self.specie.n_at > 1:
            # Get partition functions
            q, Q = self.intPF.q, self.intPF.Q
            # Get energy of the levels
            E = tf.transpose(self.specie.rv_lev['E'])
            # Evaluate internal energy [J]
            e_int = []
            for bin_i in range(self.specie.n_bins):
                mask = self.specie.lev_to_bin == bin_i
                e_int.append(
                    tf.math.reduce_sum(
                        tf.boolean_mask(E*q, mask, axis=1), axis=1, keepdims=True
                    )
                )
            e_int = tf.concat(e_int, 1) / Q
            # Return the specific internal energy per unit mass [J/kg]
            return e_int / self.specie.m
        else:
            return tf.zeros_like(T, dtype=backend.floatx())

    def cv_int_(self, T):
        # [J/(kg K)]
        if self.specie.n_at > 1:
            # Get internal energy value [J]
            e_int = self.e_int * self.specie.m
            # Get partition functions
            Q = self.intPF.Q
            dqdT, dQdT = self.intPF.dqdT, self.intPF.dQdT
            # Get energy of the levels
            E = tf.transpose(self.specie.rv_lev['E'])
            # Evaluate internal heat capacities [J/K]
            cv_int = np.zeros(self.specie.n_bins)
            for bin_i in range(self.specie.n_bins):
                mask = self.specie.lev_to_bin == bin_i
                cv_int.append(
                    tf.math.reduce_sum(
                        tf.boolean_mask(E*dqdT, mask, axis=1), axis=1, keepdims=True
                    )
                )
            cv_int = tf.concat(cv_int, 1)
            cv_int = (cv_int - e_int * dQdT) / Q
            # Return the specific internal heat capacities per unit mass [J/kg]
            return cv_int / self.specie.m
        else:
            return tf.zeros_like(T, dtype=backend.floatx())
