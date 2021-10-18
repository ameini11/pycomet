import numpy as np
import tensorflow as tf

from prode.physics.general import const
from prode.physics.thermochemistry.specie.partitionFun.internalPF import Internal


class CoupledEnergyModes(object):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        *args,
        **kwargs
    ):
        # Specie Properties ===================================================
        self.specie = specie
        self.intPF = Internal(specie)

    # Methods
    ###########################################################################
    # Enthalpy ================================================================
    def cp(self, T):
        # [J/(mol K)]
        return self.cv(T) + const.URG

    def h(self, T):
        # [J/mol]
        return self.e(T) + const.URG*T

    # Energy ==================================================================
    def cv(self, T):
        # [J/(mol K)]
        return self.cv_tr() + self.cv_int(T)

    def e(self, T):
        # [J/mol]
        return self.e_tr(T) + self.e_int(T) + self.e_f()

    # Energy of formation -----------------------------------------------------
    def e_f(self):
        # [J/mol]
        return self.specie.Ef

    # Translational Energy ----------------------------------------------------
    def cv_tr(self):
        # [J/(mol K)]
        return 3./2.*const.URG

    def e_tr(self, T):
        # [J/mol]
        return 3./2.*const.URG*T

    # Ro-vibrational Energy ---------------------------------------------------
    def cv_int(self, T):
        # [J/(mol K)]
        if self.specie.n_at > 1:
            q, dq_dT = self.intPF.q(T), self.intPF.dq_dT(T)
            Q = tf.split(
                self.intPF.Q(T), num_or_size_splits=self.specie.n_bins, axis=1
            )
            dQ_dT = tf.split(
                self.intPF.dQ_dT(T), num_or_size_splits=self.specie.n_bins, axis=1
            )
            cv_int_ = []
            for bin_i in range(self.specie.n_bins):
                mask = self.specie.lev_to_bin == bin_i
                f1 = self.specie.rv_lev['E'] / Q[bin_i]
                f2 = dq_dT - q * dQ_dT[bin_i] / Q[bin_i]
                cv_int_.append(
                    tf.math.reduce_sum(
                        tf.boolean_mask(f1*f2, mask, axis=1), axis=1, keepdims=True
                    )
                )
            return tf.concat(cv_int_, 1) * const.UNA
        else:
            return tf.zeros_like(T, dtype=backend.floatx())

    def e_int(self, T):
        # [J/mol]
        if self.specie.n_at > 1:
            q, Q = self.intPF.q(T), self.intPF.Q(T)
            E = tf.transpose(self.specie.rv_lev['E'])
            e_int_ = []
            for bin_i in range(self.specie.n_bins):
                mask = self.specie.lev_to_bin == bin_i
                e_int_.append(
                    tf.math.reduce_sum(
                        tf.boolean_mask(E*q, mask, axis=1), axis=1, keepdims=True
                    )
                )
            return tf.concat(e_int_, 1) / Q * const.UNA
        else:
            return tf.zeros_like(T, dtype=backend.floatx())
