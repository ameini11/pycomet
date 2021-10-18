import tensorflow as tf

from tensorflow.python.keras import backend
from prode.physics.general import const
from prode.physics.thermochemistry.specie.partitionFun.basicPF import PartitionFun


class Internal(PartitionFun):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        *args,
        **kwargs
    ):
        super(Internal, self).__init__(specie)

    # Internal partition functions --------------------------------------------
    def z(self, T):
        return - tf.linalg.matmul(
            1./(T*const.UKB), self.specie.rv_lev['E'], transpose_b=True
        )

    def dz_dT(self, T):
        return - self.z(T) / T

    def q(self, T):
        return tf.transpose(self.specie.rv_lev['g']) * tf.math.exp(self.z(T))

    def dq_dT(self, T):
        return self.q(T) * self.dz_dT(T)

    def Q(self, T):
        if self.specie.n_at > 1:
            q_ = self.q(T)
            Q_ = []
            for bin_i in range(self.specie.n_bins):
                mask = self.specie.lev_to_bin == bin_i
                Q_.append(
                    tf.math.reduce_sum(
                        tf.boolean_mask(q_, mask, axis=1), axis=1, keepdims=True
                    )
                )
            return tf.concat(Q_, 1)
        else:
            return self.specie.g_e * tf.ones_like(T, dtype=backend.floatx())

    def dQ_dT(self, T):
        if self.specie.n_at > 1:
            dq_dT_ = self.dq_dT(T)
            dQ_dT_ = []
            for bin_i in range(self.specie.n_bins):
                mask = self.specie.lev_to_bin == bin_i
                dQ_dT_.append(
                    tf.math.reduce_sum(
                        tf.boolean_mask(dq_dT_, mask, axis=1), axis=1, keepdims=True
                    )
                )
            return tf.concat(dQ_dT_, 1)
        else:
            return tf.zeros_like(T, dtype=backend.floatx())
