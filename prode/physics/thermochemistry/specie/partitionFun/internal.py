import tensorflow as tf

from tensorflow.python.keras import backend
from prode.physics.general import const
from prode.physics.thermochemistry.specie.partitionFun.basic import Basic


class Internal(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        *args,
        **kwargs
    ):
        super(Internal, self).__init__(specie)

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, T):
        self.z = self.z_(T)
        self.dzdT = self.dzdT_(T)
        self.q = self.q_(T)
        self.dqdT = self.dqdT_(T)
        super(Internal, self).update(T)

    # Internal partition functions --------------------------------------------
    def z_(self, T):
        if self.specie.n_at > 1:
            return - tf.linalg.matmul(
                1./(T*const.UKB), self.specie.rv_lev['E'], transpose_b=True
            )
        else:
            return tf.ones_like(T, dtype=backend.floatx())

    def dzdT_(self, T):
        if self.specie.n_at > 1:
            return - self.z / T
        else:
            return tf.zeros_like(T, dtype=backend.floatx())

    def q_(self, T):
        if self.specie.n_at > 1:
            return tf.transpose(self.specie.rv_lev['g']) * tf.math.exp(self.z)
        else:
            return tf.ones_like(T, dtype=backend.floatx())

    def dqdT_(self, T):
        if self.specie.n_at > 1:
            return self.q * self.dzdT
        else:
            return tf.zeros_like(T, dtype=backend.floatx())

    def Q_(self, T):
        if self.specie.n_at > 1:
            _Q = []
            for bin_i in range(self.specie.n_bins):
                mask = self.specie.lev_to_bin == bin_i
                _Q.append(
                    tf.math.reduce_sum(
                        tf.boolean_mask(self.q, mask, axis=1), axis=1, keepdims=True
                    )
                )
            return tf.concat(_Q, 1)
        else:
            return self.specie.g_e * tf.ones_like(T, dtype=backend.floatx())

    def dQdT_(self, T):
        if self.specie.n_at > 1:
            _dQdT = []
            for bin_i in range(self.specie.n_bins):
                mask = self.specie.lev_to_bin == bin_i
                _dQdT.append(
                    tf.math.reduce_sum(
                        tf.boolean_mask(self.dqdT, mask, axis=1), axis=1, keepdims=True
                    )
                )
            return tf.concat(_dQdT, 1)
        else:
            return tf.zeros_like(T, dtype=backend.floatx())
