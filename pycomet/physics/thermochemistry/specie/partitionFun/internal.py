import tensorflow as tf

from tensorflow.python.keras import backend
from pycomet.physics.general import const
from pycomet.physics.thermochemistry.specie.partitionFun.basic import Basic


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

    # Levels distribution -----------------------------------------------------
    def levels_distribution(self, T=0.):
        # [J/mol]
        n, g, E = [], [], []
        if self.specie.n_at > 1:
            g_sp = self.specie.rv_lev['g']
            E_sp = self.specie.rv_lev['E'] / const.EV_to_J
            n_over_Q = self.specie.n / self.Q
            for bin_i in range(self.specie.n_bins):
                mask = self.specie.lev_to_bin == bin_i
                n_over_Q_i = tf.gather(n_over_Q, [bin_i], axis=1)
                n.append(n_over_Q_i * tf.boolean_mask(self.q, mask, axis=1))
                g.append(tf.boolean_mask(g_sp, mask))
                E.append(tf.boolean_mask(E_sp, mask))
        return n, g, E