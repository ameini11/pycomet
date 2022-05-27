import tensorflow as tf

from tensorflow.python.keras import backend
from pycomet.physics.general import const
from pycomet.physics.thermochemistry.specie_thermo.partition_fun.basic import Basic


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
        if not self._updated:
            self.z = self._z(T)
            self.dzdT = self._dzdT(T)
            self.q = self._q(T)
            self.dqdT = self._dqdT(T)
            if not self.specie.n_comp == self.specie.lev['num']:
                self.Q = self._Q()
                self.dQdT = self._dQdT()
            else:
                self.Q = self.q
                self.dQdT = self.dqdT
            self._updated = True

    # Internal partition functions --------------------------------------------
    def _z(self, T):
        z = []
        for c in range(self.specie.n_comp):
            lev_mask = self.specie.mapping['lev'] == c
            temp_mask = self.specie.mapping['temp'][c]
            a = - 1./(tf.boolean_mask(T, temp_mask, axis=1)*const.UKB)
            b = tf.boolean_mask(self.specie.lev['E'], mask, axis=0)
            z.append(tf.linalg.matmul(a, b, transpose_b=True))
        return tf.concat(z, axis=1)

    def _dzdT(self, T):
        dzdT = []
        for c in range(self.specie.n_comp):
            lev_mask = self.specie.mapping['lev'] == c
            temp_mask = self.specie.mapping['temp'][c]
            a = - tf.boolean_mask(self.z, lev_mask, axis=1)
            b = tf.boolean_mask(T, temp_mask, axis=1)
            dzdT.append(a / b)
        return tf.concat(dzdT, axis=1)

    def _q(self, T):
        return tf.transpose(self.specie.lev['g']) * tf.math.exp(self.z)

    def _dqdT(self, T):
        return self.q * self.dzdT

    def _Q(self):
        Q = []
        for c in range(self.specie.n_comp):
            lev_mask = self.specie.mapping['lev'] == c
            Q.append(
                tf.math.reduce_sum(
                    tf.boolean_mask(self.q, lev_mask, axis=1),
                    axis=1,
                    keepdims=True
                )
            )
        return tf.concat(Q, axis=1)

    def _dQdT(self):
        dQdT = []
        for c in range(self.specie.n_comp):
            lev_mask = self.specie.mapping['lev'] == c
            dQdT.append(
                tf.math.reduce_sum(
                    tf.boolean_mask(self.dqdT, lev_mask, axis=1),
                    axis=1,
                    keepdims=True
                )
            )
        return tf.concat(dQdT, axis=1)

    # Levels distribution -----------------------------------------------------
    def levels_distribution(self, T=None):
        # Update partition functions
        if T is not None:
            self._updated = False
            self.update(T)
        # Initialize levels number density, degeneracy and energy
        n, g, E = [], [], []
        if self.specie.n_at > 1:
            g = self.specie.rv_lev['g']
            E = self.specie.rv_lev['E'] / const.EV_to_J
            n_ov_Q = self.specie.n / self.Q
            for bin_i in range(self.specie.n_bins):
                mask = self.specie.lev_to_bin == bin_i
                n_ov_Q_i = tf.gather(n_over_Q, [bin_i], axis=1)
                n.append(n_ov_Q_i * tf.boolean_mask(self.q, mask, axis=1))
                g.append(tf.boolean_mask(g_sp, mask))
                E.append(tf.boolean_mask(E_sp, mask))
        return n, g, E