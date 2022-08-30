import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend
from pycomet.physics.general import const
from pycomet.physics.thermochemistry.specie_thermo.partition_fun.basic import Basic


class Translational(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        specie,
        *args,
        **kwargs
    ):
        super(Translational, self).__init__(specie)
        # Constant base
        self.base = tf.cast(
            2.*np.pi*self.specie.m*const.UKB/(const.UH**2), backend.floatx()
        )

    # Methods
    ###########################################################################
    # Update ------------------------------------------------------------------
    def update(self, T):
        if not self._updated:
            # Extract translational temperature
            T = tf.gather(T, indices=[self.specie.Th_index], axis=1)
        super(Translational, self).update(T)

    # Partition functions -----------------------------------------------------
    def _q(self, T):
        return tf.math.pow(self.base*T, 3./2.), backend.floatx()

    def _dqdT(self, T):
        return 3./2.*self.base*tf.math.pow(self.base*T, 1./2.)

    def _Q(self):
        return self.q

    def _dQdT(self):
        return self.dqdT
