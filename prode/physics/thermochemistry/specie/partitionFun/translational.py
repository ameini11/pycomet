import numpy as np
import tensorflow as tf

from tensorflow.python.keras import backend
from prode.physics.general import const
from prode.physics.thermochemistry.specie.partitionFun.basic import Basic


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
        self.basis = 2.*np.pi*const.UKB*self.specie.m/const.UH**2

    # Translational partition functions ---------------------------------------
    def Q_(self, T):
        return tf.cast(
            tf.math.pow(self.basis*T, 3./2.), backend.floatx()
        )

    def dQdT_(self, T):
        return  tf.cast(
            3./2.*self.basis*tf.math.pow(self.basis*T, 1./2.), backend.floatx()
        )
