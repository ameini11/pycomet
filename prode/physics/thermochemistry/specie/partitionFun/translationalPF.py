import numpy as np
import tensorflow as tf

from prode.physics.general import const
from prode.physics.thermochemistry.specie.partitionFun.basicPF import PartitionFun


class Translational(PartitionFun):

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
    def Q(self, T):
        return tf.math.pow(self.basis*T, 3./2.)

    def dQ_dT(self, T):
        return 3./2.*self.basis*tf.math.pow(self.basis*T, 1./2.)
