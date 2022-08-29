import numpy as np
import tensorflow as tf

from pycomet.physics.thermochemistry.chemistry.reactions.reaction_rate import Basic


class Arrhenius(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        reactions_dtb,
        *args,
        **kwargs
    ):
        super(Arrhenius, self).__init__(
            reactions_dtb,
            *args,
            **kwargs
        )
        self.A = self.dtb['A'].to_numpy().reshape(1,-1)
        self.beta = self.dtb['beta'].to_numpy().reshape(1,-1)
        self.Ta = self.dtb['Ta'].to_numpy().reshape(1,-1)

    # Methods
    ###########################################################################
    # Forward reaction rates --------------------------------------------------
    def k_(self, T):
        k = self.A
        k = k * tf.pow(T, self.beta)
        k = k * tf.math.exp(-tf.linalg.matmul(1./T, self.Ta))
        return k
