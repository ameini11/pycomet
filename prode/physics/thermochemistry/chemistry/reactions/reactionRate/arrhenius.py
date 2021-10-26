import numpy as np
import tensorflow as tf

from prode.physics.thermochemistry.chemistry.reactions.reactionRate import Basic


class Arrhenius(Basic):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        reactionsDatabase,
        *args,
        **kwargs
    ):
        super(Arrhenius, self).__init__(
            reactionsDatabase,
            *args,
            **kwargs
        )
        self.A = np.expand_dims(self.reacDB['A'].to_numpy(), 0)
        self.beta = np.expand_dims(self.reacDB['beta'].to_numpy(), 0)
        self.Ta = np.expand_dims(self.reacDB['Ta'].to_numpy(), 0)

    # Methods
    ###########################################################################
    # Forward reaction rates --------------------------------------------------
    def k_(self, T):
        k = self.A
        k = k * tf.pow(T, self.beta)
        k = k * tf.math.exp(-tf.linalg.matmul(1./T, self.Ta))
        return k