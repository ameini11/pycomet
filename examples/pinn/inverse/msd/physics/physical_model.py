import numpy as np
import pycomet as pcm
import tensorflow as tf

from tensorflow.python.keras import backend


class PhysicalModel(pcm.physics.PhysicalModel):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        params
    ):
        super(PhysicalModel, self).__init__()
        # Extra trainable variables
        for k in ('C', 'K'):
            setattr(
                self, k, tf.Variable(
                    initial_value=0.,
                    trainable=True,
                    name=k,
                    dtype=backend.floatx()
                )
            )
        self.extra_trainable_variables = [self.C, self.K]

    # ODEs Residual
    ###########################################################################
    def xde(self, x, y, grads):
        # Compute gradients
        dydt = grads(y, x, j=[0,1], i=[0], order=1, key='dydt')

        # Compute rhs
        pos, vel = tf.split(y, 2, axis=1)
        rhs = tf.concat([vel, - self.K*pos - self.C*vel], axis=1)

        # Return residual
        return dydt - rhs