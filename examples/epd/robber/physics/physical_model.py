import numpy as np
import pycomet as pcm
import tensorflow as tf


class PhysicalModel(pcm.physics.PhysicalModel):

    # Initialization
    ###########################################################################
    def __init__(
        self
    ):
        super(PhysicalModel, self).__init__()

    # ODEs Residual
    ###########################################################################
    def xde(self, x, y, grads):
        
        # ODE coefficients
        K, C = tf.split(x[0], [1,1], axis=1)

        # Compute gradients
        dydt = grads(y, x[2], j=[0,1], i=[0], order=1, key='dydt')

        # Compute residuals
        x, v = tf.split(y, [1,1], axis=1)
        dxdt, dvdt = tf.split(dydt, [1,1], axis=1)
        res = tf.concat([dxdt - v, dvdt - K*x - C*v], 1)

        return res
