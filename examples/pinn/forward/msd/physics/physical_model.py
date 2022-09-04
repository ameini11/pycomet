import numpy as np
import pycomet as pcm
import tensorflow as tf


class PhysicalModel(pcm.physics.PhysicalModel):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        params
    ):
        super(PhysicalModel, self).__init__()
        
        # ODE matrices
        self.K = self.get_matrices(params)

    # ODEs Residual
    ###########################################################################
    def xde(self, x, y, grads):
        # Compute gradients
        dydt = grads(y, x, j=[0,1], i=[0], order=1, key='dydt')

        # Compute residuals
        res = dydt - tf.matmul(y, self.K, transpose_b=True)

        return res

    # Build Matrices
    ###########################################################################
    def get_matrices(self, params):
        K = params['k']/params['m']
        C = params['c']/params['m']
        return np.array([[0, 1.],[-K, -C]])
