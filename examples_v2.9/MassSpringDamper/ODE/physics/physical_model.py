import numpy as np
import pycomet as pcm
import tensorflow as tf


class PhysicalModel(pcm.PhysicalModel):

    # Initialization
    ###########################################################################
    def __init__(
        self,
        params
    ):
        # ODE matrices --------------------------------------------------------
        self.K = self.get_matrices(params)

    # ODEs Residual
    ###########################################################################
    def xde(self, x, y, grads):
        """Evaluates the residual error of a ODEs system given the Initial
        Conditions, the time vector and the n-order derivatives of the model 
        outputs w.r.t. the time input.

        >> Study case herein: Mass-Spring-Damper

        Args:
            ic: The Initial Conditions (IC) input `Tensor` with shape 
                (batch_size, d), where `d` is the number of the IC variables.
            t: Time input `Tensor` with shape (batch_size, 1).
            grads: List of `Tensors` (or list of list of `Tensors`) with shape 
                (batch_size, var) of the derivatives of the model outputs
                (`var`) w.r.t. the time input `t`. From 0-th to n-order
                derivatives:
                    d^i(y)/dt^i with i = 0,...,n

        Returns:
            A `Tensor` (or list of `Tensors`) with shape (batch_size, var),
            with `var` the number of variables of the ODEs system.
        """
        # Extract gradients
        dy_dt = grads(y, x[1], i=[0,1], j=[0], order=1)

        # Compute residuals
        res = dy_dt - tf.matmul(y, self.K, transpose_b=True)

        return res

    # Build Matrices
    ###########################################################################
    def get_matrices(self, params):
        a1 = -params['k']/params['m']
        a2 = -params['c']/params['m']
        return np.array([[0, 1.],[a1, a2]], dtype=np.float64)
