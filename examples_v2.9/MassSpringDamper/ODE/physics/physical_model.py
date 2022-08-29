import numpy as np
import tensorflow as tf


class PhysicalModel(object):

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
    def xde(self, ic, t, grads):
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
        y, dy_dt = grads

        # Compute residuals
        res = dy_dt - tf.matmul(y, self.K, transpose_b=True)

        return res

    # Build Matrices
    ###########################################################################
    def get_matrices(self, params):

        m, k, c = params
        K = np.array([[0, 1.],[-k/m, -c/m]], dtype=np.float64)

        return K


