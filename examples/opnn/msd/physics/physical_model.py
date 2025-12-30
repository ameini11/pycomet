import numpy as np
import pycomet as pcm
import tensorflow as tf


class PhysicalModel(pcm.physics.PhysicalModel):

  # Initialization
  # ===================================
  def __init__(
    self,
    params
  ):
    super(PhysicalModel, self).__init__()
    # ODE coefficients matrix
    K = params['k']/params['m']
    C = params['c']/params['m']
    self.mat = np.array([[0, 1],[-K, -C]])

  # ODEs Residual
  # ===================================
  def xde(self, x, y, grads):
    # Compute gradients
    dydt = grads(y, x[1], j=[0,1], i=[0], order=1, key='dydt')
    # Compute residuals
    res = dydt - tf.matmul(y, self.mat, transpose_b=True)
    return res
