import numpy as np

from .functions import *
from hypernet import utils
from hypernet import backend as tf


class PolyHarmInterpolant(utils.DefObject):
  """Polyharmonic interpolant.
  See https://en.wikipedia.org/wiki/Polyharmonic_spline.
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    c,
    f,
    order=3,
    smoothing=0.0,
    xform=None,
    *args,
    **kwargs
  ):
    super(PolyHarmInterpolant, self).__init__(*args, **kwargs)
    # Set data
    self.c, self.f = c, f
    # Get tensor ranks
    self.ranks = {"c": tf.rank(self.c), "f": tf.rank(self.f)}
    # Set order and smoothing
    self.order = int(np.clip(order, 1.0, 3.0))
    self.smoothing = float(smoothing)
    # Set transformation
    if (xform in (None, "lin")):
      self.xform = None
    elif (xform == "log"):
      self.xform = (tf.log, tf.exp)
    else:
      utils.raise_err(
        ValueError,
        "Input 'xform' not recognized. Valid options: [None, 'lin', 'log']"
      )
    self.phi = get_phi(self.order)
    # Set default rank of input/output tensors
    self.rank = 2
    # Fit the interpolant to the observed data
    self._build()

  # Class methods
  # ===================================
  # Calling
  # -----------------------------------
  def __call__(self, x):
    # Reshape
    x = to_rank(x, self.rank)
    # Apply interpolation
    y = self._apply(x)
    # Transform
    if (self.xform is not None):
      y = self.xform[1](y)
    # Reshape
    y = to_rank(y, self.ranks["f"])
    return y

  def _apply(
    self,
    x
  ):
    # Compute the contribution from the rbf term
    d_phi = self.phi(cross_squared_distance_matrix(x, self.c))
    rbf_term = tf.matmul(d_phi, self.w)
    # Compute the contribution from the linear term
    ones = tf.ones_like(x[...,:1])
    x_pad = tf.concat([x, ones], axis=1)
    linear_term = tf.matmul(x_pad, self.v)
    return rbf_term + linear_term

  # Building
  # -----------------------------------
  def _build(self):
    # Transform
    if (self.xform is not None):
      self.f = self.xform[0](self.f)
    # Reshape
    self.c = to_rank(self.c, self.rank)
    self.f = to_rank(self.f, self.rank)
    # Compute coefficients
    self.w, self.v = self._solve()

  def _solve(
    self
  ):
    # Get dimensions
    n, d = tf.shape(self.c)
    k = tf.shape(self.f)[-1]
    # Construct the linear system
    # > Matrix A
    amat = self.phi(pairwise_squared_distance_matrix(self.c))
    if (self.smoothing > 0):
      amat += self.smoothing * tf.eye((n,))
    # > Matrix B
    ones = tf.ones((n, 1))
    bmat = tf.concat([self.c, ones], axis=1)
    bmat_ncols = bmat.shape[1]
    # > Left hand side
    lhs_zeros = tf.zeros((bmat_ncols, bmat_ncols))
    block_right = tf.concat([bmat, lhs_zeros], axis=0)
    block_left = tf.concat([amat, tf.transpose(bmat)], axis=0)
    lhs = tf.concat([block_left, block_right], axis=1)
    # > Right hand side
    rhs_zeros = tf.zeros((d + 1, k))
    rhs = tf.concat([self.f, rhs_zeros], axis=0)
    # Solve the linear system
    w_v = tf.solve(lhs, rhs)
    w = w_v[:n,:]
    v = w_v[n:,:]
    return w, v
