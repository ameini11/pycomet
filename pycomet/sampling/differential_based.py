import numpy as np
import tensorflow as tf

from scipy.spatial import KDTree
from pycomet.sampling.sampler import Sampler


class DifferentialBased(Sampler):

  # Initialization
  # ===================================
  def __init__(
    self,
    method="source",
    grad_split=0.75,
    xlim=None,
    name=None,
    eps=1.e-5,
    *args,
    **kwargs
  ):
    super(DifferentialBased, self).__init__(
      name=name,
      *args,
      **kwargs
    )
    self.method = method
    self.grad_split = grad_split
    self.xlim = xlim
    self.eps = eps

  # Sampling
  # ===================================
  def sample(self, x, y, nb_pts):
    # Handling solution
    y = np.hstack(tf.nest.flatten(y))
    self.shape = y.shape
    # Rolling indices for 2nd order centered difference approximations
    indices = np.arange(self.shape[0])
    indices = {k: np.roll(indices, -1*int(k)) for k in ("-1","+1")}
    # Compute differential quantities
    dx, dy = [self._compute_diff(z, indices) for z in [x, y]]
    if (self.method == "jac"):
      # Jacobian-based sampling method
      norm = self._compute_norm_jac(dx, dy, indices)
    elif (self.method == "source"):
      # Source-based sampling method
      norm = self._compute_norm_source(dx, dy)
    # Limit the sampling domain
    if (self.xlim is not None):
      idx = np.where((x >= self.xlim[0]) * (x <= self.xlim[1]))[0]
      x, norm = x[idx], norm[idx]
    # Sampling
    return self._combined_sampling(x, norm, nb_pts)

  # Numerical differentiation
  # -----------------------------------
  def _compute_norm_source(self, dx, dy):
    # Compute the norm of the source terms
    return np.linalg.norm(dy/dx, axis=1)

  def _compute_norm_jac(self, dx, dy, indices):
    df = self._compute_diff(dy/dx, indices)
    dy_min = np.amin(dy[np.nonzero(dy)])*1.e-2
    # Compute Jacobian
    n, d = self.shape
    J = np.divide(
      x1=df.reshape(n,d,1),
      x2=(dy + dy_min).reshape(n,1,d)
    )
    dy_zeros = np.where(~dy.astype(bool))
    J[dy_zeros[0],:,dy_zeros[1]] = 0.0
    # Compute the norm of the Jacobian
    return np.linalg.norm(J.reshape(n,d**2), axis=1)

  def _compute_diff(self, x, i):
    dx = x[i["+1"]] - x[i["-1"]]
    dx[0], dx[-1] = dx[1], dx[-2]
    return dx

  # Sampling methods
  # -----------------------------------
  def _combined_sampling(self, x, grad_norm, nb_pts):
    indices = np.arange(len(x))
    # Uniform sampling
    size = int(nb_pts*(1.0-self.grad_split))
    samples = self._uniform_sampling(x, size)
    # Gradient-based sampling
    if (grad_norm.sum() > 0.0):
      p = grad_norm + self.eps
    else:
      p = np.ones_like(grad_norm)
    p /= p.sum()
    while (size > 0):
      grad_samples = np.random.choice(
        indices, replace=False, p=p, size=size
      )
      samples = np.concatenate([samples, grad_samples])
      samples = np.unique(samples)
      size = nb_pts-len(samples)
    # Return sorted indices
    return np.sort(samples)

  def _uniform_sampling(self, x, size):
    # Construct nearest neighbors tree
    tree = KDTree(np.c_[x.ravel()])
    # Uniform sampling
    x_un = np.linspace(np.amin(x), np.amax(x), size+2)[1:-1]
    step = (x_un[1]-x_un[0]) * 0.8
    x_un = x_un + (np.random.rand(size)-0.5) * step
    # Find indices
    samples = tree.query(x=np.split(x_un, size), k=1)[1]
    samples = np.unique(samples)
    return samples
