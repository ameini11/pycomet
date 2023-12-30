import numpy as np
import pandas as pd

from pyDOE import lhs
from pycomet import utils
from .sampler import Sampler
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


class PhysicalSpace(Sampler):

  # Initialization
  # ===================================
  def __init__(
    self,
    data,
    cols,
    xform=None,
    conds_phys=None,
    bounds_phys=None,
    bounds_only_phys=False,
    bounds_extra_perc=0.0,
    round_bounds=False,
    kde_samples_perc=0.0,
    kde_bandwidth=0.01,
    kde_kernel='gaussian',
    noise_std=None,
    scaling='std',
    saving=False,
    saving_path='./',
    name=None,
    *args,
    **kwargs
  ):
    super(PhysicalSpace, self).__init__(
      name=name,
      *args,
      **kwargs
    )
    # Data
    self.data = np.array(data)
    self.cols = np.array(cols)
    # Path
    self.saving_path = saving_path
    # Transformation
    self.xform = xform
    # Scaling
    self.scaler = self._construct_scaler(
      data=self.data,
      method=scaling
    )
    # Bounds
    self.round_bounds = round_bounds
    self.bounds = self._compute_bounds(
      bounds_phys=bounds_phys,
      bounds_only_phys=bounds_only_phys,
      extra_perc=bounds_extra_perc
    )
    if saving:
      self._save_bounds(bounds_phys)
    # Physical conditions
    self.conds_phys = conds_phys
    # KDE sampler
    self.kde_samples_perc = kde_samples_perc
    if (self.kde_samples_perc > 0.0):
      self.kde = self._construct_kde(
        bandwidth=kde_bandwidth,
        kernel=kde_kernel
      )
    # Noisy samples
    self.noise_std = noise_std

  # Stats
  # ===================================
  def compute_eigenvalues(
    self,
    data
  ):
    if (self.xform is not None):
      data = self.xform['dir'](data)
    if (self.scaler is not None):
      data = self.scaler.transform(data)
    u, s, vh = np.linalg.svd(data, full_matrices=False)
    l = s**2 / np.sum(s**2)
    return l

  def compute_covariance_entropy(
    self
  ):
    cov = self.clusterer.covariances_
    cov_diag = [np.diag(c)/np.sum(np.diag(c)) for c in cov]
    cov_diag_entr = np.array([entropy(c) for c in cov_diag])
    return cov_diag_entr

  # Bounds and scaling
  # ===================================
  def _construct_scaler(
    self,
    data,
    method
  ):
    if (self.xform is not None):
      data = self.xform['dir'](data)
    if (method == 'std'):
      return StandardScaler().fit(data)
    else:
      return MinMaxScaler().fit(data)

  def _compute_bounds(
    self,
    bounds_phys=None,
    bounds_only_phys=False,
    extra_perc=0.1
  ):
    scaler = MinMaxScaler().fit(self.data)
    # Set minimum values
    min_values = []
    for v in scaler.data_min_:
      if (v >= 0):
        min_values.append(v*(1.0-extra_perc))
      else:
        min_values.append(v*(1.0+extra_perc))
    # Set maximum values
    max_values = []
    for v in scaler.data_max_:
      if (v >= 0):
        max_values.append(v*(1.0+extra_perc))
      else:
        max_values.append(v*(1.0-extra_perc))
    bounds = np.stack([np.array(min_values), np.array(max_values)], 0)
    # Check physical bounds
    if (bounds_phys is not None):
      if isinstance(bounds_phys, dict):
        bounds_phys = np.array(list(bounds_phys.values())).T
      if bounds_only_phys:
        bounds = bounds_phys
      else:
        d = bounds_phys.shape[1]
        bounds[0,:d] = np.maximum(bounds[0,:d], bounds_phys[0])
        bounds[1,:d] = np.minimum(bounds[1,:d], bounds_phys[1])
    # Round bounds
    if self.round_bounds:
      order = np.floor(np.log10(np.absolute(bounds)))
      magn = np.power(10, order)
      bounds[0] = np.floor(bounds[0]/magn[0]) * magn[0]
      bounds[1] = np.ceil(bounds[1]/magn[1]) * magn[1]
    # Check bounds
    if any(bounds[0] > bounds[1]):
      utils.raise_err(
        error=ValueError,
        text="The bounds minimum values are greater than maximum ones."
      )
    return bounds

  def _save_bounds(
    self,
    bounds_phys
  ):
    df = pd.DataFrame(self.bounds, columns=list(bounds_phys.keys()))
    df.index = ['Min', 'Max']
    self.bounds_df = df.T
    self.bounds_df.to_csv(
      self.saving_path + '/bounds.csv', float_format='%.2e'
    )

  # Sampling strategies
  # ===================================
  def sample(
    self,
    nb_samples
  ):
    # Define number of samples
    nb_samples_lhs = int(nb_samples*(1-self.kde_samples_perc))
    nb_samples_kde = nb_samples - nb_samples_lhs
    samples = []
    # LHS
    if (nb_samples_lhs > 0):
      samples.append(
        self._iterative_sample(self._sample_lhs, nb_samples_lhs)
      )
    # KDE
    if (nb_samples_kde > 0):
      samples.append(
        self._iterative_sample(self._sample_kde, nb_samples_kde)
      )
    if (len(samples) > 0):
      return np.vstack(samples)
    else:
      return samples

  def _iterative_sample(
    self,
    sampling,
    nb_samples,
    **kwargs
  ):
    samples = []
    while (nb_samples > 0):
      samples_i = sampling(nb_samples=nb_samples, **kwargs)
      # Add noise
      if (self.noise_std is not None):
        samples_i = np.random.normal(
          loc=samples_i,
          scale=np.absolute(self.noise_std*samples_i)
        )
      samples_i, nb_samples = self._check_samples(samples_i)
      samples.append(samples_i)
    return np.vstack(samples)

  def _check_samples(self, samples):
    n = len(samples)
    # Physical conditions
    if (self.conds_phys is not None):
      for cond in self.conds_phys:
        samples = cond(samples)
    # Bounds
    cond = (samples >= self.bounds[0]) * (samples <= self.bounds[1])
    cond = np.product(cond, axis=1)
    idx = np.where(cond)[0]
    samples = samples[idx]
    # Return
    return samples, n - len(samples)

  # Latin hypercube sampling (LHS)
  # -----------------------------------
  def _sample_lhs(
    self,
    nb_samples
  ):
    dim = self.bounds.shape[1]
    # Transform bounds
    bounds = self.bounds
    if (self.xform is not None):
      bounds = self.xform['dir'](bounds)
    # Contructing design matrix
    x_norm = lhs(dim, nb_samples)
    x = x_norm * (bounds[1] - bounds[0]) + bounds[0]
    # Sorting w.r.t. the projection norm
    proj = np.dot(x_norm, np.ones((dim,1))).squeeze()
    samples = x[proj.argsort()]
    # Anti-transform bounds
    if (self.xform is not None):
      samples = self.xform['inv'](samples)
    # Returning initial conditions
    return samples

  # Kernel density estimation (KDE)
  # -----------------------------------
  def _construct_kde(
    self,
    data,
    bandwidth=0.01,
    kernel='gaussian'
  ):
    if (self.xform is not None):
      data = self.xform['dir'](data)
    if (self.scaler is not None):
      data = self.scaler.transform(data)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(data)
    return kde

  def _sample_kde(
    self,
    nb_samples
  ):
    samples = self.kde.sample(nb_samples)
    if (self.scaler is not None):
      samples = self.scaler.inverse_transform(samples)
    if (self.xform is not None):
      samples = self.xform['inv'](samples)
    if (self.bounds is not None):
      samples = np.absolute(samples - self.bounds[0]) + self.bounds[0]
      samples = -np.absolute(-samples + self.bounds[1]) + self.bounds[1]
    return samples
