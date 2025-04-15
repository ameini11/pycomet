import copy
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pyDOE import lhs
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Sampler(object):

  def __init__(
    self,
    X,
    cols,
    xform=None,
    conds_phys=None,
    bounds_phys=None,
    bounds_only_phys=False,
    bounds_extra_perc=0.0,
    bounds_round=False,
    bounds_save=False,
    kde_samples_perc=0.0,
    kde_bandwidth=0.01,
    kde_kernel="gaussian",
    noise_std=None,
    scaling="std",
    saving_path="./"
  ):
    # Data
    self.X = np.asarray(X)
    self.cols = np.asarray(cols)
    # Path
    self.saving_path = saving_path
    # Transformation
    self.xform = xform
    # Scaling
    self.scaler = self.construct_scaler(method=scaling)
    # Bounds
    self.bounds_round = bounds_round
    self.bounds = self.compute_bounds(
      bounds_phys=bounds_phys,
      bounds_only_phys=bounds_only_phys,
      extra_perc=bounds_extra_perc
    )
    if bounds_save:
      self.save_bounds(bounds_phys)
    # Physical conditions
    self.conds_phys = conds_phys
    # KDE sampler
    self.kde_samples_perc = kde_samples_perc
    if (self.kde_samples_perc > 0.0):
      self.kde = self.construct_kde(
        bandwidth=kde_bandwidth,
        kernel=kde_kernel
      )
    # Noisy samples
    self.noise_std = noise_std

  # Scaling
  # -----------------------------------
  def construct_scaler(
    self,
    method
  ):
    X = copy.deepcopy(self.X)
    if (self.xform is not None):
      X = self.xform["dir"](X)
    if (method == "std"):
      scaler = StandardScaler().fit(X)
    else:
      scaler = MinMaxScaler().fit(X)
    del X
    return scaler

  # Bounds
  # -----------------------------------
  def compute_bounds(
    self,
    bounds_phys=None,
    bounds_only_phys=False,
    extra_perc=0.1
  ):
    # Minimum/maximum values
    xmin = np.amin(self.X, axis=0)
    xmin *= (1.0 - np.sign(xmin)*extra_perc)
    xmax = np.amax(self.X, axis=0)
    xmax *= (1.0 + np.sign(xmax)*extra_perc)
    bounds = np.vstack([xmin, xmax])
    # Check provided physical bounds
    if (bounds_phys is not None):
      if isinstance(bounds_phys, dict):
        bounds_phys = np.array(list(bounds_phys.values())).T
      if bounds_only_phys:
        bounds = bounds_phys
      else:
        bounds[0] = np.minimum(bounds[0], bounds_phys[0])
        bounds[1] = np.maximum(bounds[1], bounds_phys[1])
    # Round bounds
    if self.bounds_round:
      order = np.floor(np.log10(np.absolute(bounds)))
      magn = np.power(10, order)
      bounds[0] = np.floor(bounds[0]/magn[0]) * magn[0]
      bounds[1] = np.ceil(bounds[1]/magn[1]) * magn[1]
    # Check bounds
    if any(bounds[0] > bounds[1]):
      raise ValueError(
        "The bounds minimum values are greater than maximum ones."
      )
    return bounds

  def save_bounds(
    self,
    bounds_phys
  ):
    df = pd.DataFrame(self.bounds, columns=list(bounds_phys.keys()))
    df.index = ["Min", "Max"]
    self.bounds_df = df.T
    self.bounds_df.to_csv(
      self.saving_path + "/bounds.csv", float_format="%.2e"
    )

  # Sampling strategies
  # -----------------------------------
  def sample_lhs_kde(
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
        self.iterative_sample(self.sample_lhs, nb_samples_lhs)
      )
    # KDE
    if (nb_samples_kde > 0):
      samples.append(
        self.iterative_sample(self.sample_kde, nb_samples_kde)
      )
    if (len(samples) > 0):
      return np.vstack(samples)
    else:
      return samples

  def iterative_sample(
    self,
    sampling,
    nb_samples,
    **kwargs
  ):
    samples = []
    while (nb_samples > 0):
      samples_i = sampling(
        nb_samples=nb_samples,
        **kwargs
      )
      # Add noise
      if (self.noise_std is not None):
        samples_i = np.random.normal(
          loc=samples_i, scale=np.absolute(self.noise_std*samples_i)
        )
      samples_i, nb_samples = self.check_samples(samples_i)
      samples.append(samples_i)
    return np.vstack(samples)

  def check_samples(self, samples):
    n = len(samples)
    # Physical conditions
    if (self.conds_phys is not None):
      for cond in self.conds_phys:
        samples = cond(samples, self.bounds)
    # Bounds
    cond = (samples >= self.bounds[0]) * (samples <= self.bounds[1])
    cond = np.product(cond, axis=1)
    idx = np.where(cond)[0]
    samples = samples[idx]
    # Return
    return samples, n - len(samples)

  # Latin hypercube sampling (LHS)
  # -----------------------------------
  def sample_lhs(self, nb_samples):
    dim = self.bounds.shape[1]
    # Transform bounds
    bounds = copy.deepcopy(self.bounds)
    if (self.xform is not None):
      bounds = self.xform["dir"](bounds)
    # Contructing design matrix
    samples = lhs(dim, nb_samples)
    samples = samples * (bounds[1] - bounds[0]) + bounds[0]
    # Anti-transform bounds
    if (self.xform is not None):
      samples = self.xform["inv"](samples)
    # Returning initial conditions
    return samples

  # Kernel density estimation (KDE)
  # -----------------------------------
  def construct_kde(
    self,
    bandwidth=0.01,
    kernel="gaussian"
  ):
    X = copy.deepcopy(self.X)
    if (self.xform is not None):
      X = self.xform["dir"](X)
    if (self.scaler is not None):
      X = self.scaler.transform(X)
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(X)
    del X
    return kde

  def sample_kde(
    self,
    nb_samples
  ):
    samples = self.kde.sample(nb_samples)
    if (self.scaler is not None):
      samples = self.scaler.inverse_transform(samples)
    if (self.xform is not None):
      samples = self.xform["inv"](samples)
    if (self.bounds is not None):
      samples = np.absolute(samples - self.bounds[0]) + self.bounds[0]
      samples = -np.absolute(-samples + self.bounds[1]) + self.bounds[1]
    return samples

# Enforced physical conditions
# =====================================
def get_conditions(
  bounded_qoi,
  sampled_qoi
):
  # Get indices for bounded and sampled variables
  bqoi_idx = {k: i for (i, k) in enumerate(bounded_qoi)}
  sqoi_idx = {k: i for (i, k) in enumerate(sampled_qoi)}
  # Define functions to check if sample respects physical conditions
  # > Differential temperature
  # -------------------------
  def _cond_on_temp(samples, bounds):
    # Extract temperatures
    Th = np.take(samples, sqoi_idx["Th"], axis=1)
    Te = np.take(samples, sqoi_idx["Te"], axis=1)
    # Compute differential temperature
    dT = Th - Te
    # Update samples
    if ("dT" in list(sqoi_idx.keys())):
      samples[:,sqoi_idx["dT"]] = dT
    # Verify condition
    dT_min = bounds[0,bqoi_idx["dT"]]
    dT_max = bounds[1,bqoi_idx["dT"]]
    cond = (dT >= dT_min) * (dT <= dT_max)
    # Return good samples only
    return _extract_samples(samples, cond)
  # Return all the conditions that must be respected
  # -------------------------
  return [_cond_on_temp]

def _extract_samples(samples, cond):
  idx = np.where(cond)[0]
  return samples[idx]