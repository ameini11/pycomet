import numpy as np


def ratio_normal_approx(mean, std):
  """
  Normal approximation to the distribution of the ratio of two independent 
  normal random variables.
  > See:
    https://link.springer.com/article/10.1007/s00362-012-0429-2 (Eq. 7-8)
  """
  rmean, deltas = mean[0]/mean[1], std/mean
  rstd = np.sqrt(rmean**2 * (deltas[0]**2 + deltas[1]**2))
  return rmean, rstd
