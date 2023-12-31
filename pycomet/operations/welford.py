import tensorflow as tf


class Welford(object):
  """
  Implements Welford's algorithm for computing a running mean and
  standard deviation as described at:
  http://www.johndcook.com/standard_deviation.html

  :attribute mean: The current mean.
  :type mean: float

  :attribute std: The current standard deviation.
  :type std: float
  """
  def __init__(self):
    self.count = 0
    self._mean = 0.0
    self._var = 0.0

  def update(self, x=None):
    """
    Update the running mean and variance with a new value.

    :param x: The new value to update the statistics.
    :type x: float

    If `x` is None, no update is performed.
    """
    if (x is None):
      return None
    self.count += 1
    _mean = self._mean + (x - self._mean) / self.count
    _var = self._var + (x - self._mean) * (x - _mean)
    self._mean, self._var = _mean, _var

  @property
  def mean(self):
    """Get the running mean."""
    return self._mean

  @property
  def std(self):
    """Get the running standard deviation."""
    if (self.count == 1):
      return 0.0
    return tf.sqrt(self._var / (self.count - 1))
