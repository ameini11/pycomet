import os
import random
import numpy as np
import tensorflow as tf

from pycomet import operations as ops


def stratified_sampling(
  x,
  strata
):
  """
  Perform stratified sampling technique.

  This function performs stratified sampling on an input array `x` by
  dividing it into `strata` subgroups, selecting one element randomly from
  each subgroup, and returning the remaining elements as `x_2`. The selected
  elements form `x_1`.

  :param x: The input array for stratified sampling.
  :type x: numpy.ndarray

  :param strata: The number of strata for sampling.
  :type strata: int

  :return: A tuple containing two numpy arrays, (x_2, x_1).
  :rtype: tuple[numpy.ndarray, numpy.ndarray]

  It's important to note that this function assumes the input array `x` is
  sufficiently large and diverse to be divided into `strata` subgroups.
  The function also ensures that there is no intersection between the
  elements in `x_1` and `x_2`.
  """
  x_split = np.array_split(x, strata)
  x_1 = np.array([np.random.choice(i) for i in x_split])
  x_2 = np.setdiff1d(x, x_1)
  assert np.intersect1d(x_1, x_2).size == 0
  return x_2, x_1

def set_random_seed(
  seed=None
):
  """
  Set random number generator seeds for reproducibility.

  :param seed: An integer seed for random number generators.
  :type seed: int or None

  This function sets the seed for Python's built-in random module, NumPy,
  TensorFlow, and ensures deterministic operations. It's essential for
  achieving reproducible results in data processing and machine learning
  tasks. If `seed` is provided, all random generators will use the same seed.
  """
  if (seed is not None):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def sample_dataset(
  dataset,
  nb_samples=None,
  max_samples=None
):
  dataset = tf.nest.map_structure(ops.to_tensor, dataset)
  if (nb_samples is not None):
    nb_samples = ops.get_cardinality(dataset)
  indices = np.random.permutation(nb_samples)
  if ((max_samples is not None) and (nb_samples > max_samples)):
    indices = np.random.choice(indices, max_samples, replace=False)
  return tf.nest.map_structure(
    lambda z: tf.gather(z, indices, axis=0), dataset
  )
