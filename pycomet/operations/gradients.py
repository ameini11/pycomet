import numpy as np
import tensorflow as tf

from pycomet import utils


class Gradients(object):
  """
  Container for multiple gradient computations during a TensorFlow session
  using tf.gradients. Each gradient computation is identified by a unique key.

  :attribute grads: A dictionary containing key-gradient object pairs.
  :type grads: dict of str to Gradient

  :attribute built: A list of keys identifying the gradient objects already
                    built to avoid duplicates.
  :type built: list of str
  """

  # Initialization
  # ===================================
  def __init__(self):
    self.grads = {}
    self.built = []

  # Calling
  # ===================================
  def __call__(self, y, x, j=[], i=[], order=1, key=None):
    """
    Compute and return the gradient using the specified key.

    For details about the input parameters,
    see the Gradient object initialization.

    :param key: A unique identifier for this gradient computation.
    :type key: str

    :return: The computed gradient.
    :rtype: tf.Tensor

    This method computes and returns the gradient using the specified key.
    If the key is not found in the list of built gradients, it constructs
    the gradient and stores it for future reference.
    """
    if (key is None):
      utils.raise_err(
        error=ValueError,
        text="Set a key (one for each different gradient call) " \
          "to identify the specific gradient computation."
      )
    if (not any([isinstance(z, (tf.Tensor, np.ndarray)) for z in [y, x]])):
      utils.raise_err(
        error=ValueError,
        text="'y' and 'x' must be a tf.Tensor/np.ndarray " \
          "when computing gradients."
      )
    if (key not in self.built):
      self.build(y, x, j, i, order, key)
    return self.grads[key](y, x)

  def build(self, y, x, j, i, order, key):
    """
    Build and store the gradient for the specified key.

    For details about the input parameters, see the Gradient object 
    initialization.

    :param key: A unique identifier for this gradient computation.
    :type key: str

    This method constructs and stores the gradient for the specified key.
    If the key does not exist in the grads dictionary, it creates a new
    Gradient object and adds it to the dictionary.
    """
    if (key not in self.grads):
      self.grads[key] = Gradient(y, x, j, i, order)
    self.built.append(key)


class Gradient(object):
  """
  Container for a specific gradient computations during a running
  session within TensorFlow.

  :param y: The output tensor to compute the gradient for.
  :type y: tf.Tensor or np.ndarray

  :param x: The input tensor with respect to which to compute the gradient.
  :type x: tf.Tensor or np.ndarray

  :param j: List of column indices indicating which variables of the
            output tensor need gradients.
  :type j: list of int

  :param i: List of column indices indicating which variables of the
            input tensor need gradients.
  :type i: list of int

  :param order: The order of the gradient (default is 1).
  :type order: int
  """

  # Initialization
  # ===================================
  def __init__(self, y, x, j, i, order):
    # Tensor dimensions
    self.dim_y, self.dim_x = tf.nest.map_structure(
      lambda x: x.shape[1], [y, x]
    )
    # Gradient indices
    self.build_indices(j, i)
    # Gradient order
    self.order = order

  def build_indices(self, j, i):
    """
    Build the indices.

    This method constructs the output and input indices
    for gradient computation. It validates the indices
    for both output and input tensors.

    :param j: List of column indices indicating which variables of the
              output tensor need gradients.
    :type j: list of int

    :param i: List of column indices indicating which variables of the
              input tensor need gradients.
    :type i: list of int
    """
    j, i = list(map(lambda x: tuple(tf.nest.flatten(x)), [j, i]))
    # Output indices
    self.j = tuple(np.arange(self.dim_y)) if (len(j) == 0) else j
    if any([not 0 <= k < self.dim_y for k in self.j]):
      utils.raise_err(
        error=ValueError,
        text=f"Index value j={self.j} is not valid for output tensor."
      )
    # Input indices
    self.i = tuple(np.arange(self.dim_x)) if (len(i) == 0) else i
    if any([not 0 <= k < self.dim_x for k in self.i]):
      utils.raise_err(
        error=ValueError,
        text=f"Index value i={self.i} is not valid for input tensor."
      )

  # Calling
  # ===================================
  def __call__(self, y, x):
    """
    Compute and return the gradient.

    This method computes and returns the gradient
    based on the provided output and input tensors.
    It respects the specified gradient order and variable indices.

    :param y: The output tensor.
    :type y: tf.Tensor

    :param x: The input tensor.
    :type x: tf.Tensor

    :return: The computed gradient tensor.
    :rtype y: tf.Tensor
    """
    if (self.order == 0):
      dydx = y
    else:
      dydx = []
      for k in self.j:
        dydx_k = self.compute_grad(
          y=tf.gather(y, k, axis=1),
          x=x,
          order=self.order
        )
        dydx.append(dydx_k)
      dydx = [tf.split(g, self.dim_x, axis=1) for g in dydx]
      dydx = [tf.concat(x, 1) for x in list(zip(*dydx))]
    dydx = [tf.gather(g, self.j, axis=1) for g in dydx]
    return utils.squeeze_list([dydx[k] for k in self.i])

  def compute_grad(self, y, x, order):
    """
    Compute and return the gradient using tf.gradients.

    This method computes the gradient using TensorFlow's tf.gradients
    function. It calculates gradients up to the specified order and
    returns the resulting gradient tensor.

    :param y: The output tensor.
    :type y: tf.Tensor

    :param x: The input tensor.
    :type x: tf.Tensor

    :param order: The order of the gradient.
    :type order: int

    :return: The computed gradient tensor.
    :rtype: tf.Tensor
    """
    if (order == 0):
      return y
    else:
      dydx = tf.gradients(y, x)[-1]
      return self.compute_grad(dydx, x, order=order-1)
