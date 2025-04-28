from .opnn import OpNN
from pycomet import utils
from .. import elements as nn_elem


class PODOpNN(OpNN):
  """POD-Deep Operator neural network.
  See: https://doi.org/10.1016/j.cma.2022.114778
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    dim,
    lays_name=None,
    lays_propty=None,
    noisy_io=None,
    inference=None,
    masking=False,
    force_siso=False,
    return_latent_space=None,
    *args,
    **kwargs
  ):
    super(PODOpNN, self).__init__(
      dim=dim,
      lays_name=lays_name,
      lays_propty=lays_propty,
      noisy_io=noisy_io,
      inference=inference,
      masking=masking,
      force_siso=force_siso,
      return_latent_space=return_latent_space,
      *args,
      **kwargs
    )
    # Sanity check
    for t in self.active_lays["trunk"]:
      if ((not isinstance(t, nn_elem.FFBlock)) and (t.interpolating is None)):
        utils.raise_err(
          error=ValueError,
          test=f"The 'trunk' block in {self.name} must be a 'FFBlock' with "
               "interpolating basis functions (the last basis is the mean)."
        )
      t.trainable = False

  # Calling
  # ===================================
  def _tensor_product(self, g_u, g_y):
    g_y = [nn_elem.Split([g_y_i.shape[1]-1,1], 1)(g_y_i) for g_y_i in g_y]
    g_y, m = list(zip(*g_y))
    y = super(PODOpNN, self)._tensor_product(g_u, g_y)
    return [y[i]+m[i] for i in range(len(m))]
