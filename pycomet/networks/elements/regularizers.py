import os
import h5py
import tensorflow as tf
import tensorflow.python.keras.regularizers as reg

from pycomet import utils
from pycomet import backend as bkd
from pycomet.config import get_config
_CFG = get_config()


def get(
  name="l1+l2",
  l1=0,
  l2=0,
  model_name=None,
  weights_path=None
):
  if (name == "l1"):
    return reg.L1(l1=l1)
  elif (name == "l2"):
    return reg.L2(l2=l2)
  elif (name == "l1+l2"):
    return reg.L1L2(l1=l1, l2=l2)
  elif (name == "fine_tuning"):
    return FineTuningRegularizer(
      l1=l1, l2=l2, model_name=model_name, weights_path=weights_path
    )


@tf.keras.utils.register_keras_serializable(
  package=_CFG.name, name='FineTuningRegularizer'
)
class FineTuningRegularizer(reg.L1L2):
  """Transfer learning technique for fine-tuning.
  See:
    Xuhong Li, Yves Grandvalet, Franck Davoine
    'A baseline regularization scheme for transfer
      learning with convolutional neural networks'
    https://doi.org/10.1016/j.patcog.2019.107049
  """

  # Initialization
  # ===================================
  def __init__(
    self,
    l1=0,
    l2=0,
    model_name=None,
    weights_path=None,
    *args,
    **kwargs
  ):
    super(FineTuningRegularizer, self).__init__(l1=l1, l2=l2, *args, **kwargs)
    if ((weights_path is None) and (model_name is not None)):
      self.weights_path = f'./saving/{model_name}/model_weights_clean.h5'
    else:
      self.weights_path = weights_path
    if (self.weights_path is not None):
      self.weights_path = os.path.abspath(str(self.weights_path))
      if (not h5py.is_hdf5(self.weights_path)):
        utils.raise_err(
          ValueError,
          "The 'weights_path' argument of 'FineTuningRegularizer' " \
          f"has to be an hdf5 file. Check '{self.weights_path}'"
        )
    else:
      utils.warning(
        "'FineTuningRegularizer' is not given any starting " \
        "point weights (see 'weights_path')."
      )
    self.sp_weights = utils.H5DataSet(self.weights_path).data
    self.sp_weights = tf.nest.map_structure(
      lambda x: tf.convert_to_tensor(x, dtype=bkd.get_floatx()),
      self.sp_weights
    )

  # Configuration update
  # ===================================
  def get_config(self):
    config = {
      'weights_path': self.weights_path
    }
    base_config = super(FineTuningRegularizer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  # Calling
  # ===================================
  def __call__(self, x):
    if ((self.sp_weights is not None) and (x.name in self.sp_weights)):
      x = x - self.sp_weights[x.name]
    return super(FineTuningRegularizer, self).__call__(x)
