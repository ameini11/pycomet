import h5py
import tensorflow as tf
import tensorflow.python.keras.regularizers as tf_reg

from tensorflow.python.keras import backend as K
from pycomet.utilities import utils


def get(name="l1+l2", l1=0., l2=0., weights_path=None):
    return (
        tf_reg.L1(l1=l1)
        if name == "l1"
        else tf_reg.L2(l2=l2)
        if name == "l2"
        else tf_reg.L1L2(l1=l1, l2=l2)
        if name == "l1+l2"
        else FineTuningRegularizer(l1=l1, l2=l2, weights_path=weights_path)
        if name == "fine_tuning"
        else None
    )


# =========================================================================== #
# Fine Tuning Regularizer
# =========================================================================== #
@tf.keras.utils.register_keras_serializable(
    package='PyCOMET', name='FineTuningRegularizer'
)
class FineTuningRegularizer(tf_reg.L1L2):
    """Transfer learning technique for fine-tuning.
    See:
        Xuhong Li, Yves Grandvalet, Franck Davoine
        `A baseline regularization scheme for transfer
            learning with convolutional neural networks`
        https://doi.org/10.1016/j.patcog.2019.107049"""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        l1=0.,
        l2=0.,
        weights_path=None,
        name='fine_tuning_reg'
    ):
        super(FineTuningRegularizer, self).__init__(l1=l1, l2=l2)

        self.name = name
        self.weights_path = weights_path

        if self.weights_path is not None:
            if not h5py.is_hdf5(self.weights_path):
                utils.raise_value_err(
                    "The `weights_path` argument of `FineTuningRegularizer` "
                    f"has to be an hdf5 file. Check `{self.weights_path}`"
                )
        else:
            utils.warning(
                "`FineTuningRegularizer` is not given any starting point "
                "weights (see `weights_path`)."
            )

        self.weights_SP = utils.H5DataSet(self.weights_path).data
        self.weights_SP = tf.nest.map_structure(
            lambda x: tf.convert_to_tensor(x, dtype=K.floatx()),
            self.weights_SP
        )

    # Configuration update
    ###########################################################################
    def get_config(self):
        config = {
            'name': self.name,
            'weights_SP': self.weights_SP
        }
        base_config = super(FineTuningRegularizer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    # Calling
    ###########################################################################
    def __call__(self, x):
        if self.weights_SP is not None and x.name in self.weights_SP:
            x = x - self.weights_SP[x.name]
        return super(FineTuningRegularizer, self).__call__(x)
