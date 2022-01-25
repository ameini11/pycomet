import h5py
import tensorflow as tf
import tensorflow.python.keras.regularizers as tf_reg

from prode.utilities import utils


def get(name="l1+l2", l1=0., l2=0., weights_SP=None):
    return (
        tf_reg.L1(l1=l1)
        if name == "l1"
        else tf_reg.L2(l2=l2)
        if name == "l2"
        else tf_reg.L1L2(l1=l1, l2=l2)
        if name == "l1+l2"
        else FineTuningRegularizer(l1=l1, l2=l2, weights_SP=weights_SP)
        if name == "fine_tuning"
        else None
    )


@tf.keras.utils.register_keras_serializable(
    package='PrODE', name='FineTuningRegularizer'
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
        weights_SP=None,
        name='fine_tuning_reg'
    ):
        super(FineTuningRegularizer, self).__init__(l1=l1, l2=l2)

        self.name = name

        if weights_SP is not None:
            if not h5py.is_hdf5(weights_SP):
                utils.raise_value_err("The `weights_SP` argument of "
                    "`FineTuningRegularizer` has to be an hdf5 file.")
            self.weights_SP = h5py.File(weights_SP,'r')
        else:
            utils.warning(
                "`FineTuningRegularizer` is not given any Starting Point "
                "weights (see `weights_SP`)."
            )
            self.weights_SP = None

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
            x = x - self.weights_SP[x.name][()]
        return super(FineTuningRegularizer, self).__call__(x)
