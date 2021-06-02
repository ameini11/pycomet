import tensorflow as tf

from prode.net.weighted_loss.adapter import LossWeightsAdapter
from prode import utils


@tf.keras.utils.register_keras_serializable(package='PrODE', name='ConstantAdapter')
class ConstantAdapter(LossWeightsAdapter):
    """Base class for adaptive loss weights techniques."""

    # Initialization
    ###########################################################################
    def __init__(
        self,
        data_ids,
        weights=None,
        freq=1,
        name='constant_adapter'):
        super(ConstantAdapter, self).__init__(
            data_ids=data_ids,
            weights=weights,
            freq=freq,
            name=name)
        